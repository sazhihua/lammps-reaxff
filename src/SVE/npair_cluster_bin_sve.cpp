/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "npair_cluster_bin_sve.h"

#include "atom.h"
#include "atom_vec.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "molecule.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "memory.h"
#include "nstencil.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <algorithm>

using namespace LAMMPS_NS;
using namespace NeighConst;
struct hilbert_gen {
  int (*h2p)[3] = nullptr;
  int np = 0;
  void hilbertC(int s, int x, int y, int z, int dx, int dy, int dz, int dx2, int dy2, int dz2, int dx3, int dy3, int dz3)
  {
      if(s==1)
      {
          h2p[np][0] = x;
          h2p[np][1] = y;
          h2p[np][2] = z;
          np++;
      }
      else
      {
          s/=2;
          if(dx<0) x-=s*dx;
          if(dy<0) y-=s*dy;
          if(dz<0) z-=s*dz;
          if(dx2<0) x-=s*dx2;
          if(dy2<0) y-=s*dy2;
          if(dz2<0) z-=s*dz2;
          if(dx3<0) x-=s*dx3;
          if(dy3<0) y-=s*dy3;
          if(dz3<0) z-=s*dz3;
          hilbertC(s, x, y, z, dx2, dy2, dz2, dx3, dy3, dz3, dx, dy, dz);
          hilbertC(s, x+s*dx, y+s*dy, z+s*dz, dx3, dy3, dz3, dx, dy, dz, dx2, dy2, dz2);
          hilbertC(s, x+s*dx+s*dx2, y+s*dy+s*dy2, z+s*dz+s*dz2, dx3, dy3, dz3, dx, dy, dz, dx2, dy2, dz2);
          hilbertC(s, x+s*dx2, y+s*dy2, z+s*dz2, -dx, -dy, -dz, -dx2, -dy2, -dz2, dx3, dy3, dz3);
          hilbertC(s, x+s*dx2+s*dx3, y+s*dy2+s*dy3, z+s*dz2+s*dz3, -dx, -dy, -dz, -dx2, -dy2, -dz2, dx3, dy3, dz3);
          hilbertC(s, x+s*dx+s*dx2+s*dx3, y+s*dy+s*dy2+s*dy3, z+s*dz+s*dz2+s*dz3, -dx3, -dy3, -dz3, dx, dy, dz, -dx2, -dy2, -dz2);
          hilbertC(s, x+s*dx+s*dx3, y+s*dy+s*dy3, z+s*dz+s*dz3, -dx3, -dy3, -dz3, dx, dy, dz, -dx2, -dy2, -dz2);
          hilbertC(s, x+s*dx3, y+s*dy3, z+s*dz3, dx2, dy2, dz2, -dx3, -dy3, -dz3, -dx, -dy, -dz);
      }
  }
  void build(int n){
    destroy();
    int edge = 1;
    while (edge < n) edge <<= 1;
    np = 0;
    h2p = new int[edge*edge*edge][3];
    hilbertC(edge,0,0,0,1,0,0,0,1,0,0,0,1);
  }
  void destroy(){
    delete[] h2p;
    h2p = nullptr;
  }
  ~hilbert_gen(){
    destroy();
  }
};

/* ---------------------------------------------------------------------- */
template<int HALF, int NEWTON, int TRI, int SIZE, int ATOMONLY>
NPairClusterBinSVE<HALF, NEWTON, TRI, SIZE, ATOMONLY>::NPairClusterBinSVE(LAMMPS *lmp) : NPair(lmp) {
  firstneigh_cluster = nullptr;
  jnum_cluster = nullptr;
  neigh_buffer = nullptr;
  firstneigh_alloc = 0;
}

template<int HALF, int NEWTON, int TRI, int SIZE, int ATOMONLY>
 	NPairClusterBinSVE<HALF, NEWTON, TRI, SIZE, ATOMONLY>::~NPairClusterBinSVE()
 	{
 	  memory->destroy(jnum_cluster);
 	  memory->destroy(neigh_buffer);
 	  memory->sfree(firstneigh_cluster);
 	}

/* ----------------------------------------------------------------------
   Full:
     binned neighbor list construction for all neighbors
     every neighbor pair appears in list of both atoms i and j
   Half + Newtoff:
     binned neighbor list construction with partial Newton's 3rd law
     each owned atom i checks own bin and other bins in stencil
     pair stored once if i,j are both owned and i < j
     pair stored by me if j is ghost (also stored by proc owning j)
   Half + Newton:
     binned neighbor list construction with full Newton's 3rd law
     each owned atom i checks its own bin and other bins in Newton stencil
     every pair stored exactly once by some processor
------------------------------------------------------------------------- */

template<int HALF, int NEWTON, int TRI, int SIZE, int ATOMONLY>
void NPairClusterBinSVE<HALF, NEWTON, TRI, SIZE, ATOMONLY>::build(NeighList *list)
{
  build_normal(list);
}

template<int HALF, int NEWTON, int TRI, int SIZE, int ATOMONLY>
void NPairClusterBinSVE<HALF, NEWTON, TRI, SIZE, ATOMONLY>::build_normal(NeighList *list)
{
  // constexpr int cluster_size = 8;
  int i, ii, ic, j, jj, jh, k, n, itype, jtype, ibin, bin_start, which, imol, iatom, moltemplate;
  tagint itag, jtag, tagprev;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq, radsum,cut,cutsq;

  const double delta = 0.01 * force->angstrom;

  double **x = atom->x;
  double *radius = atom->radius;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  if (includegroup) nlocal = atom->nfirst;

  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  Molecule **onemols = atom->avec->onemols;
  if (!ATOMONLY) {
    if (molecular == Atom::TEMPLATE)
      moltemplate = 1;
    else
      moltemplate = 0;
  }

  int history = list->history;
  int mask_history = 1 << HISTBITS;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int jlo, jhi;
  int inum = 0, gnum = 0;

  hilbert_gen h;
  h.build(std::max(std::max(mbinx, mbiny), mbinz));

  int *glist, *gbinlo, *gbinhi;
 	int safezone = 8;
  memory->create(glist, nlocal + nghost, "NPairClusterBinSVE:glist");
  memory->create(gbinlo, mbins, "NPairClusterBinSVE:gbinlo");
  memory->create(gbinhi, mbins, "NPairClusterBinSVE:gbinhi");
  memory->grow(jnum_cluster, nlocal + nghost + safezone, "NPairClusterBinSVE:jnum");
  firstneigh_cluster = (decltype(firstneigh_cluster))memory->srealloc((void*)firstneigh_cluster, nlocal + safezone, "NPairClusterBinSVE:firstneigh_cluster");
  memory->grow(neigh_buffer, (nlocal + safezone) * neighbor->oneatom, "NPairClusterBinSVE:neigh_buffer");

  for (int iibin = 0; iibin < mbins - 1; iibin ++) {
    gbinlo[iibin] = gnum;
    for (i = binhead[iibin]; i >= 0; i = bins[i]) {
      glist[gnum ++] = i;
      if (i < nlocal) ilist[inum ++] = i;
    }
    gbinhi[iibin] = gnum;
  }


  int num_clusters = (inum + CLUSTERSIZE - 1) / CLUSTERSIZE;
#pragma omp parallel
{
  std::vector<int> jset(nlocal + nghost, -1);
  std::vector<int> jloc(nlocal + nghost, -1);

#pragma omp for 
  for (ic = 0; ic < num_clusters; ic++) {
    int iis = ic * CLUSTERSIZE;
    int iie = std::min(iis + CLUSTERSIZE, inum);
    firstneigh_cluster[ic] = neigh_buffer + neighbor->oneatom * ic;
    ClusterNeighEntry<ATOMONLY> *jlist_cluster = firstneigh_cluster[ic];
    int nc = 0;
    auto append_cluster = [&](int ii, int j, int special){
      if (jset[j] != ic) {
        jset[j] = ic;
        ClusterNeighEntry<ATOMONLY> &ent = jlist_cluster[nc];
        ent.init(j);
        ent.add_i(ii - iis, special);
        jloc[j] = nc;
        nc ++;
      } else {
        ClusterNeighEntry<ATOMONLY> &ent = jlist_cluster[jloc[j]];
        ent.add_i(ii - iis, special);
      }
      
    };
    for (ii = iis; ii < iie; ii++) {
      i = ilist[ii];
      n = 0;

      itag = tag[i];
      itype = type[i];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      if (!ATOMONLY) {
        if (moltemplate) {
          imol = molindex[i];
          iatom = molatom[i];
          tagprev = tag[i] - iatom - 1;
        }
      }

      ibin = atom2bin[i];
      
      for (k = 0; k < nstencil; k++) {
        jlo = gbinlo[ibin + stencil[k]];
        jhi = gbinhi[ibin + stencil[k]];
        if (HALF && NEWTON && (!TRI)) {
          if (k == 0) {
            while (glist[jlo] <= i) jlo++;
          }
        }

        for (jj = jlo; jj < jhi; jj ++) {
          j = glist[jj];
          if (!HALF) {
            // Full neighbor list
            // only skip i = j
            if (i == j) continue;
          } else if (!NEWTON) {
            // Half neighbor list, newton off
            // only store pair if i < j
            // stores own/own pairs only once
            // stores own/ghost pairs on both procs
            if (j <= i) continue;
          } else if (TRI) {
            // for triclinic, bin stencil is full in all 3 dims
            // must use itag/jtag to eliminate half the I/J interactions
            // cannot use I/J exact coord comparision
            //   b/c transforming orthog -> lambda -> orthog for ghost atoms
            //   with an added PBC offset can shift all 3 coords by epsilon
            if (j <= i) continue;
            if (j >= nlocal) {
              jtag = tag[j];
              if (itag > jtag) {
                if ((itag + jtag) % 2 == 0) continue;
              } else if (itag < jtag) {
                if ((itag + jtag) % 2 == 1) continue;
              } else {
                if (fabs(x[j][2] - ztmp) > delta) {
                  if (x[j][2] < ztmp) continue;
                } else if (fabs(x[j][1] - ytmp) > delta) {
                  if (x[j][1] < ytmp) continue;
                } else {
                  if (x[j][0] < xtmp) continue;
                }
              }
            }
          } else {
            // Half neighbor list, newton on, orthonormal
            // store every pair for every bin in stencil, except for i's bin

            if (k == 0) {
              // if j is owned atom, store it, since j is beyond i in linked list
              // if j is ghost, only store if j coords are "above and to the "right" of i
              if (j >= nlocal) {
                if (x[j][2] < ztmp) continue;
                if (x[j][2] == ztmp) {
                  if (x[j][1] < ytmp) continue;
                  if (x[j][1] == ytmp && x[j][0] < xtmp) continue;
                }
              }
            }
          }

          jtype = type[j];
          //if (exclude && exclusion(i, j, itype, jtype, mask, molecule)) continue;

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;

          if (SIZE) {
            radsum = radius[i] + radius[j];
            cut = radsum + skin;
            cutsq = cut * cut;

            if (ATOMONLY) {
              if (rsq <= cutsq) {
                jh = j;
                if (history && rsq < (radsum * radsum))
                  jh = jh ^ mask_history;
                n++;
                append_cluster(ii, jh, 0);
              }
            } else {
              if (rsq <= cutsq) {
                jh = j;
                if (history && rsq < (radsum * radsum))
                  jh = jh ^ mask_history;

                if (molecular != Atom::ATOMIC) {
                  if (!moltemplate)
                    which = find_special(special[i], nspecial[i], tag[j]);
                  else if (imol >= 0)
                    which = find_special(onemols[imol]->special[iatom], onemols[imol]->nspecial[iatom],
                                         tag[j] - tagprev);
                  else
                    which = 0;
                  if (which == 0) {
                    n++;
                    append_cluster(ii, jh, 0);
                  }
                  else if (domain->minimum_image_check(delx, dely, delz)) {
                    n++;
                    append_cluster(ii, jh, 0);
                  }
                  else if (which > 0) {
                    n++;
                    append_cluster(ii, jh, which);
                  }
                } else {
                  n++;
                  append_cluster(ii, jh, 0);
                }
              }
            }
          } else {
            if (ATOMONLY) {
              if (rsq <= cutneighsq[itype][jtype]) {
                n++;
                append_cluster(ii, j, 0);
              }
            } else {
              if (rsq <= cutneighsq[itype][jtype]) {
                if (molecular != Atom::ATOMIC) {
                  if (!moltemplate)
                    which = find_special(special[i], nspecial[i], tag[j]);
                  else if (imol >= 0)
                    which = find_special(onemols[imol]->special[iatom], onemols[imol]->nspecial[iatom],
                                         tag[j] - tagprev);
                  else
                    which = 0;
                  if (which == 0) {
                    n++;
                    append_cluster(ii, j, 0);
                  } else if (domain->minimum_image_check(delx, dely, delz)) {
                    n++;
                    append_cluster(ii, j, 0);
                  } else if (which > 0) {
                    n++;
                    append_cluster(ii, j, which);
                  }
                } else {
                  n++;
                  append_cluster(ii, j, 0);
                }
              }
            }
          }
        }
      }
      numneigh[i] = n;
    }
    jnum_cluster[ic] = nc;
    if (nc >= neighbor->oneatom) {
      error->one(FLERR, "Neighbor list overflow, boost neigh_modify one");
    }
    while (nc & 3) {
      jlist_cluster[nc++].init(ilist[iis]);
    }
  }
}

  list->inum = inum;
  if (!HALF) list->gnum = 0;
  memory->destroy(glist);
  memory->destroy(gbinlo);
  memory->destroy(gbinhi);
  list->firstneigh_inner = (int**)firstneigh_cluster;
  list->numneigh_inner = jnum_cluster;
}

namespace LAMMPS_NS {
template class NPairClusterBinSVE<0,1,0,0,0>;
template class NPairClusterBinSVE<1,0,0,0,0>;
template class NPairClusterBinSVE<1,1,0,0,0>;
template class NPairClusterBinSVE<1,1,1,0,0>;
template class NPairClusterBinSVE<0,1,0,1,0>;
template class NPairClusterBinSVE<1,0,0,1,0>;
template class NPairClusterBinSVE<1,1,0,1,0>;
template class NPairClusterBinSVE<1,1,1,1,0>;
template class NPairClusterBinSVE<0,1,0,0,1>;
template class NPairClusterBinSVE<1,0,0,0,1>;
template class NPairClusterBinSVE<1,1,0,0,1>;
template class NPairClusterBinSVE<1,1,1,0,1>;
template class NPairClusterBinSVE<0,1,0,1,1>;
template class NPairClusterBinSVE<1,0,0,1,1>;
template class NPairClusterBinSVE<1,1,0,1,1>;
template class NPairClusterBinSVE<1,1,1,1,1>;
}

