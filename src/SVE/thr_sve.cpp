#include "thr_sve.h"
#include <omp.h>
#include <sys/cdefs.h>
#include "memory.h"
#include "neigh_list.h"
#include "cluster_neigh.h"
static __always_inline int align_up(int n, int align) {
  return (n + align - 1) & ~(align - 1);
}
namespace LAMMPS_NS{
ThrSVE::ThrSVE(LAMMPS *lmp) : lmp(lmp){
  nthr = omp_get_max_threads();
  nalloc = 0;
  all_force = nullptr;
  all_vatom = nullptr;
  all_eatom = nullptr;
  all_fflag = nullptr;
  lmp->memory->create(eng_virial, nthr, "ThrSVE:virial buffer");
  lmp->memory->create(all_pvector, nthr, "ThrSVE:pvector buffer");
}
ThrSVE::~ThrSVE()
{
  lmp->memory->destroy(all_force);
  lmp->memory->destroy(all_fflag);
  lmp->memory->destroy(all_eatom);
  lmp->memory->destroy(all_vatom);
  lmp->memory->destroy(eng_virial);
  lmp->memory->destroy(all_pvector);
  nalloc = 0;
  all_force = nullptr;
  all_vatom = nullptr;
  all_eatom = nullptr;
  all_fflag = nullptr;
  eng_virial = nullptr;
  all_pvector = nullptr;
}

void ThrSVE::allocate(int nall, int eflag_atom, int vflag_atom) {
  if (nall > nalloc) {
    int nalloc_new = align_up(nall * 1.5, ATOMS_PER_PAGE);
    lmp->memory->destroy(all_force);
    lmp->memory->destroy(all_fflag);
    lmp->memory->destroy(all_eatom);
    lmp->memory->destroy(all_vatom);
    lmp->memory->create(all_force, nalloc_new * nthr, "ThrSVE:force buffer");
    lmp->memory->create(all_fflag, nalloc_new/ATOMS_PER_PAGE * nthr, "ThrSVE:force flag");
    if (eflag_atom || all_eatom)
      lmp->memory->create(all_eatom, nalloc_new * nthr, "ThrSVE:eatom buffer");
    if (vflag_atom || all_vatom)
      lmp->memory->create(all_vatom, nalloc_new * nthr, "ThrSVE:vatom buffer");
    nalloc = nalloc_new;
  }
}
void ThrSVE::set_range(int &start, int &end, int ithr, int n) {
  int perthr = n / nthr;
  int remthr = n % nthr;
  start = perthr * ithr + std::min(remthr, ithr);
  end = start + perthr + (ithr < remthr);
}

template<int MOL>
void ThrSVE::inspect_cluster(int ithr, NeighList *list) {
  
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh_inner;
  ClusterNeighEntry<!MOL> **firstneigh = (ClusterNeighEntry<!MOL> **)list->firstneigh_inner;

  int icnum = align_up(inum, CLUSTERSIZE) / CLUSTERSIZE;
  int iclo, ichi;
  set_range(iclo, ichi, ithr, icnum);
  
  bool *fflag = fflag_thr(ithr);
  int npage = nalloc / ATOMS_PER_PAGE;

  /* clear all fflag first*/
  for (int i = 0; i < npage; i ++) {
    fflag[i] = false;
  }

  /* mark updated fflag*/
  for (int ic = iclo; ic < ichi; ic ++) {
    int iilo = ic * CLUSTERSIZE;
    int iihi = std::min(iilo + CLUSTERSIZE, inum);
    for (int ii = iilo; ii < iihi; ii ++) {
      int i = ilist[ii];
      /* center atoms are updated*/
      fflag[i / ATOMS_PER_PAGE] = true;
      
    }
    /* iterate over the neighbor list to mark other atoms */
    ClusterNeighEntry<!MOL> *jlist = firstneigh[ic];
    int jnum = numneigh[ic];
    for (int jj = 0; jj < jnum; jj ++) {
      int j = jlist[jj].j;
      fflag[j / ATOMS_PER_PAGE] = true;
    }
  }
}
template void ThrSVE::inspect_cluster<0>(int ithr, NeighList *list);
template void ThrSVE::inspect_cluster<1>(int ithr, NeighList *list);

void ThrSVE::inspect_no_cluster(int ithr, NeighList *list) {
  
  int inum = list->inum;
  int *ilist = list->ilist;
  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;

  int iclo, ichi;
  set_range(iclo, ichi, ithr, inum);

  bool *fflag = fflag_thr(ithr);
  int npage = nalloc / ATOMS_PER_PAGE;
  /* clear all fflag first*/
  for (int i = 0; i < npage; i ++) {
    fflag[i] = false;
  }
  for (int ic = iclo; ic < ichi; ic ++) {      
    int i = ilist[ic];
    fflag[i / ATOMS_PER_PAGE] = true;
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj ++) {
      int j = jlist[jj];
      fflag[j / ATOMS_PER_PAGE] = true;
    }
  }
}

void ThrSVE::init_pair(int ithr, int nall, int eflag_atom, int vflag_atom, int vflag_fdotr){
  auto fflag = fflag_thr(ithr);
  double (*f)[3] = force_thr(ithr);
  int npage = align_up(nall, ATOMS_PER_PAGE) / ATOMS_PER_PAGE;
  for (int ip = 0; ip < npage; ip ++) {
    if (fflag[ip]) {
      int first = ip * ATOMS_PER_PAGE;
      for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
        f[first+i][0] = 0;
        f[first+i][1] = 0;
        f[first+i][2] = 0;
      }
    }
  }
  if (vflag_atom) {
    double (*vatom)[6] = vatom_thr(ithr);
    for (int ip = 0; ip < npage; ip ++) {
      if (fflag[ip]) {
        int first = ip * ATOMS_PER_PAGE;
        for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
          vatom[first+i][0] = 0;
          vatom[first+i][1] = 0;
          vatom[first+i][2] = 0;
          vatom[first+i][3] = 0;
          vatom[first+i][4] = 0;
          vatom[first+i][5] = 0;
        }
      }
    }
  }
  if (eflag_atom) {
    double *eatom = eatom_thr(ithr);
    for (int ip = 0; ip < npage; ip ++) {
      if (fflag[ip]) {
        int first = ip * ATOMS_PER_PAGE;
        for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
          eatom[first+i] = 0;
        }
      }
    }
  }
  for (int i = 0; i < 8; i ++)
    eng_virial[ithr][i] = 0;

  //////for reaxff//////
  for (int i = 0; i < 16; i++)
    all_pvector[ithr][i] = 0;
}
template<int VFLAG_FDOTR>
void ThrSVE::reduce_pair(Atom *atom, Pair *pair, int ithr, int nthr, int eflag_atom, int vflag_atom){
  int nall = atom->nlocal + atom->nghost;
  
  int iplo, iphi;
  int npage = align_up(nall, ATOMS_PER_PAGE) / ATOMS_PER_PAGE;
  set_range(iplo, iphi, ithr, npage);
  
  double (*f)[3] = (double (*)[3])atom->f[0];
  double (*x)[3] = (double (*)[3])atom->x[0];

  double (*t)[3] = nullptr;
  double *virial = eng_virial_thr(ithr);
  for (int ip = iplo; ip < iphi; ip ++) {
    int first = ip * ATOMS_PER_PAGE;
    for (int jthr = 0; jthr < nthr; jthr ++) {
      if (fflag_thr(jthr)[ip]) {
        double (*frep)[3] = force_thr(jthr);
        for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
          f[first+i][0] += frep[first+i][0];
          f[first+i][1] += frep[first+i][1];
          f[first+i][2] += frep[first+i][2];
        }
      }
    }
    if (VFLAG_FDOTR) {
      for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
        virial[0] += f[first+i][0] * x[first+i][0];
        virial[1] += f[first+i][1] * x[first+i][1];
        virial[2] += f[first+i][2] * x[first+i][2];
        virial[3] += f[first+i][1] * x[first+i][0];
        virial[4] += f[first+i][2] * x[first+i][0];
        virial[5] += f[first+i][2] * x[first+i][1];
      }
    }
  }
  if (vflag_atom) {
    double (*vatom)[6] = vatom_thr(ithr);
    for (int ip = iplo; ip < iphi; ip ++) {
      for (int jthr = 0; jthr < nthr; jthr ++) {
        if (fflag_thr(jthr)[ip]) {
          double (*vrep)[6] = vatom_thr(jthr);
          int first = ip * ATOMS_PER_PAGE;
          for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
            vatom[first+i][0] += vrep[first+i][0];
            vatom[first+i][1] += vrep[first+i][1];
            vatom[first+i][2] += vrep[first+i][2];
            vatom[first+i][3] += vrep[first+i][3];
            vatom[first+i][4] += vrep[first+i][4];
            vatom[first+i][5] += vrep[first+i][5];
          }
        }
      }
    }
  }
  if (eflag_atom) {
    double *eatom = eatom_thr(ithr);
    for (int ip = iplo; ip < iphi; ip ++) {
      for (int jthr = 0; jthr < nthr; jthr ++) {
        if (fflag_thr(jthr)[ip]) {
          double (*erep) = eatom_thr(jthr);
          int first = ip * ATOMS_PER_PAGE;
          for (int i = 0; i < ATOMS_PER_PAGE; i ++) {
            eatom[first+i] += erep[first+i];
          }
        }
      }
    }
  }
}
void ThrSVE::reduce_scalar(Pair *pair, int eflag_global, int vflag_global) {
  if (eflag_global)
    for (int ithr = 0; ithr < nthr; ithr ++) {
      pair->eng_vdwl += eng_virial_thr(ithr)[6];
      pair->eng_coul += eng_virial_thr(ithr)[7];
    }
  if (vflag_global)
    for (int ithr = 0; ithr < nthr; ithr ++) {
      for (int id = 0; id < 6; id ++)
        pair->virial[id] += eng_virial_thr(ithr)[id];
    }

  //////for airebo//////
  for(int ithr = 0; ithr < nthr; ithr++)
  {
    for(int id = 0; id < pair->nextra; id++)
    {
      pair->pvector[id] += pvector_thr(ithr)[id];
    }
  }
}
template void ThrSVE::reduce_pair<0>(Atom *atom, Pair *pair, int ithr, int nthr, int eflag_atom, int vflag_atom);
template void ThrSVE::reduce_pair<1>(Atom *atom, Pair *pair, int ithr, int nthr, int eflag_atom, int vflag_atom);
}
