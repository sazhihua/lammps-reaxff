// clang-format off
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

/* ----------------------------------------------------------------------
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)

     Hybrid and sub-group capabilities: Ray Shan (Sandia)
------------------------------------------------------------------------- */

#include "fix_qeq_reaxff_sve.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_efield_sve.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "region.h"
#include "respa.h"
#include "text_file_reader.h"
#include "update.h"

#include "pair_reaxff_sve.h"
#include "reaxff_api_sve.h"
#include "arm_sve.h"
#include "sve_util.h"
#include "cluster_neigh.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <omp.h>

// #include "gptl.h"

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double EV_TO_KCAL_PER_MOL = 14.4;
static constexpr double SMALL = 1.0e-14;
static constexpr double QSUMSMALL = 0.00001;

static const char cite_fix_qeq_reaxff[] =
  "fix qeq/reaxff command: doi:10.1016/j.parco.2011.08.005\n\n"
  "@Article{Aktulga12,\n"
  " author = {H. M. Aktulga and J. C. Fogarty and S. A. Pandit and A. Y. Grama},\n"
  " title = {Parallel Reactive Molecular Dynamics: {N}umerical Methods and Algorithmic Techniques},\n"
  " journal = {Parallel Computing},\n"
  " year =    2012,\n"
  " volume =  38,\n"
  " pages =   {245--259}\n"
  "}\n\n";

#define CLUSTERSIZE 8
// #define NOREVERSE

/* ---------------------------------------------------------------------- */

FixQEqReaxFFSVE::FixQEqReaxFFSVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), matvecs(0), pertype_option(nullptr)
{
  scalar_flag = 1;
  extscalar = 0;
  imax = 200;
  maxwarn = 1;

  if ((narg < 8) || (narg > 12)) error->all(FLERR,"Illegal fix qeq/reaxff command");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix qeq/reaxff command");

  swa = utils::numeric(FLERR,arg[4],false,lmp);
  swb = utils::numeric(FLERR,arg[5],false,lmp);
  tolerance = utils::numeric(FLERR,arg[6],false,lmp);
  pertype_option = utils::strdup(arg[7]);

  // dual CG support only available for OPENMP variant
  // check for compatibility is in Fix::post_constructor()

  dual_enabled = 0;

  cg_algo = 1;
  thread0_weight = 0.0;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"dual") == 0) dual_enabled = 1;
    else if (strcmp(arg[iarg],"nowarn") == 0) maxwarn = 0;
    else if (strcmp(arg[iarg],"maxiter") == 0) {
      if (iarg+1 > narg-1)
        error->all(FLERR,"Illegal fix {} command", style);
      imax = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg++;
    } 
    else if (strcmp(arg[iarg],"cg_algo") == 0) {
      if (iarg+1 > narg-1)
        error->all(FLERR,"Illegal fix {} command: missing cg_algo value", style);
      cg_algo = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (cg_algo < 0 || cg_algo > 3) 
        error->all(FLERR,"Illegal cg_algo: 0: original CG, 1: original CG with omp, 2: pipeline CG with omp, 3: pipeline CG with omp and thread0 weight");
      if (cg_algo == 3) {
        if (iarg+2 > narg-1) {
          error->all(FLERR,"Illegal fix {} command: missing thread0_weight after cg_algo 3", style);
        }
        thread0_weight = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        if (thread0_weight < 0.0 || thread0_weight > 1.0)
          error->all(FLERR, "Illegal thread0_weight: 0.0 <= thread0_weight <= 1.0");
        iarg++;
      }
      iarg++;
    } 
    else error->all(FLERR,"Illegal fix {} command", style);
    iarg++;
  }
  shld = nullptr;

  nn = n_cap = 0;
  nmax = 0;
  m_fill = m_cap = 0;
  pack_flag = 0;
  s = nullptr;
  t = nullptr;
  nprev = 4;

  Hdia_inv = nullptr;
  b_s = nullptr;
  chi_field = nullptr;
  b_t = nullptr;
  b_prc = nullptr;
  b_prm = nullptr;

  // CG

  p = nullptr;
  q = nullptr;
  r = nullptr;
  d = nullptr;

  // Pipeline CG
  m = nullptr;
  u = nullptr;
  v = nullptr;
  w = nullptr;
  z = nullptr;
  p_s = nullptr;

  // H matrix

  H.firstnbr = nullptr;
  H.numnbrs = nullptr;
  H.jlist = nullptr;
  H.val = nullptr;

  // for multi-thread
  b_temp = nullptr;

  // dual CG support
  // Update comm sizes for this fix

  if (dual_enabled) comm_forward = comm_reverse = 2;
  else comm_forward = comm_reverse = 1;

  // perform initial allocation of atom-based arrays
  // register with Atom class

  reaxff = dynamic_cast<PairReaxFFSVE *>(force->pair_match("^reaxff",0));

  s_hist = t_hist = nullptr;
  atom->add_callback(Atom::GROW);
}

/* ---------------------------------------------------------------------- */

FixQEqReaxFFSVE::~FixQEqReaxFFSVE()
{
  if (copymode) return;

  delete[] pertype_option;

  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,Atom::GROW);

  memory->destroy(s_hist);
  memory->destroy(t_hist);

  FixQEqReaxFFSVE::deallocate_storage();
  FixQEqReaxFFSVE::deallocate_matrix();

  memory->destroy(shld);

  if (!reaxflag) {
    memory->destroy(chi);
    memory->destroy(eta);
    memory->destroy(gamma);
  }

  // for multi-thread
  memory->destroy(b_temp);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_qeq_reaxff);

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = t_hist[i][j] = 0;

  pertype_parameters(pertype_option);
  if (dual_enabled)
    error->all(FLERR,"Dual keyword only supported with fix qeq/reaxff/omp");
}

/* ---------------------------------------------------------------------- */

int FixQEqReaxFFSVE::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::pertype_parameters(char *arg)
{
  const int nlocal = atom->nlocal;
  const int *mask = atom->mask;
  const int *type = atom->type;

  if (utils::strmatch(arg,"^reaxff")) {
    reaxflag = 1;
    Pair *pair = force->pair_match("^reaxff",0);
    if (!pair) error->all(FLERR,"No reaxff pair style for fix qeq/reaxff");

    int tmp, tmp_all;
    chi = (double *) pair->extract("chi",tmp);
    eta = (double *) pair->extract("eta",tmp);
    gamma = (double *) pair->extract("gamma",tmp);
    if ((chi == nullptr) || (eta == nullptr) || (gamma == nullptr))
      error->all(FLERR, "Fix qeq/reaxff could not extract all QEq parameters from pair reaxff");
    tmp = tmp_all = 0;
    for (int i = 0; i < nlocal; ++i) {
      if (mask[i] & groupbit) {
        if ((chi[type[i]] == 0.0) && (eta[type[i]] == 0.0) && (gamma[type[i]] == 0.0))
          tmp = type[i];
      }
    }
    MPI_Allreduce(&tmp, &tmp_all, 1, MPI_INT, MPI_MAX, world);
    if (tmp_all)
      error->all(FLERR, "No QEq parameters for atom type {} provided by pair reaxff", tmp_all);
    return;
  } else if (utils::strmatch(arg,"^reax/c")) {
    error->all(FLERR, "Fix qeq/reaxff keyword 'reax/c' is obsolete; please use 'reaxff'");
  } else if (platform::file_is_readable(arg)) {
    ; // arg is readable file. will read below
  } else {
    error->all(FLERR, "Unknown fix qeq/reaxff keyword {}", arg);
  }

  reaxflag = 0;

  const int ntypes = atom->ntypes;
  memory->create(chi,ntypes+1,"qeq/reaxff:chi");
  memory->create(eta,ntypes+1,"qeq/reaxff:eta");
  memory->create(gamma,ntypes+1,"qeq/reaxff:gamma");

  if (comm->me == 0) {
    chi[0] = eta[0] = gamma[0] = 0.0;
    try {
      TextFileReader reader(arg,"qeq/reaxff parameter");
      reader.ignore_comments = false;
      for (int i = 1; i <= ntypes; i++) {
        const char *line = reader.next_line();
        if (!line)
          throw TokenizerException("Fix qeq/reaxff: Invalid param file format","");
        ValueTokenizer values(line);

        if (values.count() != 4)
          throw TokenizerException("Fix qeq/reaxff: Incorrect format of param file","");

        int itype = values.next_int();
        if ((itype < 1) || (itype > ntypes))
          throw TokenizerException("Fix qeq/reaxff: invalid atom type in param file",
                                   std::to_string(itype));

        chi[itype] = values.next_double();
        eta[itype] = values.next_double();
        gamma[itype] = values.next_double();
      }
    } catch (std::exception &e) {
      error->one(FLERR,e.what());
    }
  }

  MPI_Bcast(chi,ntypes+1,MPI_DOUBLE,0,world);
  MPI_Bcast(eta,ntypes+1,MPI_DOUBLE,0,world);
  MPI_Bcast(gamma,ntypes+1,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::allocate_storage()
{
  nmax = atom->nmax;

  memory->create(s,nmax,"qeq:s");
  memory->create(t,nmax,"qeq:t");

  memory->create(Hdia_inv,nmax,"qeq:Hdia_inv");
  memory->create(b_s,nmax,"qeq:b_s");
  memory->create(chi_field,nmax,"qeq:chi_field");
  memory->create(b_t,nmax,"qeq:b_t");
  memory->create(b_prc,nmax,"qeq:b_prc");
  memory->create(b_prm,nmax,"qeq:b_prm");

  // dual CG support
  int size = nmax;
  if (dual_enabled) size*= 2;

  memory->create(p,size,"qeq:p");
  memory->create(q,size,"qeq:q");
  memory->create(r,size,"qeq:r");
  memory->create(d,size,"qeq:d");

  // pipeline CG
  memory->create(m,size,"qeq:m");
  memory->create(u,size,"qeq:u");
  memory->create(v,size,"qeq:v");
  memory->create(w,size,"qeq:w");
  memory->create(z,size,"qeq:z");
  memory->create(p_s,size,"qeq:p_s");

  // for multi-thread
  memory->create(b_temp,omp_get_max_threads(),size,"qeq:b_temp");
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::deallocate_storage()
{
  memory->destroy(s);
  memory->destroy(t);

  memory->destroy(Hdia_inv);
  memory->destroy(b_s);
  memory->destroy(b_t);
  memory->destroy(b_prc);
  memory->destroy(b_prm);
  memory->destroy(chi_field);

  memory->destroy(p);
  memory->destroy(q);
  memory->destroy(r);
  memory->destroy(d);

  memory->destroy(m);
  memory->destroy(u);
  memory->destroy(v);
  memory->destroy(w);
  memory->destroy(z);
  memory->destroy(p_s);
  
  // for multi-thread
  memory->destroy(b_temp);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::reallocate_storage()
{
  deallocate_storage();
  allocate_storage();
  init_storage();
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::allocate_matrix()
{
  int i,ii;
  bigint m;

  int mincap;
  double safezone;

  if (reaxflag && reaxff) {
    mincap = reaxff->api->system->mincap;
    safezone = reaxff->api->system->safezone;
  } else {
    mincap = REAX_MIN_CAP;
    safezone = REAX_SAFE_ZONE;
  }

  n_cap = MAX((int)(atom->nlocal * safezone), mincap);

  // determine the total space for the H matrix

  m = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    m += numneigh[i];
  }
  bigint m_cap_big = (bigint)MAX(m * safezone, mincap * REAX_MIN_NBRS);
  if (m_cap_big > MAXSMALLINT)
    error->one(FLERR,"Too many neighbors in fix {}",style);
  m_cap = m_cap_big;

  H.n = n_cap;
  H.m = m_cap;
  memory->create(H.firstnbr,n_cap,"qeq:H.firstnbr");
  memory->create(H.numnbrs,n_cap,"qeq:H.numnbrs");
  memory->create(H.jlist,m_cap,"qeq:H.jlist");
  memory->create(H.val,m_cap,"qeq:H.val");
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::deallocate_matrix()
{
  memory->destroy(H.firstnbr);
  memory->destroy(H.numnbrs);
  memory->destroy(H.jlist);
  memory->destroy(H.val);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::reallocate_matrix()
{
  deallocate_matrix();
  allocate_matrix();
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::init()
{
  if (!atom->q_flag)
    error->all(FLERR,"Fix {} requires atom attribute q", style);

  if (group->count(igroup) == 0)
    error->all(FLERR,"Fix {} group has no atoms", style);

  // compute net charge and print warning if too large

  double qsum_local = 0.0, qsum = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & groupbit)
      qsum_local += atom->q[i];
  }
  MPI_Allreduce(&qsum_local,&qsum,1,MPI_DOUBLE,MPI_SUM,world);

  if ((comm->me == 0) && (fabs(qsum) > QSUMSMALL))
    error->warning(FLERR,"Fix {} group is not charge neutral, net charge = {:.8}" + utils::errorurl(29), style, qsum);

  // get pointer to fix efield if present. there may be at most one instance of fix efield in use.

  efield = nullptr;
  auto fixes = modify->get_fix_by_style("^efield");
  if (fixes.size() == 1) efield = dynamic_cast<FixEfieldSVE *>(fixes.front());
  else if (fixes.size() > 1)
    error->all(FLERR, "There may be only one fix efield instance used with fix {}", style);

  // ensure that fix efield is properly initialized before accessing its data and check some settings
  if (efield) {
    efield->init();
    if (strcmp(update->unit_style,"real") != 0)
      error->all(FLERR,"Must use unit_style real with fix {} and external fields", style);

    if (efield->varflag == FixEfieldSVE::ATOM && efield->pstyle != FixEfieldSVE::ATOM)
      error->all(FLERR,"Atom-style external electric field requires atom-style "
                       "potential variable when used with fix {}", style);
    if (((efield->xstyle != FixEfieldSVE::CONSTANT) && domain->xperiodic) ||
         ((efield->ystyle != FixEfieldSVE::CONSTANT) && domain->yperiodic) ||
         ((efield->zstyle != FixEfieldSVE::CONSTANT) && domain->zperiodic))
      error->all(FLERR,"Must not have electric field component in direction of periodic "
                       "boundary when using charge equilibration with ReaxFFSVE.");
    if (((fabs(efield->ex) > SMALL) && domain->xperiodic) ||
         ((fabs(efield->ey) > SMALL) && domain->yperiodic) ||
         ((fabs(efield->ez) > SMALL) && domain->zperiodic))
      error->all(FLERR,"Must not have electric field component in direction of periodic "
                       "boundary when using charge equilibration with ReaxFFSVE.");
  }

  // we need a half neighbor list w/ Newton off
  // built whenever re-neighboring occurs

  neighbor->add_request(this, NeighConst::REQ_NEWTON_OFF | NeighConst::REQ_CLUSTER);

  init_shielding();
  init_taper();

  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;
}

/* ---------------------------------------------------------------------- */

double FixQEqReaxFFSVE::compute_scalar()
{
  return matvecs/2.0;
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::init_shielding()
{
  int i,j;
  int ntypes;

  ntypes = atom->ntypes;
  if (shld == nullptr)
    memory->create(shld,ntypes+1,ntypes+1,"qeq:shielding");

  for (i = 1; i <= ntypes; ++i)
    for (j = 1; j <= ntypes; ++j)
      shld[i][j] = pow(gamma[i] * gamma[j], -1.5);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::init_taper()
{
  double d7, swa2, swa3, swb2, swb3;

  if (fabs(swa) > 0.01 && comm->me == 0)
    error->warning(FLERR,"Fix qeq/reaxff has non-zero lower Taper radius cutoff");
  if (swb < 0)
    error->all(FLERR, "Fix qeq/reaxff has negative upper Taper radius cutoff");
  else if (swb < 5 && comm->me == 0)
    error->warning(FLERR,"Fix qeq/reaxff has very low Taper radius cutoff");

  d7 = pow(swb - swa, 7);
  swa2 = SQR(swa);
  swa3 = CUBE(swa);
  swb2 = SQR(swb);
  swb3 = CUBE(swb);

  Tap[7] =  20.0 / d7;
  Tap[6] = -70.0 * (swa + swb) / d7;
  Tap[5] =  84.0 * (swa2 + 3.0*swa*swb + swb2) / d7;
  Tap[4] = -35.0 * (swa3 + 9.0*swa2*swb + 9.0*swa*swb2 + swb3) / d7;
  Tap[3] = 140.0 * (swa3*swb + 3.0*swa2*swb2 + swa*swb3) / d7;
  Tap[2] =-210.0 * (swa3*swb2 + swa2*swb3) / d7;
  Tap[1] = 140.0 * swa3 * swb3 / d7;
  Tap[0] = (-35.0*swa3*swb2*swb2 + 21.0*swa2*swb3*swb2 -
            7.0*swa*swb3*swb3 + swb3*swb3*swb) / d7;
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::setup_pre_force(int vflag)
{
  if (reaxff) {
    nn = reaxff->list->inum;
    ilist = reaxff->list->ilist;
    numneigh = reaxff->list->numneigh;
    //firstneigh = reaxff->list->firstneigh;
  } else {
    nn = list->inum;
    ilist = list->ilist;
    //numneigh = list->numneigh;
    //firstneigh = list->firstneigh;
  }

  deallocate_storage();
  allocate_storage();

  init_storage();

  deallocate_matrix();
  allocate_matrix();

  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel < nlevels_respa-1) return;
  setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::min_setup_pre_force(int vflag)
{
  setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::init_storage()
{
  if (efield) get_chi_field();

  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      Hdia_inv[i] = 1. / eta[atom->type[i]];
      b_s[i] = -chi[atom->type[i]];
      if (efield) b_s[i] -= chi_field[i];
      b_t[i] = -1.0;
      b_prc[i] = 0;
      b_prm[i] = 0;
      s[i] = t[i] = 0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::pre_force(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;

  int n = atom->nlocal;

  if (reaxff) {
    nn = reaxff->list->inum;
    ilist = reaxff->list->ilist;
    numneigh = reaxff->list->numneigh;
    //firstneigh = reaxff->list->firstneigh;
  } else {
    nn = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    //firstneigh = list->firstneigh;
  }

  // grow arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) reallocate_storage();
  if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
    reallocate_matrix();

  if (efield) get_chi_field();

  if (omp_get_max_threads() == 1) {
    init_matvec();
  } else {
    init_matvec_omp();
  }
  // init_matvec();

  // printf("cg_algo: %d, thread0_weight: %f\n", cg_algo, thread0_weight);
  if (cg_algo == 0) {
    matvecs_s = CG(b_s, s);       // CG on s - parallel
    matvecs_t = CG(b_t, t);       // CG on t - parallel
  } else if (cg_algo == 1) {
    matvecs_s = CG_omp(b_s, s);
    matvecs_t = CG_omp(b_t, t);
  } else if (cg_algo == 2) {
    matvecs_s = pipeline_CG(b_s, s);
    matvecs_t = pipeline_CG(b_t, t);
  } else if (cg_algo == 3) {
    matvecs_s = pipeline_CG_weight(b_s, s);
    matvecs_t = pipeline_CG_weight(b_t, t);
  }
  matvecs = matvecs_s + matvecs_t;

  calculate_Q();
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::pre_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::min_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::init_matvec()
{
  // GPTLstart("init_matvec");
  /* fill-in H matrix */
  //compute_H(); --> is integrated into the code below

  int jnum;
  int i, j, ii, jj, flag;
  double dx, dy, dz, r_sqr;
  constexpr double EPSILON = 0.0001;
  svd_t EPSILON_vec = svdup_f64(0.0001);

  using NeighEnt = ClusterNeighEntry<1>;
  int *numneigh_cluster = reaxff->list->numneigh_inner;
  NeighEnt **firstneigh_cluster = (NeighEnt **) reaxff->list->firstneigh_inner;

  int *type = atom->type;
  tagint *tag = atom->tag;
  double (*x)[3] = (double(*)[3])atom->x[0];
  int *mask = atom->mask;

  svd_t SQR_swb0 = svdup_f64(SQR(swb));

  // fill in the H matrix
  m_fill = 0;
  r_sqr = 0;

  int num_clusters = (nn + CLUSTERSIZE - 1) / CLUSTERSIZE;
  for(int ic=0; ic<num_clusters; ic++) {
    int iis = ic * CLUSTERSIZE;
    int iie = std::min(iis+CLUSTERSIZE, nn);
    int iicnt = iie - iis;
        
    NeighEnt *jlist_cluster = firstneigh_cluster[ic];
    int jnum_cluster = numneigh_cluster[ic];
    
    for (ii = iis; ii < iie; ii++) {
      i = ilist[ii];
      if (mask[i] & groupbit) {

        H.firstnbr[i] = m_fill;

        for (jj = 0; jj < jnum_cluster; jj++) {
          NeighEnt jent0 = jlist_cluster[jj];
          j = jent0.j;
          j &= NEIGHMASK;

          if(j <= i) continue;

          dx = x[j][0] - x[i][0];
          dy = x[j][1] - x[i][1];
          dz = x[j][2] - x[i][2];
          flag = 0;
#ifndef NOREVERSE
          if (j < atom->nlocal) flag = 1;
          else if (tag[i] < tag[j]) flag = 1;
          else if (tag[i] == tag[j]) {
            if (dz > EPSILON) flag = 1;
            else if (dz < EPSILON) {
              if (dy > EPSILON) flag = 1;
              else if (dy < EPSILON && dx > EPSILON)
                flag = 1;
            }
          }

	        if(!flag) continue;
#else
          flag = 1;
#endif

          r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

          if (r_sqr <= SQR(swb)) {
            H.jlist[m_fill] = j;
            H.val[m_fill] = calculate_H(sqrt(r_sqr), shld[type[i]][type[j]]);
            m_fill++;
          }
        }

        H.numnbrs[i] = m_fill - H.firstnbr[i];

        Hdia_inv[i] = 1. / eta[atom->type[i]];
        b_s[i]      = -chi[atom->type[i]];
        if (efield) b_s[i] -= chi_field[i];
        b_t[i]      = -1.0;

        /* quadratic extrapolation for s & t from previous solutions */
        t[i] = t_hist[i][2] + 3 * (t_hist[i][0] - t_hist[i][1]);

        /* cubic extrapolation for s & t from previous solutions */
        s[i] = 4*(s_hist[i][0]+s_hist[i][2])-(6*s_hist[i][1]+s_hist[i][3]);
      }
    }
  }

  pack_flag = 2;
  comm->forward_comm(this); //Dist_vector(s);
  pack_flag = 3;
  comm->forward_comm(this); //Dist_vector(t);
  // GPTLstop("init_matvec");
}

void FixQEqReaxFFSVE::init_matvec_omp()
{
  // GPTLstart("init_matvec");
  constexpr double EPSILON = 1.0e-4;
  const double swb2 = swb * swb;

  using NeighEnt = ClusterNeighEntry<1>;

  int *type  = atom->type;
  tagint *tag = atom->tag;
  double (*x)[3] = (double (*)[3]) atom->x[0];
  int *mask  = atom->mask;

  int *numneigh_cluster = reaxff->list->numneigh_inner;
  NeighEnt **firstneigh_cluster =
      (NeighEnt **) reaxff->list->firstneigh_inner;

  int num_clusters = (nn + CLUSTERSIZE - 1) / CLUSTERSIZE;

  /* ------------------------------------------------------------
   * Phase 1: count valid neighbors for each atom (parallel)
   * ------------------------------------------------------------ */
  std::vector<int> nbr_count(nn, 0);

#pragma omp parallel for schedule(dynamic)
  for (int ic = 0; ic < num_clusters; ic++) {
    int iis = ic * CLUSTERSIZE;
    int iie = std::min(iis + CLUSTERSIZE, nn);

    NeighEnt *jlist_cluster = firstneigh_cluster[ic];
    int jnum_cluster = numneigh_cluster[ic];

    for (int ii = iis; ii < iie; ii++) {
      int i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;

      int cnt = 0;

      for (int jj = 0; jj < jnum_cluster; jj++) {
        int j = jlist_cluster[jj].j & NEIGHMASK;
        if (j <= i) continue;

        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];

        int flag = 0;
#ifndef NOREVERSE
        if (j < atom->nlocal) flag = 1;
        else if (tag[i] < tag[j]) flag = 1;
        else if (tag[i] == tag[j]) {
          if (dz > EPSILON) flag = 1;
          else if (fabs(dz) < EPSILON) {
            if (dy > EPSILON) flag = 1;
            else if (fabs(dy) < EPSILON && dx > EPSILON)
              flag = 1;
          }
        }
        if (!flag) continue;
#else
        flag = 1;
#endif

        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 <= swb2)
          cnt++;
      }

      nbr_count[i] = cnt;
    }
  }

  /* ------------------------------------------------------------
   * Phase 2: prefix-sum (serial, cheap)
   * ------------------------------------------------------------ */
  m_fill = 0;
  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (mask[i] & groupbit) {
      H.firstnbr[i] = m_fill;
      H.numnbrs[i]  = nbr_count[i];
      m_fill += nbr_count[i];
    }
  }

  if (m_fill >= H.m)
    error->all(FLERR,
      "Fix qeq/reaxff/sve: H matrix size exceeded: m_fill={} H.m={}",
      m_fill, H.m);

  /* ------------------------------------------------------------
   * Phase 3: fill H matrix (parallel, no atomics)
   * ------------------------------------------------------------ */
#pragma omp parallel for schedule(dynamic)
  for (int ic = 0; ic < num_clusters; ic++) {
    int iis = ic * CLUSTERSIZE;
    int iie = std::min(iis + CLUSTERSIZE, nn);

    NeighEnt *jlist_cluster = firstneigh_cluster[ic];
    int jnum_cluster = numneigh_cluster[ic];

    for (int ii = iis; ii < iie; ii++) {
      int i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;

      int pos = H.firstnbr[i];

      for (int jj = 0; jj < jnum_cluster; jj++) {
        int j = jlist_cluster[jj].j & NEIGHMASK;
        if (j <= i) continue;

        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];

        int flag = 0;
#ifndef NOREVERSE
        if (j < atom->nlocal) flag = 1;
        else if (tag[i] < tag[j]) flag = 1;
        else if (tag[i] == tag[j]) {
          if (dz > EPSILON) flag = 1;
          else if (fabs(dz) < EPSILON) {
            if (dy > EPSILON) flag = 1;
            else if (fabs(dy) < EPSILON && dx > EPSILON)
              flag = 1;
          }
        }
        if (!flag) continue;
#else
        flag = 1;
#endif

        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 <= swb2) {
          H.jlist[pos] = j;
          H.val[pos]   = calculate_H(sqrt(r2),
                           shld[type[i]][type[j]]);
          pos++;
        }
      }

      /* diagonal & RHS terms: private per atom */
      Hdia_inv[i] = 1.0 / eta[type[i]];
      b_s[i] = -chi[type[i]];
      if (efield) b_s[i] -= chi_field[i];
      b_t[i] = -1.0;

      t[i] = t_hist[i][2] + 3.0 * (t_hist[i][0] - t_hist[i][1]);
      s[i] = 4.0*(s_hist[i][0] + s_hist[i][2])
           - (6.0*s_hist[i][1] + s_hist[i][3]);
    }
  }

  /* ------------------------------------------------------------
   * communication
   * ------------------------------------------------------------ */
  pack_flag = 2;
  comm->forward_comm(this);
  pack_flag = 3;
  comm->forward_comm(this);
  // GPTLstop("init_matvec");
}


/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::compute_H()
{
  int jnum;
  int i, j, ii, jj, flag;
  double dx, dy, dz, r_sqr;
  constexpr double EPSILON = 0.0001;

  int *type = atom->type;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int *mask = atom->mask;

  // fill in the H matrix
  m_fill = 0;
  r_sqr = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      H.firstnbr[i] = m_fill;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        dx = x[j][0] - x[i][0];
        dy = x[j][1] - x[i][1];
        dz = x[j][2] - x[i][2];
        r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

        flag = 0;
        if (r_sqr <= SQR(swb)) {
          if (j < atom->nlocal) flag = 1;
          else if (tag[i] < tag[j]) flag = 1;
          else if (tag[i] == tag[j]) {
            if (dz > EPSILON) flag = 1;
            else if (fabs(dz) < EPSILON) {
              if (dy > EPSILON) flag = 1;
              else if (fabs(dy) < EPSILON && dx > EPSILON)
                flag = 1;
            }
          }
        }

        if (flag) {
          H.jlist[m_fill] = j;
          H.val[m_fill] = calculate_H(sqrt(r_sqr), shld[type[i]][type[j]]);
          m_fill++;
        }
      }
      H.numnbrs[i] = m_fill - H.firstnbr[i];
    }
  }
  if (m_fill >= H.m)
    error->all(FLERR,"Fix qeq/reaxff H matrix size has been exceeded: m_fill={} H.m={}\n",
               m_fill, H.m);
}

/* ---------------------------------------------------------------------- */

double FixQEqReaxFFSVE::calculate_H(double r, double gamma)
{
  double Taper, denom;

  Taper = Tap[7] * r + Tap[6];
  Taper = Taper * r + Tap[5];
  Taper = Taper * r + Tap[4];
  Taper = Taper * r + Tap[3];
  Taper = Taper * r + Tap[2];
  Taper = Taper * r + Tap[1];
  Taper = Taper * r + Tap[0];

  denom = r * r * r + gamma;
  denom = pow(denom,1.0/3.0);

  return Taper * EV_TO_KCAL_PER_MOL / denom;
}

/* ---------------------------------------------------------------------- */

// NOTE: init version
int FixQEqReaxFFSVE::CG(double *b, double *x)
{
  // GPTLstart("CG");
  int  i, j;
  double tmp, alpha, beta, b_norm;
  double sig_old, sig_new;

  int jj;

  pack_flag = 1;

  // GPTLstart("compute stage 1");
  sparse_matvec(&H, x, q);
#ifndef NOREVERSE
  comm->reverse_comm(this); //Coll_Vector(q);
#endif

  vector_sum(r , 1.,  b, -1., q, nn);

  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    if (atom->mask[j] & groupbit)
      d[j] = r[j] * Hdia_inv[j]; //pre-condition
  }

  b_norm = parallel_norm(b, nn);
  sig_new = parallel_dot(r, d, nn);
  // GPTLstop("compute stage 1");

  // GPTLstart("compute stage 2");
  for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
    // GPTLstart("spmv inner");
    comm->forward_comm(this); //Dist_vector(d);
    sparse_matvec(&H, d, q);
#ifndef NOREVERSE
    comm->reverse_comm(this); //Coll_vector(q);
#endif
    // GPTLstop("spmv inner");

    // GPTLstart("update spmv");
    tmp = parallel_dot(d, q, nn);
    alpha = sig_new / tmp;

    vector_add(x, alpha, d, nn);
    vector_add(r, -alpha, q, nn);

    // pre-conditioning
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit)
        p[j] = r[j] * Hdia_inv[j];
    }

    sig_old = sig_new;
    sig_new = parallel_dot(r, p, nn);

    beta = sig_new / sig_old;
    vector_sum(d, 1., p, beta, d, nn);
    // GPTLstop("update spmv");
  }
  // GPTLstop("compute stage 2");

  if ((i >= imax) && maxwarn && (comm->me == 0))
    error->warning(FLERR, "Fix qeq/reaxff CG convergence failed after {} iterations at step {}",
                   i,update->ntimestep);
  // GPTLstop("CG");
  return i;
}

// NOTE: multi-thread init version
int FixQEqReaxFFSVE::CG_omp(double *b, double *x)
{
  // GPTLstart("CG");
  int  i, j;
  double tmp = 0.0, alpha, beta, b_norm;
  double sig_old, sig_new = 0.0;

  int jj;
  pack_flag = 1;

  // GPTLstart("compute stage 1");
  sparse_matvec_omp(&H, x, q);
#ifndef NOREVERSE
  comm->reverse_comm(this); //Coll_Vector(q);
#endif

  double my_sum = 0.0, norm_sqr = 0.0;
  double my_dot = 0.0;

  #pragma omp parallel for reduction(+:my_sum, my_dot)
  for(jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    if (atom->mask[j] & groupbit) { 
      r[j] = b[j] - q[j];
      d[j] = r[j] * Hdia_inv[j]; 
      my_sum += b[j] * b[j];
      my_dot += r[j] * d[j];
    }
  }

  MPI_Allreduce(&my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);
  b_norm = sqrt(norm_sqr);

  MPI_Allreduce(&my_dot, &sig_new, 1, MPI_DOUBLE, MPI_SUM, world);
  // GPTLstop("compute stage 1");

  // GPTLstart("compute stage 2");
  for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
    // GPTLstart("spmv inner");
    comm->forward_comm(this); 
    sparse_matvec_omp(&H, d, q);
#ifndef NOREVERSE
    comm->reverse_comm(this);
#endif
    // GPTLstop("spmv inner"); 

    // GPTLstart("update spmv");
    tmp = 0.0;
    my_dot = 0.0;

    #pragma omp parallel for reduction(+:my_dot)
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        my_dot += d[j] * q[j];
      }
    }
     
    MPI_Allreduce(&my_dot, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);


    alpha = sig_new / tmp;

    sig_old = sig_new;
    my_dot = 0.0;
    #pragma omp parallel for reduction(+:my_dot)
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        x[j] += alpha * d[j];
        r[j] -= alpha * q[j];
        p[j] = r[j] * Hdia_inv[j];
        my_dot += r[j] * p[j];
      }
    }
    MPI_Allreduce(&my_dot, &sig_new, 1, MPI_DOUBLE, MPI_SUM, world);


    beta = sig_new / sig_old;

    #pragma omp parallel for 
    for (jj = 0; jj < nn; ++jj) {
      j = ilist[jj];
      if (atom->mask[j] & groupbit) {
        d[j] = p[j] + beta * d[j];
      }
    }
    // GPTLstop("update spmv");
  }
  // GPTLstop("compute stage 2");

  if ((i >= imax) && maxwarn && (comm->me == 0))
    error->warning(FLERR, "Fix qeq/reaxff CG convergence failed after {} iterations at step {}",
                   i,update->ntimestep);
  // GPTLstop("CG");
  return i;
}

// NOTE: pipeline cg version, but no multi-thread
// int FixQEqReaxFFSVE::CG(double *b, double *x)
// {
//   int  i, j, jj;
//   double tmp, alpha, beta, b_norm;
//   double sig_old, sig_new, deta, heta;

//   pack_flag = 1;

//   // ----------------------------------------------------------------------
//   // Pipeline PCG Arrays - Allocated locally for drop-in replacement
//   // ----------------------------------------------------------------------
//   double *m, *u, *v, *w, *z, *s;
//   memory->create(m, atom->nmax, "qeq:m");
//   memory->create(u, atom->nmax, "qeq:u");
//   memory->create(v, atom->nmax, "qeq:v");
//   memory->create(w, atom->nmax, "qeq:w");
//   memory->create(z, atom->nmax, "qeq:z");
//   memory->create(s, atom->nmax, "qeq:s");

//   // Initial Matrix-Vector multiply: q = H * x
//   sparse_matvec(&H, x, q);
//   comm->reverse_comm(this);

//   // r = b - q
//   vector_sum(r, 1., b, -1., q, nn);

//   // Initial pre-conditioning: u = M * r
//   for (jj = 0; jj < nn; ++jj) {
//     j = ilist[jj];
//     if (atom->mask[j] & groupbit) {
//       u[j] = r[j] * Hdia_inv[j];
//     } else {
//       u[j] = 0.0;
//     }
//     d[j] = u[j];
//   }

//   // q = H * d (where d = u)
//   comm->forward_comm(this);
//   sparse_matvec(&H, d, q);
//   comm->reverse_comm(this);

//   for (jj = 0; jj < nn; ++jj) {
//     j = ilist[jj];
//     w[j] = q[j];                        // w = H * M * r
//     if (atom->mask[j] & groupbit) {
//       m[j] = q[j] * Hdia_inv[j];        // m = M * H * M * r
//     } else {
//       m[j] = 0.0;
//     }
//     d[j] = m[j];
//   }                                     

//   // q = H * d (where d = m)
//   comm->forward_comm(this);
//   sparse_matvec(&H, d, q);
//   comm->reverse_comm(this);       
  
//   for (jj = 0; jj < nn; ++jj) {
//     j = ilist[jj];
//     p[j] = u[j];
//     s[j] = w[j];
//     v[j] = m[j];
//     z[j] = q[j];
//   }                                     

//   b_norm = parallel_norm(b, nn);
//   sig_old = parallel_dot(u, r, nn);     
//   deta = parallel_dot(u, w, nn);        
//   heta = deta;                           
//   if (heta == 0.0) heta = 1.0e-15;      // Prevent division by zero
//   alpha = sig_old / heta;                
//   sig_new = sig_old;

//   // omp_set_max_active_levels(2); 

//   // Main Pipeline PCG Loop
//   for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
//     double dot_global[2] = {0.0, 0.0};

//     // Vector updates
//     for (jj = 0; jj < nn; ++jj) {
//       j = ilist[jj];
//       x[j] += alpha * p[j];
//       r[j] -= alpha * s[j];
//       u[j] -= alpha * v[j];
//       w[j] -= alpha * z[j];

//       if (atom->mask[j] & groupbit) {
//         d[j] = w[j] * Hdia_inv[j];
//       } else {
//         d[j] = 0.0;
//       }
//     }                                     
    
//     // Matrix-vector multiply for the step: q = H * d
//     comm->forward_comm(this);

//     #pragma omp parallel sections
//     {
//       #pragma omp section
//       {
//         sparse_matvec_scalar(&H, d, q);
//       }

//       #pragma omp section
//       {
//         double dot_local[2] = {0.0, 0.0};
        
//         for (int k = 0; k < nn; ++k) {
//           int idx = ilist[k];
//           dot_local[0] += u[idx] * r[idx];
//           dot_local[1] += u[idx] * w[idx];
//         }
        
//         MPI_Allreduce(dot_local, dot_global, 2, MPI_DOUBLE, MPI_SUM, world);
        
//         sig_new = dot_global[0];
//         deta = dot_global[1];
        
//         beta = sig_new / sig_old;             
//         heta = deta - beta * beta * heta;
//         alpha = sig_new / heta;               

//         for (int k = 0; k < nn; ++k) {
//           int idx = ilist[k];
//           p[idx] = u[idx] + p[idx] * beta;
//           s[idx] = w[idx] + s[idx] * beta;
//           v[idx] = d[idx] + v[idx] * beta;  
//         }
//       }
//     } 

//     comm->reverse_comm(this);

//     for (jj = 0; jj < nn; ++jj) {
//       j = ilist[jj];
//       z[j] = q[j] + z[j] * beta;  
//     }

//     sig_old = sig_new;
//   }

//   // Cleanup local memory
//   memory->destroy(m);
//   memory->destroy(u);
//   memory->destroy(v);
//   memory->destroy(w);
//   memory->destroy(z);
//   memory->destroy(s);

//   if ((i >= imax) && maxwarn && (comm->me == 0)) {
//     error->warning(FLERR, "Fix qeq/reaxff CG convergence failed after {} iterations at step {}",
//                    i, update->ntimestep);
//   }
  
//   return i;
// }

// NOTE: pipeline CG, omp version
int FixQEqReaxFFSVE::pipeline_CG(double *b, double *x)
{
  // GPTLstart("CG");
  int  i, j, jj;
  double tmp, alpha, beta, b_norm;
  double sig_old, sig_new, deta, heta;

  pack_flag = 1;

  // GPTLstart("compute stage 1");
  // Initial Matrix-Vector multiply: q = H * x
  sparse_matvec_omp(&H, x, q);
#ifndef NOREVERSE
  comm->reverse_comm(this);
#endif

  // r = b - q
  vector_sum(r, 1., b, -1., q, nn);

  // Initial pre-conditioning: u = M * r
  #pragma omp parallel for
  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    if (atom->mask[j] & groupbit) {
      u[j] = r[j] * Hdia_inv[j];
    } else {
      u[j] = 0.0;
    }
    d[j] = u[j];
  }

  // q = H * d (where d = u)
  comm->forward_comm(this);
  sparse_matvec_omp(&H, d, q);
#ifndef NOREVERSE
  comm->reverse_comm(this);
#endif

  #pragma omp parallel for
  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    w[j] = q[j];                        // w = H * M * r
    if (atom->mask[j] & groupbit) {
      m[j] = q[j] * Hdia_inv[j];        // m = M * H * M * r
    } else {
      m[j] = 0.0;
    }
    d[j] = m[j];
  }                                     

  // q = H * d (where d = m)
  comm->forward_comm(this);
  sparse_matvec_omp(&H, d, q);
#ifndef NOREVERSE
  comm->reverse_comm(this);       
#endif

  #pragma omp parallel for
  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    p[j] = u[j];
    p_s[j] = w[j];
    v[j] = m[j];
    z[j] = q[j];
  }                                     

  b_norm = parallel_norm(b, nn);
  sig_old = parallel_dot(u, r, nn);     
  deta = parallel_dot(u, w, nn);        
  heta = deta;                           
  if (heta == 0.0) heta = 1.0e-15;      // Prevent division by zero
  alpha = sig_old / heta;                
  sig_new = sig_old;

  // GPTLstop("compute stage 1");
  // GPTLstart("compute stage 2");

  int nthreads_total = omp_get_max_threads();
  int spmv_threads = (nthreads_total > 1) ? (nthreads_total - 1) : 1;
  int nlocal = atom->nlocal;
  int nall   = atom->nlocal + atom->nghost;

  // Main Pipeline PCG Loop
  for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
    double dot_global[2] = {0.0, 0.0};

    // GPTLstart("vector updates");
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (jj = 0; jj < nn; ++jj) {
        int idx = ilist[jj];
        x[idx] += alpha * p[idx];
        r[idx] -= alpha * p_s[idx];
        u[idx] -= alpha * v[idx];
        w[idx] -= alpha * z[idx];

        if (atom->mask[idx] & groupbit) {
          double d_val = w[idx] * Hdia_inv[idx];
          d[idx] = d_val;
          q[idx] = eta[atom->type[idx]] * d_val;
        } else {
          d[idx] = 0.0;
        }
      }

      #pragma omp for
      for (int idx = nlocal; idx < nall; idx++) {
        if (atom->mask[idx] & groupbit) {
          q[idx] = 0.0;
        }
      }
    }
    // Matrix-vector multiply for the step: q = H * d
    comm->forward_comm(this);
    // GPTLstop("vector updates");

    // GPTLstart("init spmv");
    #pragma omp parallel for
    for (int t = 0; t < spmv_threads; t++) {
      for (int idx = 0; idx < nall; idx++) {
        b_temp[t][idx] = 0.0;
      }
    }
    // GPTLstop("init spmv");

    // GPTLstart("compute and comm");
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();

      if (tid == 0) 
      {
        // GPTLstart("comm spmv");
        double dot_local[2] = {0.0, 0.0};
        
        for (int k = 0; k < nn; ++k) {
          int idx = ilist[k];
          dot_local[0] += u[idx] * r[idx];
          dot_local[1] += u[idx] * w[idx];
        }
        
        MPI_Allreduce(dot_local, dot_global, 2, MPI_DOUBLE, MPI_SUM, world);
        
        sig_new = dot_global[0];
        deta = dot_global[1];
        beta = sig_new / sig_old;             
        heta = deta - beta * beta * heta;
        alpha = sig_new / heta;               

        for (int k = 0; k < nn; ++k) {
          int idx = ilist[k];
          p[idx] = u[idx] + p[idx] * beta;
          p_s[idx] = w[idx] + p_s[idx] * beta;
          v[idx] = d[idx] + v[idx] * beta;  
        }
        // GPTLstop("comm spmv");
      } 
      else if (nthreads_total > 1) 
      {
        sparse_matvec_inner(&H, d, q, tid - 1, spmv_threads);
      }
    }

    if (nthreads_total == 1) {
      sparse_matvec_inner(&H, d, q, 0, 1);
    }
    // GPTLstop("compute and comm");

    // GPTLstart("update spmv");
    #pragma omp parallel for
    for (int idx = 0; idx < nall; idx++) {
      double sum = 0.0;
      for (int t = 0; t < spmv_threads; t++) {
        sum += b_temp[t][idx];
      }
      q[idx] += sum;
    }
    // GPTLstop("update spmv");

    // GPTLstart("update z");
#ifndef NOREVERSE
    comm->reverse_comm(this);
#endif

    #pragma omp parallel for
    for (jj = 0; jj < nn; ++jj) {
      int idx = ilist[jj];
      z[idx] = q[idx] + z[idx] * beta;  
    }

    sig_old = sig_new;
    // GPTLstop("update z");
  }
  // GPTLstop("compute stage 2");

  if ((i >= imax) && maxwarn && (comm->me == 0)) {
    error->warning(FLERR, "Fix qeq/reaxff CG convergence failed after {} iterations at step {}",
                   i, update->ntimestep);
  }
  
  // GPTLstop("CG");
  return i;
}

// NOTE: pipeline CG, omp version, with thread 0 weight
int FixQEqReaxFFSVE::pipeline_CG_weight(double *b, double *x)
{
  // GPTLstart("CG");
  int  i, j, jj;
  double tmp, alpha, beta, b_norm;
  double sig_old, sig_new, deta, heta;

  pack_flag = 1;

  // GPTLstart("compute stage 1");
  // Initial Matrix-Vector multiply: q = H * x
  sparse_matvec_omp(&H, x, q);
#ifndef NOREVERSE
  comm->reverse_comm(this);
#endif

  // r = b - q
  vector_sum(r, 1., b, -1., q, nn);

  // Initial pre-conditioning: u = M * r
  #pragma omp parallel for
  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    if (atom->mask[j] & groupbit) {
      u[j] = r[j] * Hdia_inv[j];
    } else {
      u[j] = 0.0;
    }
    d[j] = u[j];
  }

  // q = H * d (where d = u)
  comm->forward_comm(this);
  sparse_matvec_omp(&H, d, q);
#ifndef NOREVERSE
  comm->reverse_comm(this);
#endif

  #pragma omp parallel for
  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    w[j] = q[j];                        // w = H * M * r
    if (atom->mask[j] & groupbit) {
      m[j] = q[j] * Hdia_inv[j];        // m = M * H * M * r
    } else {
      m[j] = 0.0;
    }
    d[j] = m[j];
  }                                     

  // q = H * d (where d = m)
  comm->forward_comm(this);
  sparse_matvec_omp(&H, d, q);
#ifndef NOREVERSE
  comm->reverse_comm(this);    
#endif   
  
  #pragma omp parallel for
  for (jj = 0; jj < nn; ++jj) {
    j = ilist[jj];
    p[j] = u[j];
    p_s[j] = w[j];
    v[j] = m[j];
    z[j] = q[j];
  }                                     

  b_norm = parallel_norm(b, nn);
  sig_old = parallel_dot(u, r, nn);     
  deta = parallel_dot(u, w, nn);        
  heta = deta;                           
  if (heta == 0.0) heta = 1.0e-15;      // Prevent division by zero
  alpha = sig_old / heta;                
  sig_new = sig_old;

  // GPTLstop("compute stage 1");
  // GPTLstart("compute stage 2");

  int nthreads_total = omp_get_max_threads();
  int nlocal = atom->nlocal;
  int nall   = atom->nlocal + atom->nghost;

  if (nthreads_total == 1) {
    thread0_weight = 1.0; 
  }
  
  int start_t = (thread0_weight > 0.0) ? 0 : 1;

  // Main Pipeline PCG Loop
  for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
    double dot_global[2] = {0.0, 0.0};

    // GPTLstart("vector updates");
    #pragma omp parallel 
    {
      #pragma omp for nowait
      for (jj = 0; jj < nn; ++jj) {
        int idx = ilist[jj];
        x[idx] += alpha * p[idx];
        r[idx] -= alpha * p_s[idx];
        u[idx] -= alpha * v[idx];
        w[idx] -= alpha * z[idx];

        if (atom->mask[idx] & groupbit) {
          double d_val = w[idx] * Hdia_inv[idx];
          d[idx] = d_val;
          q[idx] = eta[atom->type[idx]] * d_val;
        } else {
          d[idx] = 0.0;
        }
      }

      #pragma omp for
      for (int idx = nlocal; idx < nall; idx++) {
        if (atom->mask[idx] & groupbit) {
          q[idx] = 0.0;
        }
      }
    }
    // Matrix-vector multiply for the step: q = H * d
    comm->forward_comm(this);
    // GPTLstop("vector updates");

    // GPTLstart("init spmv");
    #pragma omp parallel for
    for (int t = start_t; t < nthreads_total; t++) {
      for (int idx = 0; idx < nall; idx++) {
        b_temp[t][idx] = 0.0;
      }
    }
    // GPTLstop("init spmv");

    // GPTLstart("compute and comm");
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int start = 0, end_idx = nn;

      if (nthreads_total > 1) {
        double total_weights = thread0_weight + (nthreads_total - 1) * 1.0; 
        
        if (tid == 0) {
          start = 0;
          end_idx = (int)((nn * thread0_weight) / total_weights);
        } else {
          double start_weight = thread0_weight + (tid - 1) * 1.0;
          double end_weight = start_weight + 1.0;
          start = (int)((nn * start_weight) / total_weights);
          end_idx = (int)((nn * end_weight) / total_weights);
        }
        
        if (tid == nthreads_total - 1) {
          end_idx = nn;
        }
      }

      if (tid == 0) 
      {
        // GPTLstart("comm spmv");
        double dot_local[2] = {0.0, 0.0};
        
        for (int k = 0; k < nn; ++k) {
          int idx = ilist[k];
          dot_local[0] += u[idx] * r[idx];
          dot_local[1] += u[idx] * w[idx];
        }
        
        MPI_Allreduce(dot_local, dot_global, 2, MPI_DOUBLE, MPI_SUM, world);
        
        sig_new = dot_global[0];
        deta = dot_global[1];
        beta = sig_new / sig_old;             
        heta = deta - beta * beta * heta;
        alpha = sig_new / heta;               

        for (int k = 0; k < nn; ++k) {
          int idx = ilist[k];
          p[idx] = u[idx] + p[idx] * beta;
          p_s[idx] = w[idx] + p_s[idx] * beta;
          v[idx] = d[idx] + v[idx] * beta;  
        }
        // GPTLstop("comm spmv");
        
        if (thread0_weight > 0.0 && start < end_idx) {
          sparse_matvec_inner(&H, d, q, start, end_idx, tid);
        }
      } 
      else 
      {
        if (start < end_idx) {
          sparse_matvec_inner(&H, d, q, start, end_idx, tid);
        }
      }
    }
    // GPTLstop("compute and comm");

    // GPTLstart("update spmv");
    #pragma omp parallel for
    for (int idx = 0; idx < nall; idx++) {
      double sum = 0.0;
      for (int t = start_t; t < nthreads_total; t++) {
        sum += b_temp[t][idx];
      }
      q[idx] += sum;
    }
    // GPTLstop("update spmv");

    // GPTLstart("update z");
#ifndef NOREVERSE
    comm->reverse_comm(this);
#endif

    #pragma omp parallel for
    for (jj = 0; jj < nn; ++jj) {
      int idx = ilist[jj];
      z[idx] = q[idx] + z[idx] * beta;  
    }

    sig_old = sig_new;
    // GPTLstop("update z");
  }
  // GPTLstop("compute stage 2");

  if ((i >= imax) && maxwarn && (comm->me == 0)) {
    error->warning(FLERR, "Fix qeq/reaxff CG convergence failed after {} iterations at step {}",
                   i, update->ntimestep);
  }
  
  // GPTLstop("CG");
  return i;
}


/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::sparse_matvec(sparse_matrix *A, double *x, double *b)
{
  int i, j, itr_j;
  int ii;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      b[i] = eta[atom->type[i]] * x[i];
  }

  int nall = atom->nlocal + atom->nghost;
  for (i = atom->nlocal; i < nall; ++i) 
    b[i] = 0;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      svint64_t vi = svdup_s64(i);
      svd_t vxi = svdup_f64(x[i]);
      int end = A->firstnbr[i] + A->numnbrs[i];
      svd_t vbi = svdup_f64(0.0);
      for (itr_j = A->firstnbr[i]; itr_j < end; itr_j += CLUSTERSIZE) {
        svbool_t predi = svwhilelt_b64(itr_j, end);
        svint64_t vj = svld1sw_s64(predi, &A->jlist[itr_j]);
        svd_t vval = svld1(predi, &A->val[itr_j]);
        svd_t vxj = svld1_gather_index(predi, x, vj);
        svd_t vbj = svld1_gather_index(predi, b, vj);
        // svst1_scatter_index(predi, b, vj, vbj + vval * vxi);
        // vbi = svadd_m(predi, vbi, vval * vxj);
        svst1_scatter_index(predi, b, vj, svmla_x(predi, vbj, vval, vxi));
        vbi = svmla_m(predi, vbi, vval, vxj);
      }
      b[i] += svaddv_f64(svptrue_b64(), vbi);
    }
  }
}

void FixQEqReaxFFSVE::sparse_matvec_omp(sparse_matrix *A, double *x, double *b)
{
  // // GPTLstart("sparse_matvec");
  int nlocal = atom->nlocal;
  int nall   = atom->nlocal + atom->nghost;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_max_threads();

    /* 1. b[i] = eta_i * x[i] */
    #pragma omp for schedule(dynamic,64)
    for (int ii = 0; ii < nn; ii++) {
      int i = ilist[ii];
      if (atom->mask[i] & groupbit)
        b[i] = eta[atom->type[i]] * x[i];
    }

    /* 2. ghost part */
    #pragma omp for schedule(dynamic,64)
    for (int i = nlocal; i < nall; i++)
      if (atom->mask[i] & groupbit)
        b[i] = 0.0;

    /* 3. zero thread-local buffer */
    #pragma omp for schedule(dynamic,64)
    for (int i = 0; i < nall; i++)
      for (int t = 0; t < nthreads; t++)
        b_temp[t][i] = 0.0;

    #pragma omp barrier

    /* 4. SpMV main loop (SVE accelerated) */
    #pragma omp for schedule(dynamic,64)
    for (int ii = 0; ii < nn; ++ii) {
      int i = ilist[ii];
      if (atom->mask[i] & groupbit) {
        svint64_t vi = svdup_s64(i);
        svd_t vxi = svdup_f64(x[i]);
        int end = A->firstnbr[i] + A->numnbrs[i];
        svd_t vbi = svdup_f64(0.0);
        for (int itr_j = A->firstnbr[i]; itr_j < end; itr_j += CLUSTERSIZE) {
          svbool_t predi = svwhilelt_b64(itr_j, end);
          svint64_t vj = svld1sw_s64(predi, &A->jlist[itr_j]);
          svd_t vval = svld1(predi, &A->val[itr_j]);
          svd_t vxj = svld1_gather_index(predi, x, vj);
          vbi = svmla_m(predi, vbi, vval, vxj);
          svd_t vbj = svld1_gather_index(predi, b_temp[tid], vj);
          svst1_scatter_index(predi, b_temp[tid], vj, svmla_x(predi, vbj, vval, vxi));
        }
        b[i] += svaddv_f64(svptrue_b64(), vbi);
      }
    }

    #pragma omp barrier

    /* 5. reduction */
    #pragma omp for schedule(dynamic,64)
    for (int i = 0; i < nall; i++)
      for (int t = 0; t < nthreads; t++)
        b[i] += b_temp[t][i];
  }
  // // GPTLstop("sparse_matvec");
}

void FixQEqReaxFFSVE::sparse_matvec_inner(sparse_matrix *A, double *x, double *b, int spmv_tid, int spmv_threads)
{
  // GPTLstart("sparse_matvec_inner");
  
  int start = (nn * spmv_tid) / spmv_threads;
  int end_idx = (nn * (spmv_tid + 1)) / spmv_threads;

  for (int ii = start; ii < end_idx; ++ii) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      svint64_t vi = svdup_s64(i);
      svd_t vxi = svdup_f64(x[i]);
      int row_start = A->firstnbr[i];
      int row_end = row_start + A->numnbrs[i];
      svd_t vbi = svdup_f64(0.0);

      for (int itr_j = row_start; itr_j < row_end; itr_j += CLUSTERSIZE) {
        svbool_t predi = svwhilelt_b64(itr_j, row_end);
        svint64_t vj = svld1sw_s64(predi, &A->jlist[itr_j]);
        svd_t vval = svld1(predi, &A->val[itr_j]);
        svd_t vxj = svld1_gather_index(predi, x, vj);
        vbi = svmla_m(predi, vbi, vval, vxj);
        
        svd_t vbj = svld1_gather_index(predi, b_temp[spmv_tid], vj);
        svst1_scatter_index(predi, b_temp[spmv_tid], vj, svmla_x(predi, vbj, vval, vxi));
      }
      b[i] += svaddv_f64(svptrue_b64(), vbi);
    }
  }

  // GPTLstop("sparse_matvec_inner");
}

void FixQEqReaxFFSVE::sparse_matvec_inner(sparse_matrix *A, double *x, double *b, int start, int end_idx, int t_idx)
{
  // GPTLstart("sparse_matvec_inner");
  
  for (int ii = start; ii < end_idx; ++ii) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      svint64_t vi = svdup_s64(i);
      svd_t vxi = svdup_f64(x[i]);
      int row_start = A->firstnbr[i];
      int row_end = row_start + A->numnbrs[i];
      svd_t vbi = svdup_f64(0.0);

      for (int itr_j = row_start; itr_j < row_end; itr_j += CLUSTERSIZE) {
        svbool_t predi = svwhilelt_b64(itr_j, row_end);
        svint64_t vj = svld1sw_s64(predi, &A->jlist[itr_j]);
        svd_t vval = svld1(predi, &A->val[itr_j]);
        svd_t vxj = svld1_gather_index(predi, x, vj);
        vbi = svmla_m(predi, vbi, vval, vxj);
        
        svd_t vbj = svld1_gather_index(predi, b_temp[t_idx], vj);
        svst1_scatter_index(predi, b_temp[t_idx], vj, svmla_x(predi, vbj, vval, vxi));
      }
      b[i] += svaddv_f64(svptrue_b64(), vbi);
    }
  }

  // GPTLstop("sparse_matvec_inner");
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::calculate_Q()
{
  int i, k;
  double u, s_sum, t_sum;
  double *q = atom->q;

  int ii;

  s_sum = parallel_vector_acc(s, nn);
  t_sum = parallel_vector_acc(t, nn);
  u = s_sum / t_sum;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      q[i] = s[i] - u * t[i];

      /* backup s & t */
      for (k = nprev-1; k > 0; --k) {
        s_hist[i][k] = s_hist[i][k-1];
        t_hist[i][k] = t_hist[i][k-1];
      }
      s_hist[i][0] = s[i];
      t_hist[i][0] = t[i];
    }
  }

  pack_flag = 4;
  comm->forward_comm(this); //Dist_vector(atom->q);
}

/* ---------------------------------------------------------------------- */

int FixQEqReaxFFSVE::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int m;

  if (pack_flag == 1)
    for (m = 0; m < n; m++) buf[m] = d[list[m]];
  else if (pack_flag == 2)
    for (m = 0; m < n; m++) buf[m] = s[list[m]];
  else if (pack_flag == 3)
    for (m = 0; m < n; m++) buf[m] = t[list[m]];
  else if (pack_flag == 4)
    for (m = 0; m < n; m++) buf[m] = atom->q[list[m]];
  else if (pack_flag == 5) {
    m = 0;
    for (int i = 0; i < n; i++) {
      int j = 2 * list[i];
      buf[m++] = d[j];
      buf[m++] = d[j+1];
    }
    return m;
  }
  return n;
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  if (pack_flag == 1)
    for (m = 0, i = first; m < n; m++, i++) d[i] = buf[m];
  else if (pack_flag == 2)
    for (m = 0, i = first; m < n; m++, i++) s[i] = buf[m];
  else if (pack_flag == 3)
    for (m = 0, i = first; m < n; m++, i++) t[i] = buf[m];
  else if (pack_flag == 4)
    for (m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
  else if (pack_flag == 5) {
    int last = first + n;
    m = 0;
    for (i = first; i < last; i++) {
      int j = 2 * i;
      d[j] = buf[m++];
      d[j+1] = buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixQEqReaxFFSVE::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  if (pack_flag == 5) {
    m = 0;
    int last = first + n;
    for (i = first; i < last; i++) {
      int indxI = 2 * i;
      buf[m++] = q[indxI];
      buf[m++] = q[indxI+1];
    }
    return m;
  } else {
    for (m = 0, i = first; m < n; m++, i++) buf[m] = q[i];
    return n;
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::unpack_reverse_comm(int n, int *list, double *buf)
{
  if (pack_flag == 5) {
    int m = 0;
    for (int i = 0; i < n; i++) {
      int indxI = 2 * list[i];
      q[indxI] += buf[m++];
      q[indxI+1] += buf[m++];
    }
  } else {
    for (int m = 0; m < n; m++) q[list[m]] += buf[m];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixQEqReaxFFSVE::memory_usage()
{
  double bytes;

  bytes = (double)atom->nmax*nprev*2 * sizeof(double); // s_hist & t_hist
  bytes += (double)atom->nmax*11 * sizeof(double); // storage
  bytes += (double)n_cap*2 * sizeof(int); // matrix...
  bytes += (double)m_cap * sizeof(int);
  bytes += (double)m_cap * sizeof(double);

  if (dual_enabled)
    bytes += (double)atom->nmax*4 * sizeof(double); // double size for q, d, r, and p

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqReaxFFSVE::grow_arrays(int nmax)
{
  memory->grow(s_hist,nmax,nprev,"qeq:s_hist");
  memory->grow(t_hist,nmax,nprev,"qeq:t_hist");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqReaxFFSVE::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int m = 0; m < nprev; m++) {
    s_hist[j][m] = s_hist[i][m];
    t_hist[j][m] = t_hist[i][m];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixQEqReaxFFSVE::pack_exchange(int i, double *buf)
{
  for (int m = 0; m < nprev; m++) buf[m] = s_hist[i][m];
  for (int m = 0; m < nprev; m++) buf[nprev+m] = t_hist[i][m];
  return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixQEqReaxFFSVE::unpack_exchange(int nlocal, double *buf)
{
  for (int m = 0; m < nprev; m++) s_hist[nlocal][m] = buf[m];
  for (int m = 0; m < nprev; m++) t_hist[nlocal][m] = buf[nprev+m];
  return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixQEqReaxFFSVE::parallel_norm(double *v, int n)
{
  int  i;
  double my_sum, norm_sqr;

  int ii;

  my_sum = 0.0;
  norm_sqr = 0.0;

  #pragma omp parallel for reduction(+:my_sum)
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      my_sum += SQR(v[i]);
  }

  MPI_Allreduce(&my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

  return sqrt(norm_sqr);
}

/* ---------------------------------------------------------------------- */

double FixQEqReaxFFSVE::parallel_dot(double *v1, double *v2, int n)
{
  int  i;
  double my_dot, res;

  int ii;

  my_dot = 0.0;
  res = 0.0;

  #pragma omp parallel for reduction(+:my_dot)
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      my_dot += v1[i] * v2[i];
  }

  MPI_Allreduce(&my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ---------------------------------------------------------------------- */

double FixQEqReaxFFSVE::parallel_vector_acc(double *v, int n)
{
  int  i;
  double my_acc, res;

  int ii;

  my_acc = 0.0;
  res = 0.0;

  #pragma omp parallel for reduction(+:my_acc)
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      my_acc += v[i];
  }

  MPI_Allreduce(&my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::vector_sum(double* dest, double c, double* v,
                                double d, double* y, int k)
{
  // int kk;

  // for (--k; k>=0; --k) {
  //   kk = ilist[k];
  //   if (atom->mask[kk] & groupbit)
  //     dest[kk] = c * v[kk] + d * y[kk];
  // }
  #pragma omp parallel for
  for (int i = 0; i < k; ++i) {
    int kk = ilist[i];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = c * v[kk] + d * y[kk];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::vector_add(double* dest, double c, double* v, int k)
{
  // int kk;

  // for (--k; k>=0; --k) {
  //   kk = ilist[k];
  //   if (atom->mask[kk] & groupbit)
  //     dest[kk] += c * v[kk];
  // }
  #pragma omp parallel for
  for (int i = 0; i < k; ++i) {
    int kk = ilist[i]; 
    if (atom->mask[kk] & groupbit) {
      dest[kk] += c * v[kk];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFSVE::get_chi_field()
{
  memset(&chi_field[0],0,atom->nmax*sizeof(double));
  if (!efield) return;

  const auto x = (const double * const *)atom->x;
  const int *mask = atom->mask;
  const imageint *image = atom->image;
  const int nlocal = atom->nlocal;


  // update electric field region if necessary

  Region *region = efield->region;
  if (region) region->prematch();

  // efield energy is in real units of kcal/mol/angstrom, need to convert to eV

  const double qe2f = force->qe2f;
  const double factor = -1.0/qe2f;


  if (efield->varflag != FixEfieldSVE::CONSTANT)
    efield->update_efield_variables();

  // atom selection is for the group of fix efield

  double unwrap[3];
  const double ex = efield->ex;
  const double ey = efield->ey;
  const double ez = efield->ez;
  const int efgroupbit = efield->groupbit;

    // charge interactions
    // force = qE, potential energy = F dot x in unwrapped coords
  if (efield->varflag != FixEfieldSVE::ATOM) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & efgroupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        domain->unmap(x[i],image[i],unwrap);
        chi_field[i] = factor*(ex*unwrap[0] + ey*unwrap[1] + ez*unwrap[2]);
      }
    }
  } else { // must use atom-style potential from FixEfieldSVE
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & efgroupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        chi_field[i] = efield->efield[i][3];
      }
    }
  }
}
