#ifndef LMP_THR_SVE_H
#define LMP_THR_SVE_H
#include <omp.h>
#include <sys/cdefs.h>

#include "neigh_list.h"
#include "pointers.h"
#include "pair.h"
#include "atom.h"

#ifndef ATOMS_PER_PAGE
#define ATOMS_PER_PAGE 64
#endif
namespace LAMMPS_NS{
  class ThrSVE {
    LAMMPS *lmp;
    int nalloc; /* Number of atoms data are allocated for */
    int nthr; /* Number of threads */
    double (*all_force)[3], (*all_vatom)[6], *all_eatom; /* Per-thread per-atom force, virial and energy */
    double (*eng_virial)[8]; /* Thread total virial, energy and coulomb */
    double (*all_pvector)[16]; /* Pvector data for reaxff */
    bool *all_fflag;
  public:
    ThrSVE(LAMMPS *lmp);
    ~ThrSVE();
    void allocate(int nall, int eflag_atom, int vflag_atom);
    void set_range(int &start, int &end, int ithr, int n);
    template<int MOL>
    void inspect_cluster(int ithr, NeighList *list);
    void inspect_no_cluster(int ithr, NeighList *list);
    void inspect(int, NeighList *);
    void init_pair(int ithr, int nall, int eflag_atom, int vflag_atom, int vflag_fdotr);
    template<int VFLAG_FDOTR>
    void reduce_pair(Atom *atom, Pair *pair, int ithr, int nthr, int eflag_atom, int vflag_atom);
    void reduce_scalar(Pair *pair, int eflag_global, int vflag_global);
    __always_inline decltype(all_force) force_thr(int ithr){
      return all_force + nalloc * ithr;
    }
    __always_inline decltype(all_fflag) fflag_thr(int ithr) {
      return all_fflag + nalloc / ATOMS_PER_PAGE * ithr;
    }
    __always_inline double *eng_virial_thr(int ithr) {
      return eng_virial[ithr];
    }
    __always_inline decltype(all_vatom) vatom_thr(int ithr) {
      return all_vatom + nalloc * ithr;
    }
    __always_inline decltype(all_eatom) eatom_thr(int ithr) {
      return all_eatom + nalloc * ithr;
    }
    //////for reaxff//////
    __always_inline double *pvector_thr(int ithr) {
      return all_pvector[ithr];
    }
  };
}
#endif
