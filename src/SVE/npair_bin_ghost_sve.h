/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef NPAIR_CLASS
// clang-format off
typedef NPairBinGhostSVE<0> NPairFullBinGhostSVE;
NPairStyle(full/bin/ghost/sve,
           NPairFullBinGhostSVE,
           NP_FULL | NP_BIN | NP_NEWTON | NP_NEWTOFF | NP_GHOST | NP_ORTHO | NP_TRI | NP_SVE);

typedef NPairBinGhostSVE<1> NPairHalfBinGhostNewtoffSVE;
NPairStyle(half/bin/ghost/newtoff/sve,
           NPairHalfBinGhostNewtoffSVE,
           NP_HALF | NP_BIN | NP_NEWTOFF | NP_GHOST | NP_ORTHO | NP_TRI | NP_SVE);
// clang-format on
#else

#ifndef LMP_NPAIR_BIN_GHOST_SVE_H
#define LMP_NPAIR_BIN_GHOST_SVE_H

#include "npair.h"

namespace LAMMPS_NS {

template<int HALF>
class NPairBinGhostSVE : public NPair {
 public:
  NPairBinGhostSVE(class LAMMPS *);
  void build(class NeighList *) override;
  void build_normal(class NeighList *);
  void build_reaxff(class NeighList *);
};

}    // namespace LAMMPS_NS

#endif
#endif
