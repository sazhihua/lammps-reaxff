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
typedef NPairClusterBinSVE<0, 1, 0, 0, 0> NPairGroupFullBinSVE;
NPairStyle(full/bin/sve,
           NPairGroupFullBinSVE,
           NP_FULL | NP_BIN | NP_MOLONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 0, 0, 0, 0> NPairGroupHalfBinNewtoffSVE;
NPairStyle(half/bin/newtoff/sve,
           NPairGroupHalfBinNewtoffSVE,
           NP_HALF | NP_BIN | NP_MOLONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 0, 0, 0> NPairGroupHalfBinNewtonSVE;
NPairStyle(half/bin/newton/sve,
           NPairGroupHalfBinNewtonSVE,
           NP_HALF | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 1, 0, 0> NPairGroupHalfBinNewtonTriSVE;
NPairStyle(half/bin/newton/tri/sve,
           NPairGroupHalfBinNewtonTriSVE,
           NP_HALF | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<0, 1, 0, 1, 0> NPairGroupFullSizeBinSVE;
NPairStyle(full/size/bin/sve,
           NPairGroupFullSizeBinSVE,
           NP_FULL | NP_SIZE | NP_BIN | NP_MOLONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 0, 0, 1, 0> NPairGroupHalfSizeBinNewtoffSVE;
NPairStyle(half/size/bin/newtoff/sve,
           NPairGroupHalfSizeBinNewtoffSVE,
           NP_HALF | NP_SIZE | NP_BIN | NP_MOLONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 0, 1, 0> NPairGroupHalfSizeBinNewtonSVE;
NPairStyle(half/size/bin/newton/sve,
           NPairGroupHalfSizeBinNewtonSVE,
           NP_HALF | NP_SIZE | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 1, 1, 0> NPairGroupHalfSizeBinNewtonTriSVE;
NPairStyle(half/size/bin/newton/tri/sve,
           NPairGroupHalfSizeBinNewtonTriSVE,
           NP_HALF | NP_SIZE | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<0, 1, 0, 0, 1> NPairGroupFullBinAtomonlySVE;
NPairStyle(full/bin/atomonly/sve,
           NPairGroupFullBinAtomonlySVE,
           NP_FULL | NP_BIN | NP_ATOMONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 0, 0, 0, 1> NPairGroupHalfBinAtomonlyNewtoffSVE;
NPairStyle(half/bin/atomonly/newtoff/sve,
           NPairGroupHalfBinAtomonlyNewtoffSVE,
           NP_HALF | NP_BIN | NP_ATOMONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 0, 0, 1> NPairGroupHalfBinAtomonlyNewtonSVE;
NPairStyle(half/bin/atomonly/newton/sve,
           NPairGroupHalfBinAtomonlyNewtonSVE,
           NP_HALF | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 1, 0, 1> NPairGroupHalfBinAtomonlyNewtonTriSVE;
NPairStyle(half/bin/atomonly/newton/tri/sve,
           NPairGroupHalfBinAtomonlyNewtonTriSVE,
           NP_HALF | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<0, 1, 0, 1, 1> NPairGroupFullSizeBinAtomonlySVE;
NPairStyle(full/size/bin/atomonly/sve,
           NPairGroupFullSizeBinAtomonlySVE,
           NP_FULL | NP_SIZE | NP_BIN | NP_ATOMONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 0, 0, 1, 1> NPairGroupHalfSizeBinAtomonlyNewtoffSVE;
NPairStyle(half/size/bin/atomonly/newtoff/sve,
           NPairGroupHalfSizeBinAtomonlyNewtoffSVE,
           NP_HALF | NP_SIZE | NP_BIN | NP_ATOMONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 0, 1, 1> NPairGroupHalfSizeBinAtomonlyNewtonSVE;
NPairStyle(half/size/bin/atomonly/newton/sve,
           NPairGroupHalfSizeBinAtomonlyNewtonSVE,
           NP_HALF | NP_SIZE | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER | NP_SVE);

typedef NPairClusterBinSVE<1, 1, 1, 1, 1> NPairGroupHalfSizeBinAtomonlyNewtonTriSVE;
NPairStyle(half/size/bin/atomonly/newton/tri/sve,
           NPairGroupHalfSizeBinAtomonlyNewtonTriSVE,
           NP_HALF | NP_SIZE | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_TRI| NP_CLUSTER | NP_SVE);
// clang-format on
#else

#ifndef LMP_NPAIR_CLUSTER_BIN_SVE_H
#define LMP_NPAIR_CLUSTER_BIN_SVE_H
#include "my_page.h"
#include "npair.h"
#include "cluster_neigh.h"
#include <stdint.h>

namespace LAMMPS_NS {
template <int HALF, int NEWTON, int TRI, int SIZE, int ATOMONLY>
class NPairClusterBinSVE : public NPair {
 public:
  NPairClusterBinSVE(class LAMMPS *);
  ~NPairClusterBinSVE() override;
  void build(class NeighList *) override;
  void build_normal(class NeighList *);
  int ncluster;
  ClusterNeighEntry<ATOMONLY> **firstneigh_cluster, *neigh_buffer;
  int *jnum_cluster;
  int firstneigh_alloc;
};

}    // namespace LAMMPS_NS

#endif
#endif
