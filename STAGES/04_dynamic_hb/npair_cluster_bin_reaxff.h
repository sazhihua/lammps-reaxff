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
typedef NPairClusterBinReaxFFCluster<0, 1, 0, 0, 0> NPairGroupFullBinReaxFFCluster;
NPairStyle(full/bin/reaxff/cluster,
           NPairGroupFullBinReaxFFCluster,
           NP_FULL | NP_BIN | NP_MOLONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 0, 0, 0, 0> NPairGroupHalfBinNewtoffReaxFFCluster;
NPairStyle(half/bin/newtoff/reaxff/cluster,
           NPairGroupHalfBinNewtoffReaxFFCluster,
           NP_HALF | NP_BIN | NP_MOLONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 0, 0, 0> NPairGroupHalfBinNewtonReaxFFCluster;
NPairStyle(half/bin/newton/reaxff/cluster,
           NPairGroupHalfBinNewtonReaxFFCluster,
           NP_HALF | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 1, 0, 0> NPairGroupHalfBinNewtonTriReaxFFCluster;
NPairStyle(half/bin/newton/tri/reaxff/cluster,
           NPairGroupHalfBinNewtonTriReaxFFCluster,
           NP_HALF | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<0, 1, 0, 1, 0> NPairGroupFullSizeBinReaxFFCluster;
NPairStyle(full/size/bin/reaxff/cluster,
           NPairGroupFullSizeBinReaxFFCluster,
           NP_FULL | NP_SIZE | NP_BIN | NP_MOLONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 0, 0, 1, 0> NPairGroupHalfSizeBinNewtoffReaxFFCluster;
NPairStyle(half/size/bin/newtoff/reaxff/cluster,
           NPairGroupHalfSizeBinNewtoffReaxFFCluster,
           NP_HALF | NP_SIZE | NP_BIN | NP_MOLONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 0, 1, 0> NPairGroupHalfSizeBinNewtonReaxFFCluster;
NPairStyle(half/size/bin/newton/reaxff/cluster,
           NPairGroupHalfSizeBinNewtonReaxFFCluster,
           NP_HALF | NP_SIZE | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 1, 1, 0> NPairGroupHalfSizeBinNewtonTriReaxFFCluster;
NPairStyle(half/size/bin/newton/tri/reaxff/cluster,
           NPairGroupHalfSizeBinNewtonTriReaxFFCluster,
           NP_HALF | NP_SIZE | NP_BIN | NP_MOLONLY | NP_NEWTON | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<0, 1, 0, 0, 1> NPairGroupFullBinAtomonlyReaxFFCluster;
NPairStyle(full/bin/atomonly/reaxff/cluster,
           NPairGroupFullBinAtomonlyReaxFFCluster,
           NP_FULL | NP_BIN | NP_ATOMONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 0, 0, 0, 1> NPairGroupHalfBinAtomonlyNewtoffReaxFFCluster;
NPairStyle(half/bin/atomonly/newtoff/reaxff/cluster,
           NPairGroupHalfBinAtomonlyNewtoffReaxFFCluster,
           NP_HALF | NP_BIN | NP_ATOMONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 0, 0, 1> NPairGroupHalfBinAtomonlyNewtonReaxFFCluster;
NPairStyle(half/bin/atomonly/newton/reaxff/cluster,
           NPairGroupHalfBinAtomonlyNewtonReaxFFCluster,
           NP_HALF | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 1, 0, 1> NPairGroupHalfBinAtomonlyNewtonTriReaxFFCluster;
NPairStyle(half/bin/atomonly/newton/tri/reaxff/cluster,
           NPairGroupHalfBinAtomonlyNewtonTriReaxFFCluster,
           NP_HALF | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<0, 1, 0, 1, 1> NPairGroupFullSizeBinAtomonlyReaxFFCluster;
NPairStyle(full/size/bin/atomonly/reaxff/cluster,
           NPairGroupFullSizeBinAtomonlyReaxFFCluster,
           NP_FULL | NP_SIZE | NP_BIN | NP_ATOMONLY |
           NP_NEWTON | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 0, 0, 1, 1> NPairGroupHalfSizeBinAtomonlyNewtoffReaxFFCluster;
NPairStyle(half/size/bin/atomonly/newtoff/reaxff/cluster,
           NPairGroupHalfSizeBinAtomonlyNewtoffReaxFFCluster,
           NP_HALF | NP_SIZE | NP_BIN | NP_ATOMONLY | NP_NEWTOFF | NP_ORTHO | NP_TRI| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 0, 1, 1> NPairGroupHalfSizeBinAtomonlyNewtonReaxFFCluster;
NPairStyle(half/size/bin/atomonly/newton/reaxff/cluster,
           NPairGroupHalfSizeBinAtomonlyNewtonReaxFFCluster,
           NP_HALF | NP_SIZE | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_ORTHO| NP_CLUSTER);

typedef NPairClusterBinReaxFFCluster<1, 1, 1, 1, 1> NPairGroupHalfSizeBinAtomonlyNewtonTriReaxFFCluster;
NPairStyle(half/size/bin/atomonly/newton/tri/reaxff/cluster,
           NPairGroupHalfSizeBinAtomonlyNewtonTriReaxFFCluster,
           NP_HALF | NP_SIZE | NP_BIN | NP_ATOMONLY | NP_NEWTON | NP_TRI| NP_CLUSTER);
// clang-format on
#else

#ifndef LMP_NPAIR_CLUSTER_BIN_REAXFF_H
#define LMP_NPAIR_CLUSTER_BIN_REAXFF_H
#include "my_page.h"
#include "npair.h"
#include "cluster_neigh.h"
#include <stdint.h>

namespace LAMMPS_NS {
template <int HALF, int NEWTON, int TRI, int SIZE, int ATOMONLY>
class NPairClusterBinReaxFFCluster : public NPair {
 public:
  NPairClusterBinReaxFFCluster(class LAMMPS *);
  ~NPairClusterBinReaxFFCluster() override;
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
