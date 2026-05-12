#pragma once
#include <cstddef>
#include <algorithm>

namespace LAMMPS_NS {
constexpr int CLUSTERSIZE = 8;

template <int ATOMONLY> class ClusterNeighEntry {
 public:
  int j;
  static constexpr int NWORD_MASK = (CLUSTERSIZE + 31) / 32;
  static constexpr int NWORD_SPECIAL = (CLUSTERSIZE * 2 * (!ATOMONLY) + 31) / 32;
  static constexpr int NWORD = NWORD_MASK + NWORD_SPECIAL;
  int mask_special[NWORD];
  __always_inline void init(int j)
  {
    this->j = j;
    std::fill_n(mask_special, NWORD, 0);
  }
  __always_inline void add_mask(int i)
  {
    static_assert(CLUSTERSIZE <= 32, "modify code to adopt larger cluster!");
    // __builtin_assume(i < CLUSTERSIZE);
    mask_special[0] |= 1L << i;
  }
  __always_inline void add_special(int i, int special) {
   static_assert(CLUSTERSIZE <= 16, "modify code to adopt larger cluster!");
  //  __builtin_assume(i < CLUSTERSIZE);
   if (!ATOMONLY)
      mask_special[1] |= (long)special << (i * 2);
  }
  __always_inline void add_i(int i, int special) {
   add_mask(i);
   if (!ATOMONLY) add_special(i, special);
  }
  __always_inline int get_mask(int i) {
    return mask_special[0] >> i & 1;
  }
  __always_inline int get_allmask(){
    return mask_special[0];
  }
  __always_inline int get_special(int i) {
    return (mask_special[1] >> (i * 2)) & 0x3;
  }
  __always_inline void get_allspecial(int out[8]) {
    if (!ATOMONLY) {
      for (int i = 0; i < 8; ++i)
        out[i] = get_special(i);
    } else {
      for (int i = 0; i < 8; ++i)
        out[i] = 0;
    }
  }
};
}