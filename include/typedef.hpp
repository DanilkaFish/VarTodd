#pragma once

// #include <bitvector.hpp>
// #include <boost/dynatim_bitset>
#include <bit>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace todd {
using index_t   = long unsigned int;
using pivot_map = std::map<index_t, index_t>;
template <typename T> constexpr T k_single_sentinel() { return std::numeric_limits<T>::max(); }
} // namespace todd