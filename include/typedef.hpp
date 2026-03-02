#pragma once

#include <cassert>
#include <map>

namespace todd {
using index_t   = long unsigned int;
using pivot_map = std::map<index_t, index_t>;
template <typename T> constexpr T k_single_sentinel() { return std::numeric_limits<T>::max(); }
} // namespace todd