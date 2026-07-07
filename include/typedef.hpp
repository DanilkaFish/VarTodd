#pragma once

#include <cassert>
#include <cstddef>
#include <limits>

namespace todd {
using index_t = std::size_t;
template <typename T> constexpr T k_single_sentinel() { return std::numeric_limits<T>::max(); }
} // namespace todd
