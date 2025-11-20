#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace chetverikova_e_sum_matrix_elem {

using InType = std::tuple<uint32_t, uint32_t, std::vector<double>>;
using OutType = double;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chetverikova_e_sum_matrix_elem
