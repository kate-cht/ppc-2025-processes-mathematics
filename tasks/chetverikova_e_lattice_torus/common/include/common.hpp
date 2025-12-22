#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace chetverikova_e_lattice_torus {

using InType = std::tuple<int, int, std::vector<double>>; // отправитель, получатель, сообщение
using OutType = std::tuple<std::vector<double>, std::vector<int>>; // сообщение, путь
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chetverikova_e_lattice_torus
