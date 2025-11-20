#include "chetverikova_e_sum_matrix_elem/seq/include/ops_seq.hpp"

#include <cstddef>

#include "chetverikova_e_sum_matrix_elem/common/include/common.hpp"

namespace chetverikova_e_sum_matrix_elem {

ChetverikovaESumMatrixElemSEQ::ChetverikovaESumMatrixElemSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ChetverikovaESumMatrixElemSEQ::ValidationImpl() {
  return ((std::get<0>(GetInput()) > 0) && (std::get<1>(GetInput()) > 0) &&
          (static_cast<size_t>(std::get<0>(GetInput())) * static_cast<size_t>(std::get<1>(GetInput())) ==
           std::get<2>(GetInput()).size()) &&
          (!(std::get<2>(GetInput()).empty())) && (GetOutput() == 0.0));
}

bool ChetverikovaESumMatrixElemSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return (GetOutput() == 0.0);
}

bool ChetverikovaESumMatrixElemSEQ::RunImpl() {
  OutType &res = GetOutput();
  const auto &matrix = std::get<2>(GetInput());

  for (double value : matrix) {
    res += value;
  }
  return true;
}

bool ChetverikovaESumMatrixElemSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chetverikova_e_sum_matrix_elem
