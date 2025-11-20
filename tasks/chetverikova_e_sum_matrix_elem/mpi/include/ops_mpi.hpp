#pragma once
#include "chetverikova_e_sum_matrix_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chetverikova_e_sum_matrix_elem {

class ChetverikovaESumMatrixElemMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChetverikovaESumMatrixElemMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chetverikova_e_sum_matrix_elem
