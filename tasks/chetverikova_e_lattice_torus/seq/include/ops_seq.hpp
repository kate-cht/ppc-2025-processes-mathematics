#pragma once
#include "chetverikova_e_lattice_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chetverikova_e_lattice_torus {

class ChetverikovaELatticeTorusSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChetverikovaELatticeTorusSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chetverikova_e_lattice_torus
