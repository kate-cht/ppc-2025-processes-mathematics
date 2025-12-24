#pragma once
#include "chetverikova_e_sobel/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chetverikova_e_sobel {

class ChetverikovaESobelSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChetverikovaESobelSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chetverikova_e_sobel
