#pragma once
#include <vector>

#include "chetverikova_e_lattice_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chetverikova_e_lattice_torus {

class ChetverikovaELatticeTorusMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChetverikovaELatticeTorusMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DetermineGridDimensions();
  int GetRank(int row, int col) const;
  static int GetOptimalDirection(int start, int end, int size);
  int ComputeNextNode(int current, int dest) const;
  std::vector<int> ComputeFullPath(int source, int dest) const;

  int world_size_ = 0;
  int rank_ = 0;
  int rows_ = 0;
  int cols_ = 0;
};

}  // namespace chetverikova_e_lattice_torus
