#include "chetverikova_e_lattice_torus/seq/include/ops_seq.hpp"
#include <tuple>
#include <vector>
#include "chetverikova_e_lattice_torus/common/include/common.hpp"

namespace chetverikova_e_lattice_torus {

ChetverikovaELatticeTorusSEQ::ChetverikovaELatticeTorusSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(std::vector<double>{}, std::vector<int>{});
}

bool ChetverikovaELatticeTorusSEQ::ValidationImpl() {
  return true;
}

bool ChetverikovaELatticeTorusSEQ::PreProcessingImpl() {
  return true;
}

bool ChetverikovaELatticeTorusSEQ::RunImpl() {
  return true;
}

bool ChetverikovaELatticeTorusSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chetverikova_e_lattice_torus
