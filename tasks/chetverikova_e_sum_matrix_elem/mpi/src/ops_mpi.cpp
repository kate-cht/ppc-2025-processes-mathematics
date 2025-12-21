#include "chetverikova_e_sum_matrix_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "chetverikova_e_sum_matrix_elem/common/include/common.hpp"

namespace chetverikova_e_sum_matrix_elem {

ChetverikovaESumMatrixElemMPI::ChetverikovaESumMatrixElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ChetverikovaESumMatrixElemMPI::ValidationImpl() {
  return (std::get<0>(GetInput()) > 0) && (std::get<1>(GetInput()) > 0) &&
         (static_cast<size_t>(std::get<0>(GetInput())) * static_cast<size_t>(std::get<1>(GetInput())) ==
          std::get<2>(GetInput()).size()) &&
         !(std::get<2>(GetInput()).empty()) && (GetOutput() == 0.0);
}

bool ChetverikovaESumMatrixElemMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return (GetOutput() == 0.0);
}

bool ChetverikovaESumMatrixElemMPI::RunImpl() {
  int rank_proc = 0;
  int size_proc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_proc);

  const auto &rows = std::get<0>(GetInput());
  const auto &columns = std::get<1>(GetInput());
  OutType &res = GetOutput();
  const auto &matrix = std::get<2>(GetInput());

  size_t size = static_cast<size_t>(rows) * static_cast<size_t>(columns);
  size_t elem_on_proc = size / size_proc;

  std::vector<double> local_data(elem_on_proc, 0);

  MPI_Scatter(matrix.data(), static_cast<int>(elem_on_proc), MPI_DOUBLE, local_data.data(),
              static_cast<int>(elem_on_proc), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  OutType res_proc{};
  for (size_t i = 0; i < elem_on_proc; ++i) {
    res_proc += local_data[i];
  }
  MPI_Reduce(&res_proc, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if ((rank_proc == 0) && (size % size_proc != 0)) {
    size_t tail_ind = size - (size % size_proc);
    for (size_t i = tail_ind; i < size; ++i) {
      res += matrix[i];
    }
  }
  MPI_Bcast(&res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ChetverikovaESumMatrixElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chetverikova_e_sum_matrix_elem
