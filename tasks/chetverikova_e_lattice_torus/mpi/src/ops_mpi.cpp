#include "chetverikova_e_lattice_torus/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

#include "chetverikova_e_lattice_torus/common/include/common.hpp"

namespace chetverikova_e_lattice_torus {

ChetverikovaELatticeTorusMPI::ChetverikovaELatticeTorusMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(std::vector<double>{}, std::vector<int>{});
}

bool ChetverikovaELatticeTorusMPI::ValidationImpl() {
  return true;
}

bool ChetverikovaELatticeTorusMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  DetermineGridDimensions();
  return rows_ * cols_ == world_size_;
}

void ChetverikovaELatticeTorusMPI::DetermineGridDimensions() {
  int best_rows = 1;
  int min_diff = world_size_;
  for (int row = 1; row * row <= world_size_; ++row) {
    if (world_size_ % row == 0) {
      int c = world_size_ / row;
      int diff = std::abs(row - c);
      if (diff < min_diff) {
        min_diff = diff;
        best_rows = row;
      }
    }
  }
  rows_ = best_rows;
  cols_ = world_size_ / rows_;
}

int ChetverikovaELatticeTorusMPI::GetRank(int row, int col) const {
  row = ((row % rows_) + rows_) % rows_;
  col = ((col % cols_) + cols_) % cols_;
  return (row * cols_) + col;
}

int ChetverikovaELatticeTorusMPI::GetOptimalDirection(int start, int end, int size) {
  int forward = (end - start + size) % size;
  int backward = (start - end + size) % size;
  return (forward <= backward) ? 1 : -1;
}

int ChetverikovaELatticeTorusMPI::ComputeNextNode(int curr, int end) const {
  if (curr == end) {
    return -1;
  }

  int curr_row = curr / cols_;
  int curr_col = curr % cols_;
  int dest_row = end / cols_;
  int dest_col = end % cols_;

  if (curr_col != dest_col) {
    int dir = GetOptimalDirection(curr_col, dest_col, cols_);
    return GetRank(curr_row, curr_col + dir);
  }

  if (curr_row != dest_row) {
    int dir = GetOptimalDirection(curr_row, dest_row, rows_);
    return GetRank(curr_row + dir, curr_col);
  }

  return -1;
}

std::vector<int> ChetverikovaELatticeTorusMPI::ComputeFullPath(int start, int end) const {
  std::vector<int> path;
  path.push_back(start);
  int curr = start;
  while (curr != end) {
    int next = ComputeNextNode(curr, end);
    if (next == -1) {
      break;
    }
    path.push_back(next);
    curr = next;
  }
  return path;
}

std::vector<double> ChetverikovaELatticeTorusMPI::ReceiveData(int sender) {
  int recv_size = 0;
  MPI_Recv(&recv_size, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> data;
  if (recv_size > 0) {
    data.resize(recv_size);
    MPI_Recv(data.data(), recv_size, MPI_DOUBLE, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return data;
}

void ChetverikovaELatticeTorusMPI::SendDataToNext(const std::vector<double> &data, int next_node) {
  int data_size = static_cast<int>(data.size());
  MPI_Send(&data_size, 1, MPI_INT, next_node, 0, MPI_COMM_WORLD);

  if (data_size > 0) {
    MPI_Send(data.data(), data_size, MPI_DOUBLE, next_node, 1, MPI_COMM_WORLD);
  }
}

bool ChetverikovaELatticeTorusMPI::RunImpl() {
  int start = 0;
  int end = 0;

  if (rank_ == 0) {
    start = std::get<0>(GetInput());  // отправитель
    end = std::get<1>(GetInput());    // получатель
  }
  MPI_Bcast(&start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&end, 1, MPI_INT, 0, MPI_COMM_WORLD);
  bool not_correct = start >= world_size_ || end >= world_size_ || start < 0 || end < 0;
  if (not_correct) {
    GetOutput() = std::make_tuple(std::vector<double>{}, std::vector<int>{});
    return true;
  }

  std::vector<int> path = ComputeFullPath(start, end);
  auto it = std::ranges::find(path, rank_);
  bool is_on_path = (it != path.end());

  std::vector<double> recv_data;

  if (rank_ == start) {
    recv_data = std::get<2>(GetInput());
    if (start != end) {
      SendDataToNext(recv_data, path[1]);
    }
  } else if (is_on_path) {
    int index = static_cast<int>(std::distance(path.begin(), it));
    int prev_node = path[index - 1];

    recv_data = ReceiveData(prev_node);

    if (rank_ != end && (index + 1) < static_cast<int>(path.size())) {
      int next_node = path[index + 1];
      SendDataToNext(recv_data, next_node);
    }
  }
  if (rank_ == end) {
    GetOutput() = std::make_tuple(std::move(recv_data), std::move(path));
  } else {
    GetOutput() = std::make_tuple(std::vector<double>{}, std::vector<int>{});
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ChetverikovaELatticeTorusMPI::PostProcessingImpl() {
  return true;
}
}  // namespace chetverikova_e_lattice_torus
