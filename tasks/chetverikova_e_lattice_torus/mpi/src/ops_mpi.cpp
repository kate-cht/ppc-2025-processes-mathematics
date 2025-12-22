#include "chetverikova_e_lattice_torus/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "chetverikova_e_lattice_torus/common/include/common.hpp"

namespace chetverikova_e_lattice_torus {

ChetverikovaELatticeTorusMPI::ChetverikovaELatticeTorusMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // Инициализируем пустыми данными
  GetOutput() = std::make_tuple(std::vector<double>{}, std::vector<int>{});
}

bool ChetverikovaELatticeTorusMPI::ValidationImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  return (std::get<0>(GetInput()) >= 0) && (std::get<0>(GetInput()) < world_size_) && 
         (std::get<1>(GetInput()) >= 0) && (std::get<1>(GetInput()) < world_size_) && 
         !(std::get<2>(GetInput())).empty();
}

bool ChetverikovaELatticeTorusMPI::PreProcessingImpl() {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    
    DetermineGridDimensions();
    return rows_ * cols_ == world_size_;
}

void ChetverikovaELatticeTorusMPI::DetermineGridDimensions() {
    int bestRows = 1;
    int minDiff = world_size_;
    for (int r = 1; r * r <= world_size_; ++r) {
        if (world_size_ % r == 0) {
            int c = world_size_ / r;
            int diff = std::abs(r - c);
            if (diff < minDiff) {
                minDiff = diff;
                bestRows = r;
            }
        }
    }
    rows_ = bestRows;
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

  // Сначала двигаемся по столбцам
  if (curr_col != dest_col) {
    int dir = GetOptimalDirection(curr_col, dest_col, cols_);
    return GetRank(curr_row, curr_col + dir);
  }

  // Затем по строкам
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

bool ChetverikovaELatticeTorusMPI::RunImpl() {
  int start = 0;
  int end = 0;
  
  // Распространяем информацию о отправителе и получателе
  if (rank_ == 0) {
    start = std::get<0>(GetInput());  // отправитель
    end = std::get<1>(GetInput());    // получатель
  }
  MPI_Bcast(&start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&end, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Вычисляем путь для всех процессов
  std::vector<int> path = ComputeFullPath(start, end);
  
  // Ищем текущий процесс в пути
  auto it = std::find(path.begin(), path.end(), rank_);
  bool is_on_path = (it != path.end());
  
  // Подготавливаем данные для приема/отправки
  std::vector<double> recv_data;
  
  if (rank_ == start) {
    // Отправитель
    recv_data = std::get<2>(GetInput());
    
    if (start != end) {
      // Отправляем размер данных
      int data_size = static_cast<int>(recv_data.size());
      int next_node = path[1];
      MPI_Send(&data_size, 1, MPI_INT, next_node, 0, MPI_COMM_WORLD);
      
      // Отправляем сами данные
      if (data_size > 0) {
        MPI_Send(recv_data.data(), data_size, MPI_DOUBLE, next_node, 1, MPI_COMM_WORLD);
      }
    }
  } 
  else if (is_on_path && rank_ != start) {
    // Промежуточный узел или получатель
    int index = static_cast<int>(std::distance(path.begin(), it));
    int prev_node = path[index - 1];
    
    // Получаем размер данных
    int recv_size = 0;
    MPI_Recv(&recv_size, 1, MPI_INT, prev_node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Получаем данные
    if (recv_size > 0) {
      recv_data.resize(recv_size);
      MPI_Recv(recv_data.data(), recv_size, MPI_DOUBLE, prev_node, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Если не получатель - пересылаем дальше
    if (rank_ != end && (index + 1) < static_cast<int>(path.size())) {
      int next_node = path[index + 1];
      int data_size = static_cast<int>(recv_data.size());
      MPI_Send(&data_size, 1, MPI_INT, next_node, 0, MPI_COMM_WORLD);
      
      if (data_size > 0) {
        MPI_Send(recv_data.data(), data_size, MPI_DOUBLE, next_node, 1, MPI_COMM_WORLD);
      }
    }
  }

  // Устанавливаем выходные данные
  if (rank_ == end) {
    // Только получатель сохраняет данные и путь
    GetOutput() = std::make_tuple(std::move(recv_data), std::move(path));
  } else {
    // Все остальные процессы имеют пустые выходные данные
    GetOutput() = std::make_tuple(std::vector<double>{}, std::vector<int>{});
  }
  
  // Синхронизация перед завершением
  MPI_Barrier(MPI_COMM_WORLD);
  
  return true;
}

bool ChetverikovaELatticeTorusMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chetverikova_e_lattice_torus