#include "chetverikova_e_sobel/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "chetverikova_e_sobel/common/include/common.hpp"

namespace chetverikova_e_sobel {

namespace {
  constexpr std::array<std::array<int, 3>, 3> kernel_sobel_x = {{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};
  constexpr std::array<std::array<int, 3>, 3> kernel_sobel_y = {{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

  std::vector<int> ConvertToGray(const std::vector<int> &pixels, int width, int channels, int start_row,
                                        int num_rows) {
    const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(num_rows);
    std::vector<int> gray(size);

    if (channels == 1) {
      const int src_idx = start_row * width; 
      memcpy(gray.data(), pixels.data() + src_idx, width * sizeof(int) * num_rows);
    return gray;
    }

  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    const int src_y = start_row + row_idx;

    for (int col_idx = 0; col_idx < width; ++col_idx) {
      const int src_idx = ((src_y * width) + col_idx) * channels;
      const int r = pixels[src_idx];
      const int g = (channels > 1) ? pixels[src_idx + 1] : 0;
      const int b = (channels > 2) ? pixels[src_idx + 2] : 0;
      const int gray_idx = (row_idx * width) + col_idx;
      gray[gray_idx] = static_cast<int>((0.299 * r) + (0.587 * g) + (0.114 * b));
    }
  }
  return gray;
}

int ConvSobel(const std::vector<int> &local_data, int i, int width) {
  int gx = kernel_sobel_x[0][0] * local_data[i-width-1] + 
           kernel_sobel_x[0][1] * local_data[i-width] +   
           kernel_sobel_x[0][2] * local_data[i-width+1] +
           
           kernel_sobel_x[1][0] * local_data[i-1] + 
           kernel_sobel_x[1][1] * local_data[i] +          
           kernel_sobel_x[1][2] * local_data[i+1] +
           
           kernel_sobel_x[2][0] * local_data[i+width-1] + 
           kernel_sobel_x[2][1] * local_data[i+width] +   
           kernel_sobel_x[2][2] * local_data[i+width+1];

  int gy = kernel_sobel_y[0][0] * local_data[i-width-1] + 
           kernel_sobel_y[0][1] * local_data[i-width] +
           kernel_sobel_y[0][2] * local_data[i-width+1] +
           
           kernel_sobel_y[1][0] * local_data[i-1] + 
           kernel_sobel_y[1][1] * local_data[i] +     
           kernel_sobel_y[1][2] * local_data[i+1] +
           
           kernel_sobel_y[2][0] * local_data[i+width-1] + 
           kernel_sobel_y[2][1] * local_data[i+width] +
           kernel_sobel_y[2][2] * local_data[i+width+1];

  double magnitude = std::sqrt(static_cast<double>(gx * gx + gy * gy));
    magnitude = std::min(255.0, std::max(0.0, magnitude));
  
  return static_cast<int>(std::round(magnitude));
}

std::vector<int> ApplySobelOperatorLocal(const std::vector<int> &local_data, int width, int local_height) {
  int processed_height = local_height - 2;
  if (processed_height <= 0) {
    return {};
  }
  std::vector<int> result(width * processed_height, 0);
  for (int j = 1; j < local_height - 1; ++j) {
    for (int i = 1; i < width - 1; ++i) {
      int input_idx = j * width + i;
      int output_idx = (j - 1) * width + i; 

      int sobel_value = ConvSobel(local_data, input_idx, width);
      result[output_idx] = sobel_value;
    }
  }
  
  return result;
}
} //namespace

ChetverikovaESobelMPI::ChetverikovaESobelMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ChetverikovaESobelMPI::ValidationImpl() {
  return true;
}

bool ChetverikovaESobelMPI::PreProcessingImpl() {
  return true;
}
bool ChetverikovaESobelMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = 0;
  int height = 0;
  std::vector<int> all_pixels;
  const ImageData* input = nullptr;

  
  input = &GetInput();
  width = input->width;
  height = input->height;
  all_pixels = input->pixels;
  if (all_pixels.empty()) {
    if (rank == 0) {
        return false;
      }
  }
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (width < 3 || height < 3) {
      GetOutput() = all_pixels;
      
      return true;
    }
  std::vector<int> lines_on_proc(world_size, 0);
  std::vector<int> displ(world_size, 0);
  std::vector<int> final_displ(world_size, 0);
  std::vector<int> counts(world_size, 0);
  std::vector<int> final_counts(world_size, 0);
  int temp = 0;
  int temp_final = width;
  int lines_on_process = (height - 2) / world_size;
  int rem = (height - 2) % world_size;
  for (int i = 0; i < world_size; ++i) {
    lines_on_proc[i] = lines_on_process;
    if (i < rem) {
      lines_on_proc[i]++;
    }
    displ[i] = temp;
    final_displ[i] = temp_final;
    temp += lines_on_proc[i] * width;
    counts[i] = (lines_on_proc[i] + 2)* width;
    final_counts[i] = lines_on_proc[i] * width;
    temp_final += final_counts[i];
  }


  std::vector<int> local_data(counts[rank], 0);
  
  std::vector<int> res(width * (height), 0);
  
  MPI_Scatterv(
    rank == 0 ? all_pixels.data() : nullptr,
    counts.data(),
    displ.data(),
    MPI_INT,
    local_data.data(),
    counts[rank],
    MPI_INT,
    0,
    MPI_COMM_WORLD
  );
  
  
  int local_total_rows = counts[rank] / width;
  std::vector<int> res_proc = ApplySobelOperatorLocal(local_data, width, local_total_rows);
 
  MPI_Gatherv(
    res_proc.data(),
    final_counts[rank],
    MPI_INT,
    rank == 0 ? res.data() : nullptr,
    final_counts.data(),
    final_displ.data(),
    MPI_INT,
    0,
    MPI_COMM_WORLD
  );
  // Рассылаем результат
  int result_size = width * height;
  MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(res.data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = res;
  return true;
}
bool ChetverikovaESobelMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chetverikova_e_sobel
