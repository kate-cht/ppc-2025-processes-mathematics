#include "chetverikova_e_sobel/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "chetverikova_e_sobel/common/include/common.hpp"

namespace chetverikova_e_sobel {

namespace {
constexpr std::array<std::array<int, 3>, 3> kKernelSobelX = {{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};
constexpr std::array<std::array<int, 3>, 3> kKernelSobelY = {{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

int ConvSobel(const std::vector<int> &local_data, int i, int width) {
  int gx = (kKernelSobelX[0][0] * local_data[static_cast<std::size_t>(i - width - 1)]) +
           (kKernelSobelX[0][1] * local_data[static_cast<std::size_t>(i - width)]) +
           (kKernelSobelX[0][2] * local_data[static_cast<std::size_t>(i - width + 1)]) +

           (kKernelSobelX[1][0] * local_data[static_cast<std::size_t>(i - 1)]) +
           (kKernelSobelX[1][1] * local_data[static_cast<std::size_t>(i)]) +
           (kKernelSobelX[1][2] * local_data[static_cast<std::size_t>(i + 1)]) +

           (kKernelSobelX[2][0] * local_data[static_cast<std::size_t>(i + width - 1)]) +
           (kKernelSobelX[2][1] * local_data[static_cast<std::size_t>(i + width)]) +
           (kKernelSobelX[2][2] * local_data[static_cast<std::size_t>(i + width + 1)]);

  int gy = (kKernelSobelY[0][0] * local_data[static_cast<std::size_t>(i - width - 1)]) +
           (kKernelSobelY[0][1] * local_data[static_cast<std::size_t>(i - width)]) +
           (kKernelSobelY[0][2] * local_data[static_cast<std::size_t>(i - width + 1)]) +

           (kKernelSobelY[1][0] * local_data[static_cast<std::size_t>(i - 1)]) +
           (kKernelSobelY[1][1] * local_data[static_cast<std::size_t>(i)]) +
           (kKernelSobelY[1][2] * local_data[static_cast<std::size_t>(i + 1)]) +

           (kKernelSobelY[2][0] * local_data[static_cast<std::size_t>(i + width - 1)]) +
           (kKernelSobelY[2][1] * local_data[static_cast<std::size_t>(i + width)]) +
           (kKernelSobelY[2][2] * local_data[static_cast<std::size_t>(i + width + 1)]);

  double magnitude = std::sqrt(static_cast<double>((gx * gx) + (gy * gy)));
  magnitude = std::min(255.0, std::max(0.0, magnitude));
  return static_cast<int>(std::round(magnitude));
}

std::vector<int> ApplySobelOperatorLocal(const std::vector<int> &local_data, int width, int local_height) {
  int processed_height = local_height - 2;
  if (processed_height <= 0) {
    return {};
  }

  std::size_t result_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(processed_height);
  std::vector<int> result(result_size, 0);

  for (int j = 1; j < local_height - 1; ++j) {
    for (int i = 1; i < width - 1; ++i) {
      int input_idx = (j * width) + i;
      int output_idx = ((j - 1) * width) + i;

      int sobel_value = ConvSobel(local_data, input_idx, width);
      result[static_cast<std::size_t>(output_idx)] = sobel_value;
    }
  }

  return result;
}
}  // namespace

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
  const ImageData *input = nullptr;

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

  std::size_t world_size_t = static_cast<std::size_t>(world_size);
  std::vector<int> lines_on_proc(world_size_t, 0);
  std::vector<int> displ(world_size_t, 0);
  std::vector<int> final_displ(world_size_t, 0);
  std::vector<int> counts(world_size_t, 0);
  std::vector<int> final_counts(world_size_t, 0);

  int temp = 0;
  int temp_final = width;
  int lines_on_process = (height - 2) / world_size;
  int rem = (height - 2) % world_size;

  for (int i = 0; i < world_size; ++i) {
    auto idx = static_cast<std::size_t>(i);
    lines_on_proc[idx] = lines_on_process;
    if (i < rem) {
      lines_on_proc[idx]++;
    }

    displ[idx] = temp;
    final_displ[idx] = temp_final;
    temp += lines_on_proc[idx] * width;
    counts[idx] = (lines_on_proc[idx] + 2) * width;
    final_counts[idx] = lines_on_proc[idx] * width;
    temp_final += final_counts[idx];
  }

  auto local_data_size = static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)]);
  std::vector<int> local_data(local_data_size, 0);

  std::size_t res_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<int> res(res_size, 0);

  MPI_Scatterv(rank == 0 ? all_pixels.data() : nullptr, counts.data(), displ.data(), MPI_INT, local_data.data(),
               counts[static_cast<std::size_t>(rank)], MPI_INT, 0, MPI_COMM_WORLD);

  int local_total_rows = counts[static_cast<std::size_t>(rank)] / width;
  std::vector<int> res_proc = ApplySobelOperatorLocal(local_data, width, local_total_rows);

  MPI_Gatherv(res_proc.data(), final_counts[static_cast<std::size_t>(rank)], MPI_INT, rank == 0 ? res.data() : nullptr,
              final_counts.data(), final_displ.data(), MPI_INT, 0, MPI_COMM_WORLD);

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
