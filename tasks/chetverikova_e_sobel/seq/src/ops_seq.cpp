#include "chetverikova_e_sobel/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "chetverikova_e_sobel/common/include/common.hpp"
namespace chetverikova_e_sobel {

namespace {
constexpr std::array<std::array<int, 3>, 3> kKernelSobelX = {{{{-1, 0, 1}}, {{-2, 0, 2}}, {{-1, 0, 1}}}};
constexpr std::array<std::array<int, 3>, 3> kKernelSobelY = {{{{-1, -2, -1}}, {{0, 0, 0}}, {{1, 2, 1}}}};

int ConvSobel(const std::vector<int> &data, std::size_t idx, int width) {
  int gx = (kKernelSobelX[0][0] * data[idx - static_cast<std::size_t>(width) - 1]) +
           (kKernelSobelX[0][1] * data[idx - static_cast<std::size_t>(width)]) +
           (kKernelSobelX[0][2] * data[idx - static_cast<std::size_t>(width) + 1]) +

           (kKernelSobelX[1][0] * data[idx - 1]) + (kKernelSobelX[1][1] * data[idx]) +
           (kKernelSobelX[1][2] * data[idx + 1]) +

           (kKernelSobelX[2][0] * data[idx + static_cast<std::size_t>(width) - 1]) +
           (kKernelSobelX[2][1] * data[idx + static_cast<std::size_t>(width)]) +
           (kKernelSobelX[2][2] * data[idx + static_cast<std::size_t>(width) + 1]);

  int gy = (kKernelSobelY[0][0] * data[idx - static_cast<std::size_t>(width) - 1]) +
           (kKernelSobelY[0][1] * data[idx - static_cast<std::size_t>(width)]) +
           (kKernelSobelY[0][2] * data[idx - static_cast<std::size_t>(width) + 1]) +

           (kKernelSobelY[1][0] * data[idx - 1]) + (kKernelSobelY[1][1] * data[idx]) +
           (kKernelSobelY[1][2] * data[idx + 1]) +

           (kKernelSobelY[2][0] * data[idx + static_cast<std::size_t>(width) - 1]) +
           (kKernelSobelY[2][1] * data[idx + static_cast<std::size_t>(width)]) +
           (kKernelSobelY[2][2] * data[idx + static_cast<std::size_t>(width) + 1]);

  double magnitude = std::sqrt(static_cast<double>((gx * gx) + (gy * gy)));
  magnitude = std::min(255.0, std::max(0.0, magnitude));

  return static_cast<int>(std::round(magnitude));
}

std::vector<int> ApplySobelOperator(const std::vector<int> &image, int width, int height) {
  std::size_t result_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<int> result(result_size, 0);

  if (width < 3 || height < 3) {
    return result;
  }

  for (int row_index = 1; row_index < height - 1; ++row_index) {
    for (int col_index = 1; col_index < width - 1; ++col_index) {
      std::size_t idx =
          (static_cast<std::size_t>(row_index) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col_index);
      result[idx] = ConvSobel(image, idx, width);
    }
  }
  return result;
}

}  // namespace

ChetverikovaESobelSEQ::ChetverikovaESobelSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ChetverikovaESobelSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.width <= 0 || input.height <= 0 || input.channels <= 0) {
    return false;
  }

  std::size_t expected_size = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height) *
                              static_cast<std::size_t>(input.channels);
  return input.pixels.size() == expected_size;
}

bool ChetverikovaESobelSEQ::PreProcessingImpl() {
  return true;
}

bool ChetverikovaESobelSEQ::RunImpl() {
  const ImageData &input = GetInput();

  std::vector<int> gray_image;

  if (input.channels == 1) {
    gray_image = input.pixels;
  } else {
    std::size_t image_size = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
    gray_image.resize(image_size);

    const int *src = input.pixels.data();
    int *dst = gray_image.data();
    int channels = input.channels;

    if (channels >= 3) {
      std::size_t height = static_cast<std::size_t>(input.height);
      std::size_t width = static_cast<std::size_t>(input.width);
      for (std::size_t row_idx = 0; row_idx < height; ++row_idx) {
        for (std::size_t col_idx = 0; col_idx < width; ++col_idx) {
          std::size_t src_idx =
              (row_idx * static_cast<std::size_t>(input.width) + col_idx) * static_cast<std::size_t>(channels);
          std::size_t dst_idx = (row_idx * static_cast<std::size_t>(input.width)) + col_idx;
          int r = src[src_idx];
          int g = src[src_idx + 1];
          int b = src[src_idx + 2];

          int gray = static_cast<int>((0.299 * r) + (0.587 * g) + (0.114 * b));
          dst[dst_idx] = gray;
        }
      }
    } else if (channels == 2) {
      for (std::size_t i = 0; i < image_size; ++i) {
        dst[i] = src[i * static_cast<std::size_t>(channels)];
      }
    }
  }

  std::vector<int> result = ApplySobelOperator(gray_image, input.width, input.height);

  GetOutput() = std::move(result);

  return true;
}

bool ChetverikovaESobelSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();
  const auto &input = GetInput();

  std::size_t expected_size = static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);

  return output.size() == expected_size;
}

}  // namespace chetverikova_e_sobel
