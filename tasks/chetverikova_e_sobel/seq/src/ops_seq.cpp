#include "chetverikova_e_sobel/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace chetverikova_e_sobel {

namespace {
// Ядра Собеля (упрощенная инициализация)
const int kernel_sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
const int kernel_sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Функция свертки Собеля (как в MPI версии)
int ConvSobel(const std::vector<int> &data, int idx, int width) {
  int gx = kernel_sobel_x[0][0] * data[idx - width - 1] + kernel_sobel_x[0][1] * data[idx - width] +
           kernel_sobel_x[0][2] * data[idx - width + 1] +

           kernel_sobel_x[1][0] * data[idx - 1] + kernel_sobel_x[1][1] * data[idx] +
           kernel_sobel_x[1][2] * data[idx + 1] +

           kernel_sobel_x[2][0] * data[idx + width - 1] + kernel_sobel_x[2][1] * data[idx + width] +
           kernel_sobel_x[2][2] * data[idx + width + 1];

  int gy = kernel_sobel_y[0][0] * data[idx - width - 1] + kernel_sobel_y[0][1] * data[idx - width] +
           kernel_sobel_y[0][2] * data[idx - width + 1] +

           kernel_sobel_y[1][0] * data[idx - 1] + kernel_sobel_y[1][1] * data[idx] +
           kernel_sobel_y[1][2] * data[idx + 1] +

           kernel_sobel_y[2][0] * data[idx + width - 1] + kernel_sobel_y[2][1] * data[idx + width] +
           kernel_sobel_y[2][2] * data[idx + width + 1];

  double magnitude = std::sqrt(static_cast<double>(gx * gx + gy * gy));
  magnitude = std::min(255.0, std::max(0.0, magnitude));

  return static_cast<int>(std::round(magnitude));
}

std::vector<int> ApplySobelOperator(const std::vector<int> &image, int width, int height) {
  std::vector<int> result(width * height, 0);

  // Если изображение слишком маленькое, возвращаем нули
  if (width < 3 || height < 3) {
    return result;
  }

  // Обрабатываем только внутренние пиксели (как в MPI версии)
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int idx = y * width + x;
      result[idx] = ConvSobel(image, idx, width);
    }
  }

  // Граничные пиксели уже установлены в 0 (по умолчанию)
  // Это соответствует MPI версии

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

  if (input.pixels.size() != expected_size) {
    return false;
  }

  return true;
}

bool ChetverikovaESobelSEQ::PreProcessingImpl() {
  // Убираем все выводы в консоль - они замедляют выполнение!
  return true;
}

bool ChetverikovaESobelSEQ::RunImpl() {
  const ImageData &input = GetInput();

  // 1. Проверяем, нужно ли преобразовывать в оттенки серого
  std::vector<int> gray_image;

  if (input.channels == 1) {
    // Уже в градациях серого
    gray_image = input.pixels;
  } else {
    // Конвертируем в градации серого
    gray_image.resize(input.width * input.height);

    for (int y = 0; y < input.height; ++y) {
      for (int x = 0; x < input.width; ++x) {
        int src_idx = (y * input.width + x) * input.channels;
        int dst_idx = y * input.width + x;

        if (input.channels >= 3) {
          // RGB изображение
          int r = input.pixels[src_idx];
          int g = input.pixels[src_idx + 1];
          int b = input.pixels[src_idx + 2];

          // Та же формула, что и в MPI версии
          int gray = static_cast<int>((0.299 * r) + (0.587 * g) + (0.114 * b));
          gray_image[dst_idx] = gray;
        } else if (input.channels == 2) {
          // Вероятно, grayscale + alpha
          gray_image[dst_idx] = input.pixels[src_idx];
        }
      }
    }
  }

  // 2. Применяем оператор Собеля
  std::vector<int> result = ApplySobelOperator(gray_image, input.width, input.height);

  // 3. Сохраняем результат
  GetOutput() = std::move(result);

  return true;
}

bool ChetverikovaESobelSEQ::PostProcessingImpl() {
  // Убираем вывод в консоль!
  const auto &output = GetOutput();
  const auto &input = GetInput();

  // Проверяем размер
  if (output.size() != static_cast<std::size_t>(input.width * input.height)) {
    return false;
  }

  return true;
}

}  // namespace chetverikova_e_sobel
