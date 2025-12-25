#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace chetverikova_e_sobel {

struct ImageData {
  std::vector<int> pixels;  // Одномерный массив пикселей
  int width{};              // Ширина изображения
  int height{};             // Высота изображения
  int channels{};           // Количество каналов (1 для grayscale, 3 для RGB)
};
using InType = ImageData;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chetverikova_e_sobel
