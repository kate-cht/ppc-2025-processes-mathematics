#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "chetverikova_e_sobel/common/include/common.hpp"
#include "chetverikova_e_sobel/mpi/include/ops_mpi.hpp"
#include "chetverikova_e_sobel/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace chetverikova_e_sobel {

class ChetverikovaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_data_{};

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chetverikova_e_sobel, "perf.bin");
    std::ifstream file(abs_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    // Чтение размеров
    int width = 0;
    int height = 0;
    int channels = 0;

    if (!file.read(reinterpret_cast<char *>(&width), sizeof(width))) {
      throw std::runtime_error("Failed to read width");
    }
    if (!file.read(reinterpret_cast<char *>(&height), sizeof(height))) {
      throw std::runtime_error("Failed to read height");
    }
    if (!file.read(reinterpret_cast<char *>(&channels), sizeof(channels))) {
      throw std::runtime_error("Failed to read channels");
    }

    if (channels != 1) {
      throw std::runtime_error("Expected channels=1 for Sobel operator");
    }

    input_data_.width = width;
    input_data_.height = height;
    input_data_.channels = channels;

    int total_pixels = width * height * channels;
    input_data_.pixels.resize(total_pixels);

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    std::streamsize expected_size = sizeof(width) + sizeof(height) + sizeof(channels) + total_pixels * sizeof(int);

    if (file_size < expected_size) {
      throw std::runtime_error("File too small. Expected: " + std::to_string(expected_size) +
                               ", got: " + std::to_string(file_size));
    }

    file.seekg(sizeof(width) + sizeof(height) + sizeof(channels), std::ios::beg);

    if (!file.read(reinterpret_cast<char *>(input_data_.pixels.data()), total_pixels * sizeof(int))) {
      throw std::runtime_error("Failed to read pixel data");
    }

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_.width * input_data_.height)) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      size_t row = i / input_data_.width;
      size_t col = i % input_data_.width;

      if ((row == 0 || row == input_data_.height - 1 || col == 0 || col == input_data_.width - 1) &&
          output_data[i] != 0) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChetverikovaERunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChetverikovaESobelMPI>(PPC_SETTINGS_chetverikova_e_sobel);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChetverikovaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(SobelOperatorPerfTests, ChetverikovaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace chetverikova_e_sobel
