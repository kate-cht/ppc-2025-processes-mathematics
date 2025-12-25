#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "chetverikova_e_sobel/common/include/common.hpp"
#include "chetverikova_e_sobel/mpi/include/ops_mpi.hpp"
#include "chetverikova_e_sobel/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chetverikova_e_sobel {

class ChetverikovaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 private:
  InType input_data_;
  OutType expected_data_{};

 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init == 0) {
      GTEST_SKIP() << "MPI in not init";
      return;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = params + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chetverikova_e_sobel, filename);

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    int tmp{};

    if (!(file >> input_data_.width >> input_data_.height >> input_data_.channels)) {
      throw std::runtime_error("Failed to read required parameters");
    }

    while (file >> tmp) {
      input_data_.pixels.push_back(tmp);
    }

    file.close();

    std::size_t expected_size = static_cast<std::size_t>(input_data_.width) *
                                static_cast<std::size_t>(input_data_.height) *
                                static_cast<std::size_t>(input_data_.channels);
    if (input_data_.pixels.size() != expected_size) {
      throw std::runtime_error("Pixel count mismatch. Expected: " + std::to_string(expected_size) +
                               ", got: " + std::to_string(input_data_.pixels.size()));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::size_t expected_output_size =
        static_cast<std::size_t>(input_data_.width) * static_cast<std::size_t>(input_data_.height);

    if (output_data.size() != expected_output_size) {
      return false;
    }

    if (input_data_.width < 3 || input_data_.height < 3) {
      if (output_data.size() != input_data_.pixels.size()) {
        return false;
      }
      for (std::size_t i = 0; i < output_data.size(); ++i) {
        if (output_data[i] != input_data_.pixels[i]) {
          return false;
        }
      }
      return true;
    }

    const std::size_t width = static_cast<std::size_t>(input_data_.width);
    const std::size_t height = static_cast<std::size_t>(input_data_.height);

    for (std::size_t i = 0; i < output_data.size(); ++i) {
      const std::size_t row = i / width;
      const std::size_t col = i % width;

      if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {
        if (output_data[i] != 0) {
          return false;
        }
      }
    }

    bool has_non_zero = false;

    for (int row = 1; row < input_data_.height - 1; ++row) {
      for (int col = 1; col < input_data_.width - 1; ++col) {
        std::size_t idx =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(input_data_.width) + static_cast<std::size_t>(col);
        if (output_data[idx] != 0) {
          has_non_zero = true;
        }
      }
    }

    if (params == "test2" && !has_non_zero) {
      return false;
    }

    if (params == "test1" && input_data_.width == 4 && input_data_.height == 4) {
      const std::array<std::size_t, 4> expected_indices = {5, 6, 9, 10};

      for (std::size_t idx : expected_indices) {
        if (output_data[idx] != 33) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(ChetverikovaERunFuncTestsProcesses, SummOfMatrixElements) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::string("test1"), std::string("test2"), std::string("test3"),
                                            std::string("test4")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChetverikovaESobelMPI, InType>(kTestParam, PPC_SETTINGS_chetverikova_e_sobel),
    ppc::util::AddFuncTask<ChetverikovaESobelSEQ, InType>(kTestParam, PPC_SETTINGS_chetverikova_e_sobel));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChetverikovaERunFuncTestsProcesses::PrintFuncTestName<ChetverikovaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SumMatrixElemFuncTests, ChetverikovaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chetverikova_e_sobel
