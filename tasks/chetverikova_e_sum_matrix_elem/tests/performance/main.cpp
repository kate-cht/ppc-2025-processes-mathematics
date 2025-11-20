#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

#include "chetverikova_e_sum_matrix_elem/common/include/common.hpp"
#include "chetverikova_e_sum_matrix_elem/mpi/include/ops_mpi.hpp"
#include "chetverikova_e_sum_matrix_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace chetverikova_e_sum_matrix_elem {

class ChetverikovaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_data_{};

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chetverikova_e_sum_matrix_elem, "perf_2000.bin");
    std::ifstream file(abs_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file");
    }
    // static_cast<std::streamsize>(total_elements * sizeof(double))
    size_t rows = 0;
    size_t cols = 0;
    if (!file.read(reinterpret_cast<char *>(&rows), sizeof(rows))) {
      throw std::runtime_error("Failed to read rows");
    }
    if (!file.read(reinterpret_cast<char *>(&cols), sizeof(cols))) {
      throw std::runtime_error("Failed to read columns");
    }
    if (!file.read(reinterpret_cast<char *>(&expected_data_), sizeof(expected_data_))) {
      throw std::runtime_error("Failed to read expected result");
    }

    std::get<0>(input_data_) = rows;
    std::get<1>(input_data_) = cols;

    size_t total_elements = rows * cols;
    std::get<2>(input_data_).resize(total_elements);
    if (!file.read(reinterpret_cast<char *>(std::get<2>(input_data_).data()),
                   static_cast<std::streamsize>(total_elements * sizeof(double)))) {
      throw std::runtime_error("Failed to read matrix data");
    }

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_data_) < 1e-2;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChetverikovaERunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChetverikovaESumMatrixElemMPI, ChetverikovaESumMatrixElemSEQ>(
        PPC_SETTINGS_chetverikova_e_sum_matrix_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChetverikovaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(SumMatrixElemPerfTests, ChetverikovaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace chetverikova_e_sum_matrix_elem
