#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "chetverikova_e_sum_matrix_elem/common/include/common.hpp"
#include "chetverikova_e_sum_matrix_elem/mpi/include/ops_mpi.hpp"
#include "chetverikova_e_sum_matrix_elem/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chetverikova_e_sum_matrix_elem {

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
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = params + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chetverikova_e_sum_matrix_elem, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    double tmp{};
    if (!(file >> std::get<0>(input_data_) >> std::get<1>(input_data_) >> expected_data_)) {
      throw std::runtime_error("Failed to read required parameters");
    }
    while (file >> tmp) {
      std::get<2>(input_data_).push_back(tmp);
    }
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    if (params == "test1" || params == "test5"|| params == "test7") {
      return std::abs(output_data - expected_data_) < 1e-12;
    }
    if (params == "test6") {
      return std::abs(output_data - expected_data_) < 1e-16;
    }
    if (params == "test2" || params == "test3" || params == "test4") {
      return std::abs(output_data - expected_data_) < 1e-6;
    }
    return false;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(ChetverikovaERunFuncTestsProcesses, SummOfMatrixElements) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7 > kTestParam = {std::string("test1"), std::string("test2"), 
  std::string("test3"), std::string("test4"), std::string("test5"), std::string("test6"),
  std::string("test7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ChetverikovaESumMatrixElemMPI, InType>(
                                               kTestParam, PPC_SETTINGS_chetverikova_e_sum_matrix_elem),
                                           ppc::util::AddFuncTask<ChetverikovaESumMatrixElemSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_chetverikova_e_sum_matrix_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChetverikovaERunFuncTestsProcesses::PrintFuncTestName<ChetverikovaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SumMatrixElemFuncTests, ChetverikovaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chetverikova_e_sum_matrix_elem
