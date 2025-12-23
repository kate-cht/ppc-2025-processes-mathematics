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

#include "chetverikova_e_lattice_torus/common/include/common.hpp"
#include "chetverikova_e_lattice_torus/mpi/include/ops_mpi.hpp"
#include "chetverikova_e_lattice_torus/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chetverikova_e_lattice_torus {

class ChetverikovaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 private:
  InType input_data_;
  // OutType expected_data_;

 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    if (world_size < 2) {
      GTEST_SKIP() << "Number of processes is less than 2";
    }
    std::string filename = params + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chetverikova_e_lattice_torus, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    double tmp{};
    if (!(file >> std::get<0>(input_data_) >> std::get<1>(input_data_))) {
      throw std::runtime_error("Failed to read required parameters");
    }
    while (file >> tmp) {
      std::get<2>(input_data_).push_back(tmp);
    }
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<double> out_data(std::get<0>(output_data));
    std::vector<double> in_data(std::get<2>(input_data_));
    std::vector<int> path(std::get<1>(output_data));
    int end = std::get<1>(input_data_);
    // int start = std::get<0>(input_data_);
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = test_info->name();
    bool is_seq_test = test_name.find("seq") != std::string::npos;
    // bool is_mpi_test = test_name.find("mpi") != std::string::npos;
    if (is_seq_test) {
      return true;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == end) {
      if (out_data.size() != in_data.size()) {
        return false;
      }
      if (path.empty()) {
        return false;
      }
      for (size_t i = 0; i < out_data.size(); i++) {
        if (std::abs(out_data[i] - in_data[i]) > 1e-7) {
          return false;
        }
      }
      return ((path.front() == std::get<0>(input_data_)) && (path.back() == std::get<1>(input_data_)));
    } else {
      if (!out_data.empty()) {
        return false;
      }

      if (!path.empty()) {
        return false;
      }
      return true;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(ChetverikovaERunFuncTestsProcesses, TopologyTorus) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::string("test_base"), std::string("test_neib"),
                                            std::string("test_border"), std::string("test_self_to_self")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ChetverikovaELatticeTorusMPI, InType>(
    kTestParam, PPC_SETTINGS_chetverikova_e_lattice_torus));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChetverikovaERunFuncTestsProcesses::PrintFuncTestName<ChetverikovaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(TopologyTorusFuncTests, ChetverikovaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chetverikova_e_lattice_torus
