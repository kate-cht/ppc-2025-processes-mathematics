#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include "chetverikova_e_lattice_torus/common/include/common.hpp"
#include "chetverikova_e_lattice_torus/mpi/include/ops_mpi.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace chetverikova_e_lattice_torus {

class ChetverikovaERunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init == 0) {
      GTEST_SKIP() << "MPI in not init";
      return;
    }
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
      GTEST_SKIP() << "Number of processes is less than 2";
      return;
    }
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_chetverikova_e_lattice_torus, "perf.txt");
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: ");
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
    }
    if (!out_data.empty()) {
      return false;
    }
    if (!path.empty()) {
      return false;
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
    ppc::util::MakeAllPerfTasks<InType, ChetverikovaELatticeTorusMPI>(PPC_SETTINGS_chetverikova_e_lattice_torus);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChetverikovaERunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(TorusTopologyPerfTests, ChetverikovaERunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace chetverikova_e_lattice_torus
