#include "mip_solver.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: mmopt_directional_read_check "
                     "a_fixed_us a_rate_bytes_per_us b_fixed_us b_rate_bytes_per_us\n";
        return 2;
    }
    try {
        const mmopt::Hardware hardware{
            20, 32ULL * 1024 * 1024,
            {{"launch", {"count", 2.68, 1.0}},
             {"gm_read", {"bytes", 0.109620, 3710212.0}},
             {"gm_read_a", {"bytes", std::stod(argv[1]), std::stod(argv[2])}},
             {"gm_read_b", {"bytes", std::stod(argv[3]), std::stod(argv[4])}},
             {"gm_write", {"bytes", 0.162586, 5229502.0}},
             {"cube", {"flop", 0.0, 14719432.0}},
             {"vector_add", {"ops", 0.0, 1000000.0}},
             {"nd2nz", {"elements", 0.0, 1000000.0}},
             {"sync", {"count", 0.1, 1.0}}}};
        mmopt::CoreMemory memory;
        memory.base_l0c_buffers = 1;
        const std::vector<std::tuple<int, int, int>> shapes = {
            {2048, 2048, 4096}, {2048, 4096, 8192},
            {4096, 2048, 8192}, {4096, 4096, 4096},
            {4096, 4096, 8192}, {8192, 4096, 4096},
            {8192, 8192, 4096},
        };
        std::cout << "m,n,k,default_128x256_us,swapped_256x128_us,"
                     "swapped_minus_default_us\n";
        std::cout << std::fixed << std::setprecision(6);
        for (const auto& [m, n, k] : shapes) {
            const mmopt::Workload workload{static_cast<std::uint64_t>(m),
                                          static_cast<std::uint64_t>(n),
                                          static_cast<std::uint64_t>(k), "fp16"};
            auto score = [&](std::uint64_t bm, std::uint64_t bn) {
                mmopt::PlanDecision decision;
                decision.family = mmopt::KernelFamily::Base;
                decision.used_cores = 20;
                decision.single_core_m = decision.base_m = bm;
                decision.single_core_n = decision.base_n = bn;
                decision.single_core_k = k;
                decision.base_k = 64;
                return mmopt::ReducedObjectiveUs(workload, hardware,
                                                 decision, memory);
            };
            const double source_default = score(128, 256);
            const double swapped = score(256, 128);
            std::cout << m << ',' << n << ',' << k << ',' << source_default
                      << ',' << swapped << ',' << swapped - source_default << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
