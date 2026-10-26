#include "mip_solver.h"

#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

int main(int argc, char** argv) {
    // Primitive coefficients from packet_model_analysis.json. This is a
    // reproducible one-shape diagnostic, not a calibrated kernel profile.
    const mmopt::Hardware hardware{
        20, 32ULL * 1024 * 1024,
        {{"launch", {"count", 2.68, 1.0}},
         {"gm_read", {"bytes", 0.109620, 3710212.0}},
         {"gm_write", {"bytes", 0.162586, 5229502.0}},
         {"cube", {"flop", 0.0, 14719432.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}},
         {"base_mte2_a", {"bytes", 0.19097384635481152,
                          1024.0 / 0.003532113558081016}},
         {"base_mte2_b", {"bytes", 0.07348386212366087,
                          1024.0 / 0.003017312080586143}},
         {"base_mte2_a_rows", {"rows", 0.0,
                               1.0 / 0.000854946401991721}},
         {"base_mte2_b_rows", {"rows", 0.0,
                               1.0 / 0.001272028070525036}}}};
    mmopt::CoreMemory memory;
    memory.base_l0c_buffers = 1;
    using Case = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                            double, double>;
    std::vector<Case> cases = {
        {2048, 2048, 4096, 139.792, 161.178},
        {2048, 4096, 8192, 500.419, 613.983},
        {4096, 2048, 8192, 501.609, 625.737},
        {4096, 4096, 4096, 499.276, 589.771},
        {4096, 4096, 8192, 999.265, 1224.728},
        {8192, 4096, 4096, 998.312, 1163.548},
        {8192, 8192, 4096, 1985.454, 2345.877},
    };
    if (argc == 4) {
        cases = {{static_cast<std::uint64_t>(std::stoull(argv[1])),
                  static_cast<std::uint64_t>(std::stoull(argv[2])),
                  static_cast<std::uint64_t>(std::stoull(argv[3])), 0.0, 0.0}};
    }
    std::cout << "m,n,k,default_model_us,swapped_model_us,"
                 "default_measured_us,swapped_measured_us\n";
    std::cout << std::fixed << std::setprecision(6);
    for (const auto [m, n, k, actual_default, actual_swapped] : cases) {
        const mmopt::Workload workload{m, n, k, "fp16"};
        const auto estimate = [&](std::uint64_t base_m, std::uint64_t base_n) {
            mmopt::PlanDecision decision;
            decision.used_cores = 20;
            decision.single_core_m = decision.base_m = base_m;
            decision.single_core_n = decision.base_n = base_n;
            decision.single_core_k = workload.k;
            decision.base_k = 64;
            return mmopt::ReducedObjectiveUs(workload, hardware,
                                             decision, memory);
        };
        std::cout << m << ',' << n << ',' << k << ','
                  << estimate(128, 256) << ',' << estimate(256, 128)
                  << ',' << actual_default << ',' << actual_swapped << '\n';
    }
}
