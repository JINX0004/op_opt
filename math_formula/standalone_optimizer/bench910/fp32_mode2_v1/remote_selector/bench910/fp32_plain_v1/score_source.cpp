#include "model/hardware_profile.h"
#include <cstdint>
#include <iomanip>
#include <iostream>

// Input: M N K mode occupiedCores coreM coreN coreK baseM baseN baseK
//        stepK cubeFlopPerUs. Score a source-emitted FP32 tile under an
// exploratory timing profile; no source tile is substituted by the solver.
int main() {
    std::uint64_t m, n, k, cores, cm, cn, ck, bm, bn, bk, step;
    int mode;
    double cube_rate;
    std::cout << std::fixed << std::setprecision(6);
    while (std::cin >> m >> n >> k >> mode >> cores >> cm >> cn >> ck
                    >> bm >> bn >> bk >> step >> cube_rate) {
        auto hardware = mmopt::Calibrated910B3();
        hardware.primitives["cube"].rate_per_us = cube_rate;
        mmopt::Workload workload{m, n, k, "fp32"};
        mmopt::PlanDecision decision;
        decision.family = mode == 0 ? mmopt::KernelFamily::Base :
                          mmopt::KernelFamily::DeterministicSplitK;
        decision.used_cores = cores;
        decision.single_core_m = cm;
        decision.single_core_n = cn;
        decision.single_core_k = ck;
        decision.base_m = bm;
        decision.base_n = bn;
        decision.base_k = bk;
        decision.step_k = step;
        mmopt::CoreMemory memory;
        if (mode == 0) memory.base_l0c_buffers = 1;
        try {
            const auto score = mmopt::ReducedObjectiveUs(workload, hardware,
                                                         decision, memory);
            std::cout << m << ',' << n << ',' << k << ',' << mode << ','
                      << cube_rate << ',' << score << '\n';
        } catch (const std::exception& e) {
            std::cerr << m << 'x' << n << 'x' << k << ": " << e.what() << '\n';
            return 1;
        }
    }
}
