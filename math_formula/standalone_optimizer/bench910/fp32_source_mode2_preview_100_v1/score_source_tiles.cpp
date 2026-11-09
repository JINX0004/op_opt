// Score the exact source-emitted FP32 tiles used in the paired experiment.
#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mmopt;
    const Hardware hardware = ExperimentalFp32BaseMte2910B3();
    std::uint64_t m, n, k, mode, cores, core_m, core_n, core_k;
    std::uint64_t base_m, base_n, base_k, step_m, step_n, step_k;
    std::cout << std::fixed << std::setprecision(5);
    while (std::cin >> m >> n >> k >> mode >> cores >> core_m >> core_n >> core_k
                    >> base_m >> base_n >> base_k >> step_m >> step_n >> step_k) {
        PlanDecision d;
        d.family = mode == 0 ? KernelFamily::Base :
                   mode == 2 ? KernelFamily::SingleCoreSplitKMode2 :
                   mode == 3 ? KernelFamily::DeterministicSplitK :
                   throw std::invalid_argument("unsupported family");
        d.used_cores = cores;
        d.single_core_m = core_m;
        d.single_core_n = core_n;
        d.single_core_k = core_k;
        d.base_m = base_m;
        d.base_n = base_n;
        d.base_k = base_k;
        d.step_m = step_m;
        d.step_n = step_n;
        d.step_k = step_k;
        CoreMemory memory;
        if (mode == 0) memory.base_l0c_buffers = 1;
        if (mode == 2) memory.source_emitted_mode2 = true;
        try {
            const double score = ReducedObjectiveUs({m, n, k, "fp32"},
                                                    hardware, d, memory,
                                                    mode == 0);
            std::cout << m << ' ' << n << ' ' << k << ' ' << mode << ' '
                      << score << '\n';
        } catch (const std::exception& error) {
            std::cerr << m << 'x' << n << 'x' << k << " mode " << mode
                      << ": " << error.what() << '\n';
            std::cout << m << ' ' << n << ' ' << k << ' ' << mode << " nan\n";
        }
    }
}
