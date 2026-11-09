// Score audited MatMulV3 source tiles with the current FP32 reduced objective.
#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mmopt;
    const auto hardware = ExperimentalFp32Mode2Pipeline910B3();
    std::uint64_t m, n, k, mode, cores, core_m, core_n, core_k, base_m, base_n, base_k;
    std::cout << "m,n,k,mode,source_score_us\n" << std::fixed << std::setprecision(5);
    while (std::cin >> m >> n >> k >> mode >> cores >> core_m >> core_n >> core_k
                    >> base_m >> base_n >> base_k) {
        PlanDecision d;
        d.family = mode == 0 ? KernelFamily::Base :
                   mode == 3 ? KernelFamily::DeterministicSplitK :
                   throw std::invalid_argument("unsupported family");
        d.used_cores = cores;
        d.single_core_m = core_m;
        d.single_core_n = core_n;
        d.single_core_k = core_k;
        d.base_m = base_m;
        d.base_n = base_n;
        d.base_k = base_k;
        d.step_k = mode == 0 ? 1 : core_k / base_k;
        CoreMemory memory;
        if (mode == 0) memory.base_l0c_buffers = 1;
        try {
            std::cout << m << ',' << n << ',' << k << ',' << mode << ','
                      << ReducedObjectiveUs({m, n, k, "fp32"}, hardware, d,
                                            memory, mode == 0) << '\n';
        } catch (const std::exception& error) {
            std::cerr << "source tile rejected: " << m << 'x' << n << 'x' << k
                      << " mode=" << mode << ": " << error.what() << '\n';
            std::cout << m << ',' << n << ',' << k << ',' << mode << ",nan\n";
        }
    }
}
