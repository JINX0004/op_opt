// Score source-emitted FP32 tiles with the same experimental online profile.
#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mmopt;
    const auto hardware = ExperimentalFp32BaseMte2910B3();
    std::uint64_t m, n, k, mode, used, cm, cn, ck, bm, bn, bk, sm, sn;
    std::cout << "m,n,k,mode,emitted_tile_score_us\n" << std::fixed << std::setprecision(5);
    while (std::cin >> m >> n >> k >> mode >> used >> cm >> cn >> ck >> bm >> bn >> bk >> sm >> sn) {
        PlanDecision d;
        d.family = mode == 0 ? KernelFamily::Base :
                   mode == 2 ? KernelFamily::SingleCoreSplitKMode2 :
                   mode == 3 ? KernelFamily::DeterministicSplitK :
                   throw std::invalid_argument("unsupported family");
        d.used_cores = used;
        d.single_core_m = cm;
        d.single_core_n = cn;
        d.single_core_k = ck;
        d.base_m = bm;
        d.base_n = bn;
        d.base_k = bk;
        d.step_m = sm;
        d.step_n = sn;
        d.step_k = mode == 0 ? 1 : ck / bk;
        CoreMemory memory;
        if (mode == 0) memory.base_l0c_buffers = 1;
        try {
            const auto score = ReducedObjectiveUs({m, n, k, "fp32"}, hardware,
                                                   d, memory, mode == 0);
            std::cout << m << ',' << n << ',' << k << ',' << mode << ',' << score << '\n';
        } catch (const std::exception& e) {
            std::cerr << "cannot score " << m << 'x' << n << 'x' << k << " mode="
                      << mode << ": " << e.what() << '\n';
            std::cout << m << ',' << n << ',' << k << ',' << mode << ",nan\n";
        }
    }
}
