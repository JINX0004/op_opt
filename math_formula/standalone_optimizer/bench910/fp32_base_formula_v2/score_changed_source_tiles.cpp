// Score the source-emitted FP32 tiles in the seven changed decisions.
#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mmopt;
    const auto hardware = ExperimentalFp32BaseMte2910B3();
    std::uint64_t m, n, k, mode, cores, cm, cn, ck, bm, bn, bk, sm, sn;
    std::cout << "m,n,k,mode,source_score_us\n" << std::fixed
              << std::setprecision(5);
    while (std::cin >> m >> n >> k >> mode >> cores >> cm >> cn >> ck >> bm >> bn >> bk >> sm >> sn) {
        PlanDecision d;
        d.family = mode == 2 ? KernelFamily::SingleCoreSplitKMode2 :
                   mode == 3 ? KernelFamily::DeterministicSplitK :
                   throw std::invalid_argument("unsupported family");
        d.used_cores = cores;
        d.single_core_m = cm;
        d.single_core_n = cn;
        d.single_core_k = ck;
        d.base_m = bm;
        d.base_n = bn;
        d.base_k = bk;
        d.step_k = ck / bk;
        d.step_m = sm;
        d.step_n = sn;
        CoreMemory memory;
        try {
            const double score = ReducedObjectiveUs({m, n, k, "fp32"}, hardware, d, memory);
            std::cout << m << ',' << n << ',' << k << ',' << mode << ',' << score << '\n';
        } catch (const std::exception& error) {
            std::cerr << "source tile rejected: " << error.what() << '\n';
            std::cout << m << ',' << n << ',' << k << ',' << mode << ",nan\n";
        }
    }
}
