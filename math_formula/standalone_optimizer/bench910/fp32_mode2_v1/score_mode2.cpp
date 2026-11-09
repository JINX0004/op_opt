#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    std::cout << "index,source_mode2_us,search_mode2_us,base_us,deterministic_us,search_mode2_m,search_mode2_n,search_mode2_k,search_mode2_cores\n";
    std::uint64_t index, m, n, k, cores, cm, cn, ck, bm, bn, bk, sm, sn, sk;
    const auto hardware = mmopt::ExperimentalFp32910B3();
    const auto domain = mmopt::SourceCompatibleDomain();
    std::cout << std::fixed << std::setprecision(5);
    while (std::cin >> index >> m >> n >> k >> cores >> cm >> cn >> ck
                    >> bm >> bn >> bk >> sm >> sn >> sk) {
        mmopt::PlanDecision source;
        source.family = mmopt::KernelFamily::SingleCoreSplitKMode2;
        source.used_cores = cores;
        source.single_core_m = cm;
        source.single_core_n = cn;
        source.single_core_k = ck;
        source.base_m = bm;
        source.base_n = bn;
        source.base_k = bk;
        source.step_m = sm;
        source.step_n = sn;
        source.step_k = sk;
        const mmopt::Workload workload{m, n, k, "fp32"};
        const auto score = mmopt::ReducedObjectiveUs(workload, hardware, source);
        const auto search = mmopt::SolveReducedMip(workload, hardware, {}, domain);
        const auto& mode2 = search.single_core_split_k_mode2;
        std::cout << index << ',' << score << ',';
        if (mode2) std::cout << mode2->surrogate_us;
        std::cout << ',';
        if (search.base) std::cout << search.base->surrogate_us;
        std::cout << ',';
        if (search.deterministic_split_k) std::cout << search.deterministic_split_k->surrogate_us;
        std::cout << ',';
        if (mode2) {
            const auto& d = mode2->decision;
            std::cout << d.single_core_m << ',' << d.single_core_n << ','
                      << d.single_core_k << ',' << d.used_cores;
        }
        std::cout << '\n';
    }
}
