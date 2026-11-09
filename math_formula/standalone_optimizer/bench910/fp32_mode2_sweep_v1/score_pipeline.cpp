#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    using namespace mmopt;
    const auto baseline = ExperimentalFp32910B3();
    const auto pipeline = ExperimentalFp32Mode2Pipeline910B3();
    std::cout << "m,n,k,core_m,core_n,baseline_us,pipeline_us\n";
    std::cout << std::fixed << std::setprecision(6);
    std::uint64_t m, n, k, cm, cn, ck, bm, bn, bk, sm, sn, sk, cores;
    while (std::cin >> m >> n >> k >> cm >> cn >> ck >> bm >> bn >> bk
                    >> sm >> sn >> sk >> cores) {
        PlanDecision tile;
        tile.family = KernelFamily::SingleCoreSplitKMode2;
        tile.used_cores = cores;
        tile.single_core_m = cm;
        tile.single_core_n = cn;
        tile.single_core_k = ck;
        tile.base_m = bm;
        tile.base_n = bn;
        tile.base_k = bk;
        tile.step_m = sm;
        tile.step_n = sn;
        tile.step_k = sk;
        const Workload workload{m, n, k, "fp32"};
        std::cout << m << ',' << n << ',' << k << ',' << cm << ',' << cn
                  << ',' << ReducedObjectiveUs(workload, baseline, tile)
                  << ',' << ReducedObjectiveUs(workload, pipeline, tile)
                  << '\n';
    }
}
