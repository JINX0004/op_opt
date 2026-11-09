// Score one audited plain-FP32 BASE tile with both source objectives.
#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    using namespace mmopt;
    const auto baseline = ExperimentalFp32Mode2Pipeline910B3();
    const auto transfer = ExperimentalFp32BaseMte2910B3();
    std::uint64_t m, n, k, cores, cm, cn, ck, bm, bn, bk;
    std::cout << "m,n,k,baseline_us,transfer_us\n" << std::fixed
              << std::setprecision(5);
    while (std::cin >> m >> n >> k >> cores >> cm >> cn >> ck >> bm >> bn >> bk) {
        const Workload workload{m, n, k, "fp32"};
        PlanDecision d;
        d.family = KernelFamily::Base;
        d.used_cores = cores;
        d.single_core_m = cm;
        d.single_core_n = cn;
        d.single_core_k = ck;
        d.base_m = bm;
        d.base_n = bn;
        d.base_k = bk;
        d.step_k = 1;
        CoreMemory memory;
        memory.base_l0c_buffers = 1;
        std::cout << m << ',' << n << ',' << k << ','
                  << ReducedObjectiveUs(workload, baseline, d, memory, true)
                  << ',' << ReducedObjectiveUs(workload, transfer, d, memory, true)
                  << '\n';
    }
}
