// Rerun the FP32 family selector with the opt-in BASE transfer service.
#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    using namespace mmopt;
    const auto hardware = ExperimentalFp32BaseMte2910B3();
    const auto domain = SourceCompatibleDomain();
    std::cout << "m,n,k,dtype,mode,objective_us,used_cores,singleCoreM,singleCoreN,"
                 "singleCoreK,baseM,baseN,baseK,stepM,stepN,stepK,base_us,"
                 "deterministic_us,mode2_us,mode5_us,mode6_us\n";
    std::cout << std::fixed << std::setprecision(5);
    std::uint64_t m, n, k;
    while (std::cin >> m >> n >> k) {
        const auto result = SolveReducedMip({m, n, k, "fp32"}, hardware, {}, domain);
        if (!result.winner) return 2;
        const auto& d = result.winner->decision;
        const auto mode = d.family == KernelFamily::Base ? 0 :
            d.family == KernelFamily::SingleCoreSplitKMode2 ? 2 :
            d.family == KernelFamily::DeterministicSplitK ? 3 :
            d.family == KernelFamily::SingleCoreSplitKMode5 ? 5 : 6;
        const auto score = [](const auto& plan) {
            return plan ? plan->surrogate_us : -1.0;
        };
        std::cout << m << ',' << n << ',' << k << ",fp32," << mode << ','
                  << result.winner->surrogate_us << ',' << d.used_cores << ','
                  << d.single_core_m << ',' << d.single_core_n << ','
                  << d.single_core_k << ',' << d.base_m << ',' << d.base_n << ','
                  << d.base_k << ',' << d.step_m << ',' << d.step_n << ','
                  << d.step_k << ',' << score(result.base) << ','
                  << score(result.deterministic_split_k) << ','
                  << score(result.single_core_split_k_mode2) << ','
                  << score(result.single_core_split_k_mode5) << ','
                  << score(result.single_core_split_k_mode6) << '\n';
    }
}
