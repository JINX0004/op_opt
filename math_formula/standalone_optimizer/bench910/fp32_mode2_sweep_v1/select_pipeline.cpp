#include "hardware_profile.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    using namespace mmopt;
    const auto h = ExperimentalFp32Mode2Pipeline910B3();
    const auto domain = SourceCompatibleDomain();
    std::uint64_t m, n, k;
    while (std::cin >> m >> n >> k) {
        const auto result = SolveReducedMip({m, n, k, "fp32"}, h, {}, domain);
        std::cout << m << 'x' << n << 'x' << k << std::fixed << std::setprecision(3);
        if (result.base) std::cout << " BASE=" << result.base->surrogate_us;
        if (result.deterministic_split_k)
            std::cout << " deterministic=" << result.deterministic_split_k->surrogate_us;
        if (result.single_core_split_k_mode2)
            std::cout << " mode2=" << result.single_core_split_k_mode2->surrogate_us;
        if (result.winner) {
            const auto& d = result.winner->decision;
            const char* family = d.family == KernelFamily::Base ? "BASE" :
                d.family == KernelFamily::DeterministicSplitK ? "deterministic" :
                d.family == KernelFamily::SingleCoreSplitKMode2 ? "mode2" :
                d.family == KernelFamily::SingleCoreSplitKMode5 ? "mode5" :
                "mode6";
            std::cout << " winner=" << family
                      << " tile=" << d.single_core_m << 'x'
                      << d.single_core_n << 'x' << d.single_core_k;
        }
        std::cout << '\n';
    }
}
