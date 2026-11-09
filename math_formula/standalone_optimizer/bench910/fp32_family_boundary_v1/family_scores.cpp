#include "model/hardware_profile.h"

#include <cstdint>
#include <iostream>

int main() {
    std::cout << "m,n,k,base_us,split_us\n";
    std::uint64_t m, n, k;
    while (std::cin >> m >> n >> k) {
        const auto result = mmopt::SolveReducedMip(
            {m, n, k, "fp32"}, mmopt::ExperimentalFp32910B3(), {},
            mmopt::SourceCompatibleDomain());
        std::cout << m << ',' << n << ',' << k << ',';
        if (result.base) std::cout << result.base->surrogate_us;
        std::cout << ',';
        if (result.deterministic_split_k)
            std::cout << result.deterministic_split_k->surrogate_us;
        std::cout << '\n';
    }
}
