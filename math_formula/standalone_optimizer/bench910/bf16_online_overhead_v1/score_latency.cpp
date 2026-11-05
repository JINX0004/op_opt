#include "matmul_v3_mip_online.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    using mmopt::PlanDecision;
    setenv("MMV3_ONLINE_BF16_EXPERIMENT", "1", 1);
    std::uint64_t m, n, k;
    std::cout << std::fixed << std::setprecision(3);
    while (std::cin >> m >> n >> k) {
        PlanDecision base;
        base.family = mmopt::KernelFamily::Base;
        base.base_m = base.single_core_m = std::min<std::uint64_t>(128, m);
        base.base_n = base.single_core_n = std::min<std::uint64_t>(256, n);
        base.base_k = 64;
        base.single_core_k = k;
        base.used_cores = std::min<std::uint64_t>(20,
            ((m + base.base_m - 1) / base.base_m) * ((n + base.base_n - 1) / base.base_n));
        const auto start = Clock::now();
        auto choice = optiling::matmul_v3::LookupOnlineMipMode(m, n, k, "bf16", &base);
        const auto cold = Clock::now();
        constexpr int reps = 101;
        std::vector<double> hits;
        for (int i = 0; i < reps; ++i) {
            const auto a = Clock::now();
            auto hit = optiling::matmul_v3::LookupOnlineMipMode(m, n, k, "bf16", &base);
            const auto b = Clock::now();
            if (hit.mode != choice.mode) return 2;
            hits.push_back(std::chrono::duration<double, std::micro>(b-a).count());
        }
        std::sort(hits.begin(), hits.end());
        std::cout << m << ' ' << n << ' ' << k << ' ' << choice.mode << ' '
                  << std::chrono::duration<double, std::micro>(cold-start).count() << ' '
                  << hits[reps/2] << '\n';
    }
}
