#include "mip_solver.h"

#include <iomanip>
#include <iostream>
#include <limits>

int main() {
    // Freeze the existing 910B3 profile and source-in-context mode-6
    // FP32 output measurement before timing this mode-comparison suite.
    mmopt::Hardware hardware{
        20, 512ULL * 1024 * 1024,
        {{"launch", {"count", 2.68, 1.0}},
         {"gm_read", {"bytes", 0.109620, 3710212.0}},
         {"gm_write", {"bytes", 0.162586, 5229502.0}},
         {"cube", {"flop", 0.0, 14719432.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}},
         {"split_partial", {"count", 1.28, 1.0}},
         {"sc_fp32_store", {"bytes", 0.2432, 34190.0}},
         {"sc_fp32_store_steady", {"bytes", 0.0, 91000.0}},
         {"sc_atomic_extra", {"bytes", 0.0541, 106618.0}},
         {"sc_cast", {"elements", 0.0156, 116718.0}},
         {"sc_gm_to_l1_a", {"bytes", 0.3124, 270810.0}},
         {"sc_gm_to_l1_b", {"bytes", 0.0227, 115414.0}}}};
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "m,n,k,mode2_us,mode5_us,mode6_us,selected_mode\n";
    std::uint64_t m, n, k;
    while (std::cin >> m >> n >> k) {
        try {
            const auto result = mmopt::SolveReducedMip({m, n, k, "fp16"}, hardware);
            const auto& two = result.single_core_split_k_mode2;
            const auto& five = result.single_core_split_k_mode5;
            const auto& six = result.single_core_split_k_mode6;
            const double v2 = two ? two->surrogate_us : std::numeric_limits<double>::infinity();
            const double v5 = five ? five->surrogate_us : std::numeric_limits<double>::infinity();
            const double v6 = six ? six->surrogate_us : std::numeric_limits<double>::infinity();
            const int chosen = v2 <= v5 && v2 <= v6 ? 2 :
                               v5 <= v2 && v5 <= v6 ? 5 :
                               v6 < std::numeric_limits<double>::infinity() ? 6 : 0;
            std::cout << m << ',' << n << ',' << k << ',';
            if (two) std::cout << v2;
            std::cout << ',';
            if (five) std::cout << v5;
            std::cout << ',';
            if (six) std::cout << v6;
            std::cout << ',' << chosen << '\n';
        } catch (const std::exception& error) {
            std::cerr << m << 'x' << n << 'x' << k << ": " << error.what() << '\n';
            return 1;
        }
    }
}
