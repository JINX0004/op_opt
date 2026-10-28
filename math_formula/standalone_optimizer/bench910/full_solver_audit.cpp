#include "mip_solver.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

int main(int argc, char** argv) {
    // Exploratory probe slopes only. The present reduced equation fails
    // absolute-time validation; these plans are an audit, not deployable
    // recommendations. Split-K-only fields remain provisional.
    mmopt::Hardware hardware{
        20, 32ULL * 1024 * 1024,
        {{"launch", {"count", 2.68, 1.0}},
         {"gm_read", {"bytes", 0.109620, 3710212.0}},
         {"gm_write", {"bytes", 0.162586, 5229502.0}},
         {"cube", {"flop", 0.0, 14719432.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}},
         {"split_partial", {"count", 1.28, 1.0}}}};
    const std::vector<std::tuple<int, int, int>> small_shapes = {
        {128, 256, 1024}, {128, 256, 4096},
        {256, 256, 1024}, {256, 256, 4096},
        {256, 512, 1024}, {256, 512, 4096},
        {512, 256, 1024}, {512, 256, 4096},
        {512, 512, 1024}, {512, 512, 4096},
        {1024, 512, 4096}, {1024, 1024, 4096},
    };
    const std::vector<std::tuple<int, int, int>> large_shapes = {
        {2048, 2048, 4096}, {2048, 4096, 8192},
        {4096, 2048, 8192}, {4096, 4096, 4096},
        {4096, 4096, 8192}, {8192, 4096, 4096},
        {8192, 8192, 4096},
    };
    std::vector<std::tuple<int, int, int>> stdin_shapes;
    if (argc > 1 && std::string(argv[1]) == "stdin") {
        int m = 0, n = 0, k = 0;
        while (std::cin >> m >> n >> k) stdin_shapes.emplace_back(m, n, k);
    }
    const auto& shapes = argc > 1 && std::string(argv[1]) == "stdin"
                             ? stdin_shapes :
                         argc > 1 && std::string(argv[1]) == "large"
                             ? large_shapes : small_shapes;
    const std::string base_mode = argc > 2 ? argv[2] : "";
    const bool source_base = base_mode == "source_base";
    const bool unrestricted_base = base_mode == "unrestricted_base";
    const bool packet_profile = argc > 3 && std::string(argv[3]) == "packet_profile";
    const bool source_split_grid = argc > 5 &&
        std::string(argv[5]) == "source_split_grid";
    // Preserve the original 32 MiB default. Conversion-heavy K>65535 cases
    // need a caller-declared larger workspace; source device runs below use
    // hundreds of MiB and must not be compared to a 32 MiB model budget.
    if (argc > 4) {
        const auto mib = std::stoull(argv[4]);
        hardware.workspace_limit_bytes = mib * 1024ULL * 1024ULL;
    }
    mmopt::CoreMemory memory;
    mmopt::MipTileDomain domain;
    if (source_base || unrestricted_base) {
        // Observed plain BASE key 0x10000 writes only one base M/N tile per
        // output tile; source default can use one FP32 L0C buffer. The
        // unrestricted mode is a mathematical diagnostic only: the source
        // kernel is not correct for core M/N greater than base M/N.
        memory.base_l0c_buffers = 1;
        domain.base_core_equals_base = source_base;
    }
    if (packet_profile) {
        if (!source_base) {
            std::cerr << "packet profile requires source_base\n";
            return 2;
        }
        // Independent primitive fits only; no whole-kernel correction.
        hardware.primitives["base_mte2_a"] =
            {"bytes", 0.19097384635481152, 1024.0 / 0.003532113558081016};
        hardware.primitives["base_mte2_b"] =
            {"bytes", 0.07348386212366087, 1024.0 / 0.003017312080586143};
        hardware.primitives["base_mte2_a_rows"] =
            {"rows", 0.0, 1.0 / 0.000854946401991721};
        hardware.primitives["base_mte2_b_rows"] =
            {"rows", 0.0, 1.0 / 0.001272028070525036};
        // This bounded grid avoids extrapolating the packet response to
        // tiny L0 tiles. The source override will validate the winner.
        domain.single_core_m = {64, 128, 256};
        domain.single_core_n = {64, 128, 256};
        domain.base_m = domain.single_core_m;
        domain.base_n = domain.single_core_n;
    }
    for (auto [m, n, k] : shapes) {
        try {
            auto case_domain = domain;
            if (source_split_grid) {
                // Add the host's 3x128 Split-K outer tile and full output
                // axes. BASE's core=base restriction still filters these
                // from unsupported plain-BASE candidates.
                case_domain.single_core_m.push_back(384);
                case_domain.single_core_n.push_back(384);
                case_domain.single_core_m.push_back(m);
                case_domain.single_core_n.push_back(n);
            }
            auto result = mmopt::SolveReducedMip(
                mmopt::Workload{static_cast<std::uint64_t>(m),
                                static_cast<std::uint64_t>(n),
                                static_cast<std::uint64_t>(k), "fp16"},
                hardware, memory, case_domain);
            std::cout << m << ',' << n << ',' << k;
            if (result.base) {
                const auto& d = result.base->decision;
                std::cout << ",BASE," << d.single_core_m << ',' << d.single_core_n
                          << ',' << d.base_m << ',' << d.base_n << ','
                          << d.base_k << ',' << d.used_cores << ','
                          << result.base->surrogate_us;
            } else {
                std::cout << ",BASE_INFEASIBLE";
            }
            if (result.winner) {
                const auto& d = result.winner->decision;
                std::cout << ",WINNER," << result.winner->candidate.family << ','
                          << d.single_core_m << ',' << d.single_core_n << ','
                          << d.base_m << ',' << d.base_n << ','
                          << d.base_k << ',' << d.used_cores << ','
                          << result.winner->surrogate_us;
            }
            if (argc > 4 && result.deterministic_split_k) {
                const auto& d = result.deterministic_split_k->decision;
                std::cout << ",SPLIT," << d.single_core_m << ',' << d.single_core_n
                          << ',' << d.base_m << ',' << d.base_n << ',' << d.base_k
                          << ',' << d.step_k << ',' << d.used_cores << ','
                          << result.deterministic_split_k->surrogate_us;
            }
            if (source_split_grid && m >= 128 && n >= 128 && n % 16 == 0) {
                // Diagnostic value of the source pattern seen in the
                // K-boundary device audit. This is not a general host-tiler
                // prediction for arbitrary shapes.
                mmopt::PlanDecision reference;
                reference.family = mmopt::KernelFamily::DeterministicSplitK;
                reference.used_cores = 20;
                reference.single_core_m = 384;
                reference.single_core_n = n;
                reference.base_m = reference.base_n = reference.base_k = 128;
                reference.step_k = 3;
                reference.single_core_k = 384;
                try {
                    const auto value = mmopt::ReducedObjectiveUs(
                        mmopt::Workload{static_cast<std::uint64_t>(m),
                                        static_cast<std::uint64_t>(n),
                                        static_cast<std::uint64_t>(k), "fp16"},
                        hardware, reference, memory);
                    std::cout << ",SOURCE_PATTERN," << value;
                } catch (const std::invalid_argument&) {
                    std::cout << ",SOURCE_PATTERN_INFEASIBLE";
                }
            }
            std::cout << '\n';
        } catch (const std::exception& error) {
            std::cout << m << ',' << n << ',' << k << ",ERROR," << error.what() << '\n';
        }
    }
}
