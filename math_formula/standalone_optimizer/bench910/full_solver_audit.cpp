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
    if (argc > 1 && (std::string(argv[1]) == "converted_a_select" ||
                     std::string(argv[1]) == "converted_a_source" ||
                     std::string(argv[1]) == "converted_a_core_m")) {
        // Compare the two source-emitted geometries from the long-K phase
        // sweep. The 2-GiB cap is only for diagnostic scoring: the source
        // Split-K tuple needs 530 MiB, above the normal 512-MiB model cap.
        hardware.workspace_limit_bytes = 2ULL * 1024 * 1024 * 1024;
        mmopt::CoreMemory base_memory;
        base_memory.base_l0c_buffers = 1;
        mmopt::PlanDecision base;
        base.family = mmopt::KernelFamily::Base;
        base.used_cores = 20;
        base.single_core_m = base.base_m = 128;
        base.single_core_n = base.base_n = 256;
        base.base_k = 64;
        mmopt::PlanDecision split;
        split.family = mmopt::KernelFamily::DeterministicSplitK;
        split.used_cores = 20;
        split.single_core_m = 2560;
        split.single_core_n = 384;
        split.single_core_k = 384;
        split.base_m = split.base_n = split.base_k = 128;
        split.step_k = 3;
        if (std::string(argv[1]) == "converted_a_select") {
            // Match the measured-output primitive profile used by the
            // local FP16 family audit. Report both the online-style
            // conservative L0C budget and source BASE's one-buffer budget.
            hardware.workspace_limit_bytes = 512ULL * 1024 * 1024;
            auto& p = hardware.primitives;
            p["sc_fp32_store"] = {"bytes", 0.2432, 34190.0};
            p["sc_fp32_store_steady"] = {"bytes", 0.0, 91000.0};
            p["sc_atomic_extra"] = {"bytes", 0.0541, 106618.0};
            p["sc_cast"] = {"elements", 0.0156, 116718.0};
            p["sc_gm_to_l1_a"] = {"bytes", 0.3124, 270810.0};
            p["sc_gm_to_l1_b"] = {"bytes", 0.0227, 115414.0};
            mmopt::MipTileDomain domain;
            domain.base_core_equals_base = true;
            const mmopt::Workload w{2560, 1024, 73728, "fp16"};
            const auto family = [](const std::optional<mmopt::MipPlan>& plan) {
                return plan ? plan->surrogate_us : -1.0;
            };
            std::cout << "base_l0c_buffers,base_us,split_us,mode2_us,mode5_us,mode6_us,winner\n";
            for (const auto buffers : {2ULL, 1ULL}) {
                mmopt::CoreMemory memory;
                memory.base_l0c_buffers = buffers;
                const auto result = mmopt::SolveReducedMip(w, hardware, memory, domain);
                std::cout << buffers << ',' << family(result.base) << ','
                          << family(result.deterministic_split_k) << ','
                          << family(result.single_core_split_k_mode2) << ','
                          << family(result.single_core_split_k_mode5) << ','
                          << family(result.single_core_split_k_mode6) << ','
                          << (result.winner ? result.winner->candidate.family : "none")
                          << '\n';
            }
            return 0;
        }
        if (std::string(argv[1]) == "converted_a_core_m") {
            const mmopt::Workload w{2560, 1024, 73728, "fp16"};
            std::cout << "step_k,core_m,core_n,workspace_bytes,model_us,within_512_mib\n";
            for (const auto step_k : {1ULL, 3ULL}) {
                split.step_k = step_k;
                split.single_core_k = 128 * step_k;
                for (const auto core_n : {128ULL, 384ULL}) {
                    for (const auto core_m : {128ULL, 256ULL, 512ULL, 1280ULL, 2560ULL}) {
                        split.single_core_m = core_m;
                        split.single_core_n = core_n;
                        const auto workspace = mmopt::FinalizePlan(
                            w, hardware, split, "core_m_audit").workspace_bytes;
                        const auto score = mmopt::ReducedObjectiveUs(w, hardware, split);
                        std::cout << step_k << ',' << core_m << ',' << core_n << ','
                                  << workspace << ',' << score << ','
                                  << (workspace <= 512ULL * 1024 * 1024) << '\n';
                    }
                }
            }
            return 0;
        }
        std::cout << "k,base_us,split_us,model_winner\n";
        for (const auto k : {65408ULL, 65536ULL, 73728ULL}) {
            const mmopt::Workload w{2560, 1024, k, "fp16"};
            base.single_core_k = k;
            const auto base_us = mmopt::ReducedObjectiveUs(w, hardware, base, base_memory);
            const auto split_us = mmopt::ReducedObjectiveUs(w, hardware, split);
            std::cout << k << ',' << base_us << ',' << split_us << ','
                      << (base_us < split_us ? "BASE" : "SPLIT_K") << '\n';
        }
        return 0;
    }
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
