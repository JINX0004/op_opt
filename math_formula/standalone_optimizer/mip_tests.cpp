#include "mip_solver.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

mmopt::Hardware TestHardware() {
    return {20, 32ULL * 1024 * 1024,
            {{"launch", {"count", 0.2, 1.0}},
             {"gm_read", {"bytes", 0.005, 1600000.0}},
             {"gm_write", {"bytes", 0.005, 1600000.0}},
             {"cube", {"flop", 0.0, 14745600.0}},
             {"vector_add", {"ops", 0.0, 1000000.0}},
             {"nd2nz", {"elements", 0.0, 1000000.0}},
             {"sync", {"count", 0.1, 1.0}}}};
}

bool Equal(double a, double b) {
    return std::abs(a - b) <= 1e-5 * std::max({1.0, std::abs(a), std::abs(b)});
}

}  // namespace

int main() {
    const mmopt::Workload workload{16, 16, 102400, "fp16"};
    auto hardware = TestHardware();
    const auto result = mmopt::SolveReducedMip(workload, hardware);
    assert(result.base && result.deterministic_split_k && result.winner);
    assert(result.base->candidate.nd2nz_a);
    assert(result.deterministic_split_k->candidate.nd2nz_a);
    assert(result.base->candidate.tiling_key == "(0,0,0,0,0,0)");
    assert(result.deterministic_split_k->candidate.tiling_key == "(0,3,0,0,0,0)");
    assert(result.winner->event_graph_score &&
           std::isfinite(result.winner->event_graph_score->runtime_us));

    // Test-only exhaustive oracle: checks local closed-form decisions
    // against the independently evaluated reduced objective. Production
    // evaluates one cost per geometry (and Split-K core count).
    double best_base = std::numeric_limits<double>::infinity();
    for (std::uint64_t base_k = 16; base_k <= 1024; base_k += 16) {
        mmopt::PlanDecision d;
        d.used_cores = 1;
        d.single_core_m = d.base_m = 16;
        d.single_core_n = d.base_n = 16;
        d.single_core_k = workload.k;
        d.base_k = base_k;
        try {
            best_base = std::min(best_base,
                                 mmopt::ReducedObjectiveUs(workload, hardware, d));
        } catch (const std::invalid_argument&) {}
    }
    assert(Equal(best_base, result.base->surrogate_us));

    double best_split = std::numeric_limits<double>::infinity();
    for (std::uint64_t step = 1; step <= 32; ++step) {
        for (std::uint64_t cores = 2; cores <= 20; ++cores) {
            mmopt::PlanDecision d;
            d.family = mmopt::KernelFamily::DeterministicSplitK;
            d.used_cores = cores;
            d.single_core_m = d.base_m = 16;
            d.single_core_n = d.base_n = 16;
            d.base_k = 128;
            d.step_k = step;
            d.single_core_k = 128 * step;
            try {
                best_split = std::min(best_split,
                                      mmopt::ReducedObjectiveUs(workload, hardware, d));
            } catch (const std::invalid_argument&) {}
        }
    }
    assert(Equal(best_split, result.deterministic_split_k->surrogate_us));
    assert(Equal(result.winner->surrogate_us, std::min(best_base, best_split)));

    // A two-packet K dimension exposes the exact slice-count constraint:
    // stepK=2 has only one slice and cannot occupy two AICs.
    const mmopt::Workload short_k{16, 16, 256, "fp16"};
    const auto short_result = mmopt::SolveReducedMip(short_k, hardware);
    assert(short_result.deterministic_split_k);
    assert(short_result.deterministic_split_k->decision.used_cores == 2);
    assert(short_result.deterministic_split_k->decision.step_k == 1);

    // Independent M/N output and base blocks: verify the MILP against a
    // complete oracle on a deliberately small geometry domain.
    const mmopt::Workload blocked{64, 48, 256, "fp16"};
    const mmopt::MipTileDomain grid{{16, 32}, {16, 32}, {16, 32}, {16}};
    const auto block_result = mmopt::SolveReducedMip(blocked, hardware, {}, grid);
    assert(block_result.base && block_result.deterministic_split_k);
    assert(block_result.base->decision.single_core_m < blocked.m);
    assert(block_result.base->decision.single_core_n < blocked.n);
    double block_base_min = std::numeric_limits<double>::infinity();
    double block_split_min = std::numeric_limits<double>::infinity();
    for (const auto cm : grid.single_core_m)
        for (const auto cn : grid.single_core_n)
            for (const auto bm : grid.base_m) {
                if (bm > cm) continue;
                for (std::uint64_t bk = 16; bk <= blocked.k; bk += 16)
                    for (std::uint64_t c = 1; c <= 20; ++c) {
                        mmopt::PlanDecision d;
                        d.used_cores = c;
                        d.single_core_m = cm; d.single_core_n = cn;
                        d.base_m = bm; d.base_n = 16;
                        d.single_core_k = blocked.k; d.base_k = bk;
                        try {
                            block_base_min = std::min(block_base_min,
                                mmopt::ReducedObjectiveUs(blocked, hardware, d));
                        } catch (const std::invalid_argument&) {}
                    }
                for (std::uint64_t step = 1; step <= 2; ++step)
                    for (std::uint64_t c = 2; c <= 20; ++c) {
                        mmopt::PlanDecision d;
                        d.family = mmopt::KernelFamily::DeterministicSplitK;
                        d.used_cores = c;
                        d.single_core_m = cm; d.single_core_n = cn;
                        d.base_m = bm; d.base_n = 16;
                        d.base_k = 128; d.step_k = step;
                        d.single_core_k = 128 * step;
                        try {
                            block_split_min = std::min(block_split_min,
                                mmopt::ReducedObjectiveUs(blocked, hardware, d));
                        } catch (const std::invalid_argument&) {}
                    }
            }
    assert(Equal(block_base_min, block_result.base->surrogate_us));
    assert(Equal(block_split_min, block_result.deterministic_split_k->surrogate_us));

    // The deterministic kernel supplies two AIVs per occupied AIC. With
    // parallel reduction shards, this GM-bound case prefers all 20 cores;
    // a serial vector term would instead favor four.
    auto parallel_hardware = TestHardware();
    parallel_hardware.workspace_limit_bytes = 128ULL * 1024 * 1024;
    const mmopt::Workload parallel_shape{512, 512, 32768, "fp16"};
    const mmopt::MipTileDomain parallel_grid{{384}, {512}, {128}, {128}};
    const auto parallel_result = mmopt::SolveReducedMip(
        parallel_shape, parallel_hardware, {}, parallel_grid);
    assert(parallel_result.deterministic_split_k);
    assert(parallel_result.deterministic_split_k->decision.used_cores == 20);
    assert(parallel_result.deterministic_split_k->decision.step_k == 4);
    assert(Equal(parallel_result.deterministic_split_k->surrogate_us, 294.570465));
    auto four_cores = parallel_result.deterministic_split_k->decision;
    four_cores.used_cores = 4;
    assert(mmopt::ReducedObjectiveUs(parallel_shape, parallel_hardware, four_cores) >
           parallel_result.deterministic_split_k->surrogate_us);

    // If Cube service is negligible and extra cores save no read-call
    // latency, even parallel AIV reduction can favor the minimum two AICs.
    auto reduction_heavy = parallel_hardware;
    reduction_heavy.primitives["cube"].rate_per_us = 1.0e12;
    reduction_heavy.primitives["gm_read"].fixed_latency_us = 0.0;
    const auto reduction_result = mmopt::SolveReducedMip(
        parallel_shape, reduction_heavy, {}, parallel_grid);
    assert(reduction_result.deterministic_split_k);
    assert(reduction_result.deterministic_split_k->decision.used_cores == 2);

    const mmopt::MipTileDomain distinct_blocks{{32}, {32}, {16}, {16}};
    const auto distinct_result = mmopt::SolveReducedMip(blocked, hardware, {},
                                                       distinct_blocks);
    assert(distinct_result.base && distinct_result.base->event_graph_score);
    assert(distinct_result.base->decision.base_m == 16);
    assert(distinct_result.base->decision.single_core_m == 32);

    // A large spatial and K workload exercises BASE's independent core and
    // base tiles, multiple Cube cores, and the full default geometry grid.
    const mmopt::Workload multicore_shape{4096, 3072, 8192, "fp16"};
    auto compute_heavy_hardware = hardware;
    compute_heavy_hardware.primitives["cube"].rate_per_us = 1000000.0;
    const auto multicore_result = mmopt::SolveReducedMip(
        multicore_shape, compute_heavy_hardware);
    assert(multicore_result.base);
    const auto& multicore_base = multicore_result.base->decision;
    assert(multicore_base.used_cores == 20);
    assert(multicore_base.single_core_m > multicore_base.base_m);
    assert(multicore_base.single_core_n > multicore_base.base_n);
    auto one_core = multicore_base;
    one_core.used_cores = 1;
    assert(multicore_result.base->surrogate_us <
           mmopt::ReducedObjectiveUs(multicore_shape, compute_heavy_hardware,
                                     one_core));

    // The selected plan remains available if explicit events exceed the cap.
    const mmopt::Workload large{102400, 30720, 4096, "fp16"};
    const mmopt::MipTileDomain large_grid{{128}, {128}, {64}, {64}};
    const auto large_result = mmopt::SolveReducedMip(large, hardware, {}, large_grid);
    assert(large_result.winner);
    assert(large_result.winner->decision.single_core_m == 128);
    assert(large_result.winner->decision.base_m == 64);
    assert(!large_result.winner->event_graph_score);

    // A slower Cube makes multiple K owners valuable in the reduced model.
    hardware.primitives["cube"].rate_per_us = 100000.0;
    const auto compute_bound = mmopt::SolveReducedMip(workload, hardware);
    assert(compute_bound.winner &&
           compute_bound.winner->decision.family ==
               mmopt::KernelFamily::DeterministicSplitK);

    std::cout << "mmopt_mip_tests passed\n";
}
