#include "mip_solver.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

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

double NaiveBaseObjectiveUs(const mmopt::Workload& w,
                            const mmopt::Hardware& h,
                            const mmopt::PlanDecision& d) {
    // Independent small-shape oracle for the abstract balanced-class rule:
    // expand each tile, group by full/tail class, then rotate extras between
    // classes. Production uses only four class counts, never this expansion.
    const auto& read = h.primitives.at("gm_read");
    const auto a_it = h.primitives.find("gm_read_a");
    const auto b_it = h.primitives.find("gm_read_b");
    const auto& read_a = a_it == h.primitives.end() ? read : a_it->second;
    const auto& read_b = b_it == h.primitives.end() ? read : b_it->second;
    const auto& write = h.primitives.at("gm_write");
    const auto& cube = h.primitives.at("cube");
    std::vector<double> elements(d.used_cores, 0.0);
    std::vector<double> packets(d.used_cores, 0.0);
    std::vector<double> outputs(d.used_cores, 0.0);
    double a_bytes = 0.0, b_bytes = 0.0;
    std::array<std::vector<std::pair<std::uint64_t, std::uint64_t>>, 4> classes;
    for (std::uint64_t m = 0; m < w.m; m += d.single_core_m) {
        const auto cm = std::min(d.single_core_m, w.m - m);
        for (std::uint64_t n = 0; n < w.n; n += d.single_core_n) {
            const auto cn = std::min(d.single_core_n, w.n - n);
            const auto kind = 2 * (cm < d.single_core_m) +
                              (cn < d.single_core_n);
            classes[kind].emplace_back(cm, cn);
        }
    }
    std::uint64_t output_index = 0;
    for (const auto& tile_class : classes) {
        for (const auto& [cm, cn] : tile_class) {
            const auto owner = output_index++ % d.used_cores;
            elements[owner] += static_cast<double>(cm) * cn;
            outputs[owner] += 1.0;
            for (std::uint64_t bm = 0; bm < cm; bm += d.base_m) {
                const auto actual_m = std::min(d.base_m, cm - bm);
                for (std::uint64_t bn = 0; bn < cn; bn += d.base_n) {
                    const auto actual_n = std::min(d.base_n, cn - bn);
                    packets[owner] += 1.0;
                    a_bytes += 2.0 * static_cast<double>(w.k) * actual_m;
                    b_bytes += 2.0 * static_cast<double>(w.k) * actual_n;
                }
            }
        }
    }
    const auto k_packets = (w.k + d.base_k - 1) / d.base_k;
    double cube_critical = 0.0, packet_critical = 0.0;
    for (std::size_t core = 0; core < elements.size(); ++core) {
        cube_critical = std::max(cube_critical,
            2.0 * elements[core] * w.k / cube.rate_per_us);
        packet_critical = std::max(packet_critical,
            packets[core] * k_packets *
                (read_a.fixed_latency_us + read_b.fixed_latency_us) +
                outputs[core] * write.fixed_latency_us);
    }
    return h.primitives.at("launch").fixed_latency_us +
           std::max(cube_critical,
                    a_bytes / read_a.rate_per_us +
                    b_bytes / read_b.rate_per_us +
                    2.0 * w.m * w.n / write.rate_per_us + packet_critical);
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

    // Test-only exhaustive oracle: checks local choices against the
    // independently evaluated reduced objective. Production checks
    // feasible Split-K step/core pairs per geometry.
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

    // Source 910B3 K=65536 deterministic Split-K emits singleCoreM=384
    // for M=256. The oversize extent is a valid output tail and its
    // ping-pong partials plus converted A occupy exactly 67 MiB.
    const mmopt::Workload long_k{256, 256, 65536, "fp16"};
    auto long_hardware = TestHardware();
    long_hardware.workspace_limit_bytes = 512ULL * 1024 * 1024;
    mmopt::PlanDecision source_split;
    source_split.family = mmopt::KernelFamily::DeterministicSplitK;
    source_split.used_cores = 20;
    source_split.single_core_m = 384;
    source_split.single_core_n = 256;
    source_split.base_m = source_split.base_n = source_split.base_k = 128;
    source_split.step_k = 3;
    source_split.single_core_k = 384;
    const auto source_finalized = mmopt::FinalizePlan(
        long_k, long_hardware, source_split, "source_split");
    assert(source_finalized.workspace_bytes == 67ULL * 1024 * 1024);
    assert(source_finalized.nd2nz_a);
    assert(std::isfinite(mmopt::ReducedObjectiveUs(
        long_k, long_hardware, source_split)));
    const mmopt::MipTileDomain long_grid{{384}, {256}, {128}, {128}};
    const auto source_geometry = mmopt::SolveReducedMip(
        long_k, long_hardware, {}, long_grid);
    assert(!source_geometry.base);
    assert(source_geometry.deterministic_split_k);
    assert(source_geometry.deterministic_split_k->decision.single_core_m == 384);
    assert(!source_geometry.deterministic_split_k->event_graph_score);

    // With Cube work dominant, s=3 gives the busiest AIC 27 K packets,
    // while s=1 or 2 gives 26. The legal maximum step is not always best.
    auto cube_bound = long_hardware;
    cube_bound.primitives["gm_read"] = {"bytes", 0.0, 1.0e12};
    cube_bound.primitives["gm_write"] = {"bytes", 0.0, 1.0e12};
    cube_bound.primitives["cube"].rate_per_us = 1.0e6;
    cube_bound.primitives["vector_add"].rate_per_us = 1.0e12;
    const auto balanced_step = mmopt::SolveReducedMip(
        long_k, cube_bound, {}, long_grid);
    assert(balanced_step.deterministic_split_k);
    assert(balanced_step.deterministic_split_k->decision.used_cores == 20);
    assert(balanced_step.deterministic_split_k->decision.step_k < 3);
    assert(balanced_step.deterministic_split_k->surrogate_us <
           mmopt::ReducedObjectiveUs(long_k, cube_bound, source_split));

    // Independently expand the source's round/remainder K-slice assignment.
    // When Cube dominates, the reduced objective must use the busiest
    // AIC's actual K length, including a short final slice.
    auto slice_hardware = TestHardware();
    slice_hardware.primitives["launch"].fixed_latency_us = 0.0;
    slice_hardware.primitives["gm_read"] = {"bytes", 0.0, 1.0e12};
    slice_hardware.primitives["gm_write"] = {"bytes", 0.0, 1.0e12};
    slice_hardware.primitives["cube"].rate_per_us = 1.0e6;
    slice_hardware.primitives["vector_add"].rate_per_us = 1.0e12;
    slice_hardware.primitives["sync"].fixed_latency_us = 0.0;
    const mmopt::Workload sliced{256, 256, 1280, "fp16"};
    for (std::uint64_t step = 1; step <= 3; ++step) {
        const auto slice_k = step * 128;
        const auto slices = (sliced.k + slice_k - 1) / slice_k;
        for (std::uint64_t cores = 2; cores <= 4; ++cores) {
            if (cores > slices) continue;
            const auto round = (slices + cores - 1) / cores;
            auto front = slices % cores;
            if (!front) front = cores;
            std::uint64_t next = 0, busiest_k = 0;
            for (std::uint64_t core = 0; core < cores; ++core) {
                const auto count = core < front ? round : round - 1;
                std::uint64_t owned_k = 0;
                for (std::uint64_t i = 0; i < count; ++i, ++next) {
                    const auto offset = next * slice_k;
                    owned_k += std::min(slice_k, sliced.k - offset);
                }
                busiest_k = std::max(busiest_k, owned_k);
            }
            assert(next == slices);
            mmopt::PlanDecision d;
            d.family = mmopt::KernelFamily::DeterministicSplitK;
            d.used_cores = cores;
            d.single_core_m = d.base_m = 128;
            d.single_core_n = 256;
            d.base_n = d.base_k = 128;
            d.step_k = step;
            d.single_core_k = slice_k;
            const auto predicted = mmopt::ReducedObjectiveUs(
                sliced, slice_hardware, d);
            const auto explicit_cube =
                2.0 * sliced.m * sliced.n * busiest_k / 1.0e6;
            assert(std::abs(predicted - explicit_cube) < 0.001);
        }
    }

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

    // The deterministic kernel supplies two AIVs per occupied AIC and
    // assigns whole K slices to AICs. Check the solver's local choice
    // against all legal step/core pairs on a fixed source-like geometry.
    auto parallel_hardware = TestHardware();
    parallel_hardware.workspace_limit_bytes = 128ULL * 1024 * 1024;
    const mmopt::Workload parallel_shape{512, 512, 32768, "fp16"};
    const mmopt::MipTileDomain parallel_grid{{384}, {512}, {128}, {128}};
    // This historical regression fixture intentionally exercises the
    // nominal 512 KiB L1 boundary. The measured CANN 9.0.0 default is
    // 256 bytes smaller and rejects that exact boundary geometry.
    mmopt::CoreMemory nominal_memory;
    nominal_memory.l1_bytes = 524288;
    const auto parallel_result = mmopt::SolveReducedMip(
        parallel_shape, parallel_hardware, nominal_memory, parallel_grid);
    assert(parallel_result.deterministic_split_k);
    double parallel_oracle = std::numeric_limits<double>::infinity();
    for (std::uint64_t step = 1; step <= 4; ++step)
        for (std::uint64_t cores = 2; cores <= 20; ++cores) {
            mmopt::PlanDecision d;
            d.family = mmopt::KernelFamily::DeterministicSplitK;
            d.used_cores = cores;
            d.single_core_m = 384; d.single_core_n = 512;
            d.base_m = d.base_n = d.base_k = 128;
            d.step_k = step; d.single_core_k = 128 * step;
            try {
                parallel_oracle = std::min(parallel_oracle,
                    mmopt::ReducedObjectiveUs(parallel_shape,
                                              parallel_hardware, d,
                                              nominal_memory));
            } catch (const std::invalid_argument&) {}
        }
    assert(Equal(parallel_result.deterministic_split_k->surrogate_us,
                 parallel_oracle));

    // Removing Cube and read-startup pressure changes the core tradeoff;
    // the chosen plan must still beat a legal two-core alternative.
    auto reduction_heavy = parallel_hardware;
    reduction_heavy.primitives["cube"].rate_per_us = 1.0e12;
    reduction_heavy.primitives["gm_read"].fixed_latency_us = 0.0;
    const auto reduction_result = mmopt::SolveReducedMip(
        parallel_shape, reduction_heavy, nominal_memory, parallel_grid);
    assert(reduction_result.deterministic_split_k);
    auto two_cores = reduction_result.deterministic_split_k->decision;
    two_cores.used_cores = 2;
    assert(reduction_result.deterministic_split_k->surrogate_us <=
           mmopt::ReducedObjectiveUs(parallel_shape, reduction_heavy,
                                     two_cores, nominal_memory));

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
    assert(multicore_base.single_core_n >= multicore_base.base_n);
    auto one_core = multicore_base;
    one_core.used_cores = 1;
    assert(multicore_result.base->surrogate_us <
           mmopt::ReducedObjectiveUs(multicore_shape, compute_heavy_hardware,
                                     one_core));

    // Packet start-up is paid by the busiest AIC, not serially across all
    // output tiles. Check the four-class closed form against explicit tile
    // expansion with asymmetric full/tail tiles and several core counts.
    const mmopt::Workload tail_shape{368, 64, 256, "fp16"};
    mmopt::PlanDecision tail_tile;
    tail_tile.single_core_m = 32;
    tail_tile.single_core_n = 48;
    tail_tile.single_core_k = tail_shape.k;
    tail_tile.base_m = tail_tile.base_n = 16;
    tail_tile.base_k = 128;
    for (std::uint64_t cores = 1; cores <= 20; ++cores) {
        tail_tile.used_cores = cores;
        assert(Equal(NaiveBaseObjectiveUs(tail_shape, hardware, tail_tile),
                     mmopt::ReducedObjectiveUs(tail_shape, hardware,
                                               tail_tile)));
    }

    // This device-derived, exploratory profile previously chose two AICs
    // for a large shape because it serially charged every packet. The
    // per-AIC load model should expose the available 20-way parallelism.
    auto measured_profile = TestHardware();
    measured_profile.primitives["launch"].fixed_latency_us = 2.68;
    measured_profile.primitives["gm_read"] =
        {"bytes", 0.109620, 3710212.0};
    measured_profile.primitives["gm_write"] =
        {"bytes", 0.162586, 5229502.0};
    measured_profile.primitives["cube"].rate_per_us = 14719432.0;
    const mmopt::Workload high_occupancy{2048, 2048, 4096, "fp16"};
    const auto occupied = mmopt::SolveReducedMip(high_occupancy,
                                                 measured_profile);
    assert(occupied.base);
    assert(occupied.base->decision.used_cores == 20);

    // Plain source BASE key 0x10000 writes one base M/N tile per Iterate.
    // Its default 128x256 accumulator uses one L0C buffer, not two.
    mmopt::CoreMemory source_memory;
    source_memory.base_l0c_buffers = 1;
    mmopt::MipTileDomain source_domain;
    source_domain.base_core_equals_base = true;
    const auto source_compatible = mmopt::SolveReducedMip(
        high_occupancy, measured_profile, source_memory, source_domain);
    assert(source_compatible.base);
    const auto& source_tile = source_compatible.base->decision;
    assert(source_tile.single_core_m == source_tile.base_m);
    assert(source_tile.single_core_n == source_tile.base_n);
    mmopt::PlanDecision emitted_default;
    emitted_default.used_cores = 20;
    emitted_default.single_core_m = emitted_default.base_m = 128;
    emitted_default.single_core_n = emitted_default.base_n = 256;
    emitted_default.single_core_k = high_occupancy.k;
    emitted_default.base_k = 64;
    const auto default_estimate = mmopt::ReducedObjectiveUs(
        high_occupancy, measured_profile, emitted_default, source_memory);
    assert(std::isfinite(default_estimate));
    auto swapped_tile = emitted_default;
    swapped_tile.single_core_m = swapped_tile.base_m = 256;
    swapped_tile.single_core_n = swapped_tile.base_n = 128;
    assert(Equal(default_estimate, mmopt::ReducedObjectiveUs(
        high_occupancy, measured_profile, swapped_tile, source_memory)));

    // Source L1-step packet primitive fit: A rows are M, B rows are K.
    // The independently profiled 128-KiB packets give 8/4 versus 4/8
    // A/B baseK steps and break the legacy orientation tie.
    auto packet_profile = measured_profile;
    packet_profile.primitives["base_mte2_a"] =
        {"bytes", 0.19097384635481152, 1024.0 / 0.003532113558081016};
    packet_profile.primitives["base_mte2_b"] =
        {"bytes", 0.07348386212366087, 1024.0 / 0.003017312080586143};
    packet_profile.primitives["base_mte2_a_rows"] =
        {"rows", 0.0, 1.0 / 0.000854946401991721};
    packet_profile.primitives["base_mte2_b_rows"] =
        {"rows", 0.0, 1.0 / 0.001272028070525036};
    const auto packet_default = mmopt::ReducedObjectiveUs(
        high_occupancy, packet_profile, emitted_default, source_memory);
    const auto packet_swapped = mmopt::ReducedObjectiveUs(
        high_occupancy, packet_profile, swapped_tile, source_memory);
    assert(packet_default < packet_swapped);
    assert(std::abs(packet_default - 135.521144) < 0.001);
    assert(std::abs(packet_swapped - 164.175447) < 0.001);
    const auto packet_solve = mmopt::SolveReducedMip(
        high_occupancy, packet_profile, source_memory,
        {{128, 256}, {128, 256}, {128, 256}, {128, 256}, true});
    assert(packet_solve.base);
    assert(packet_solve.base->decision.base_m == 128);
    assert(packet_solve.base->decision.base_n == 256);
    assert(Equal(packet_solve.base->surrogate_us, packet_default));
    bool rejected_incomplete_packet_profile = false;
    packet_profile.primitives.erase("base_mte2_b_rows");
    try {
        (void)mmopt::ReducedObjectiveUs(high_occupancy, packet_profile,
                                        emitted_default, source_memory);
    } catch (const std::invalid_argument&) {
        rejected_incomplete_packet_profile = true;
    }
    assert(rejected_incomplete_packet_profile);

    // Equal directional services reproduce the legacy objective; changing
    // only an independently supplied A/B transfer service breaks the
    // orientation tie without fitting a whole-kernel correction.
    auto directional = measured_profile;
    directional.primitives["gm_read_a"] = directional.primitives.at("gm_read");
    directional.primitives["gm_read_b"] = directional.primitives.at("gm_read");
    assert(Equal(default_estimate, mmopt::ReducedObjectiveUs(
        high_occupancy, directional, emitted_default, source_memory)));
    directional.primitives["gm_read_a"].rate_per_us = 1000000.0;
    directional.primitives["gm_read_b"].rate_per_us = 4000000.0;
    const auto a_slow_default = mmopt::ReducedObjectiveUs(
        high_occupancy, directional, emitted_default, source_memory);
    const auto a_slow_swapped = mmopt::ReducedObjectiveUs(
        high_occupancy, directional, swapped_tile, source_memory);
    assert(a_slow_default < a_slow_swapped);
    assert(Equal(a_slow_default, NaiveBaseObjectiveUs(
        high_occupancy, directional, emitted_default)));
    assert(Equal(a_slow_swapped, NaiveBaseObjectiveUs(
        high_occupancy, directional, swapped_tile)));
    const mmopt::MipTileDomain orientation_grid{
        {128, 256}, {128, 256}, {128, 256}, {128, 256}, true};
    const auto a_slow_solve = mmopt::SolveReducedMip(
        high_occupancy, directional, source_memory, orientation_grid);
    assert(a_slow_solve.base);
    assert(a_slow_solve.base->decision.base_m == 128);
    assert(a_slow_solve.base->decision.base_n == 256);
    directional.primitives["gm_read_a"].rate_per_us = 4000000.0;
    directional.primitives["gm_read_b"].rate_per_us = 1000000.0;
    assert(mmopt::ReducedObjectiveUs(high_occupancy, directional, swapped_tile,
                                      source_memory) <
           mmopt::ReducedObjectiveUs(high_occupancy, directional, emitted_default,
                                      source_memory));
    const auto b_slow_solve = mmopt::SolveReducedMip(
        high_occupancy, directional, source_memory, orientation_grid);
    assert(b_slow_solve.base);
    assert(b_slow_solve.base->decision.base_m == 256);
    assert(b_slow_solve.base->decision.base_n == 128);
    directional.primitives.erase("gm_read_b");
    bool rejected_incomplete_directional_profile = false;
    try {
        (void)mmopt::ReducedObjectiveUs(high_occupancy, directional,
                                        emitted_default, source_memory);
    } catch (const std::invalid_argument&) {
        rejected_incomplete_directional_profile = true;
    }
    assert(rejected_incomplete_directional_profile);
    bool rejected_double_l0c = false;
    try {
        (void)mmopt::ReducedObjectiveUs(high_occupancy, measured_profile,
                                        emitted_default);
    } catch (const std::invalid_argument&) {
        rejected_double_l0c = true;
    }
    assert(rejected_double_l0c);

    // The selected plan remains available if explicit events exceed the cap.
    const mmopt::Workload large{102400, 30720, 4096, "fp16"};
    const mmopt::MipTileDomain large_grid{{128}, {128}, {64}, {64}};
    const auto large_result = mmopt::SolveReducedMip(large, hardware, {}, large_grid);
    assert(large_result.winner);
    assert(large_result.winner->decision.single_core_m == 128);
    assert(large_result.winner->decision.base_m == 64);
    assert(!large_result.winner->event_graph_score);

    // Large Split-K may reuse one output tile's partial workspace. Its
    // explicit event graph assumes whole-output partials and cannot be
    // expanded, but that reporting limitation must not abort BASE solving.
    const mmopt::Workload streamed_partials{2048, 2048, 4096, "fp16"};
    const mmopt::MipTileDomain streamed_grid{{128}, {128}, {128}, {128}};
    const auto streamed_result = mmopt::SolveReducedMip(
        streamed_partials, hardware, {}, streamed_grid);
    assert(streamed_result.base && streamed_result.deterministic_split_k);
    assert(!streamed_result.deterministic_split_k->event_graph_score);

    // A slower Cube makes multiple K owners valuable in the reduced model.
    hardware.primitives["cube"].rate_per_us = 100000.0;
    const auto compute_bound = mmopt::SolveReducedMip(workload, hardware);
    assert(compute_bound.winner &&
           compute_bound.winner->decision.family ==
               mmopt::KernelFamily::DeterministicSplitK);

    std::cout << "mmopt_mip_tests passed\n";
}
