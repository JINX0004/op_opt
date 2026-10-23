#include "event_builder.h"
#include "optimizer.h"
#include "plan_finalizer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using mmopt::Candidate;
using mmopt::Event;

Candidate MakeCandidate(std::string id, std::vector<Event> events) {
    Candidate candidate;
    candidate.id = std::move(id);
    candidate.family = "BASE";
    candidate.tiling_key = "test-key";
    candidate.backend_verified = true;
    candidate.tile = {{"usedCoreNum", 2}, {"singleCoreM", 16}, {"singleCoreN", 16},
                      {"singleCoreK", 32}, {"baseM", 16}, {"baseN", 16}, {"baseK", 32}};
    candidate.events = std::move(events);
    return candidate;
}

int main() {
    const mmopt::Workload workload{16, 16, 64, "fp16"};
    const mmopt::Hardware hardware{
        2, 1024,
        {{"load", {"bytes", 0.0, 1.0}}, {"cube", {"flop", 0.0, 1.0}},
         {"store", {"bytes", 0.0, 1.0}}}};

    // On one resource, the second transfer cannot start until the first
    // finishes. Cube work on another resource overlaps the second transfer.
    const auto pipeline = MakeCandidate("pipeline", {
        {"load0", "mte2", "load", 4.0, "bytes", {}},
        {"cube0", "cube", "cube", 3.0, "flop", {"load0"}},
        {"load1", "mte2", "load", 4.0, "bytes", {}},
        {"cube1", "cube", "cube", 3.0, "flop", {"load1", "cube0"}},
        {"store", "mte3", "store", 1.0, "bytes", {"cube1"}},
    });
    const auto serial = MakeCandidate("serial", {
        {"load", "mte2", "load", 5.0, "bytes", {}},
        {"cube", "cube", "cube", 6.0, "flop", {"load"}},
        {"store", "mte3", "store", 2.0, "bytes", {"cube"}},
    });
    auto result = mmopt::Optimize(workload, hardware, {serial, pipeline});
    assert(result.has_winner() && result.winner().id == "pipeline");
    assert(std::abs(result.winner().score.runtime_us - 12.0) < 1e-12);
    assert(result.winner().score.critical_path ==
           std::vector<std::string>({"load0", "load1", "cube1", "store"}));

    // Invalid schedules and resource/workspace violations are rejected.
    auto cycle = MakeCandidate("cycle", {{"a", "r0", "load", 1.0, "bytes", {"b"}},
                                          {"b", "r1", "cube", 1.0, "flop", {"a"}}});
    auto over_budget = pipeline;
    over_budget.id = "over-budget";
    over_budget.workspace_bytes = 1025;
    auto unverified = pipeline;
    unverified.id = "unverified";
    unverified.backend_verified = false;
    auto missing_dependency = MakeCandidate("missing-dependency",
                                            {{"a", "r0", "load", 1.0, "bytes", {"absent"}}});
    auto wrong_unit = MakeCandidate("wrong-unit",
                                    {{"a", "r0", "load", 1.0, "flop", {}}});
    result = mmopt::Optimize(workload, hardware,
                             {cycle, over_budget, unverified, missing_dependency, wrong_unit});
    assert(!result.has_winner() && result.rejected.size() == 5);

    // A shared profile change affects every candidate that uses the primitive.
    auto faster_hardware = hardware;
    faster_hardware.primitives["load"].rate_per_us = 2.0;
    result = mmopt::Optimize(workload, faster_hardware, {pipeline});
    assert(std::abs(result.winner().score.runtime_us - 9.0) < 1e-12);

    // A tie retains source order; the optimizer never claims a global
    // optimum outside the submitted finite candidate set.
    auto twin = pipeline;
    twin.id = "twin";
    result = mmopt::Optimize(workload, hardware, {twin, pipeline});
    assert(result.winner().id == "twin");

    // Two-level solve: each implementation gets an independent finite tiling
    // domain. The lower bound prunes only plans that cannot beat its own
    // implementation's incumbent.
    auto make_family = [&](const std::string& name, std::vector<std::uint64_t> values,
                           auto cost, bool bounded) {
        mmopt::Implementation implementation;
        implementation.id = name;
        implementation.variables = {{"singleCoreK", std::move(values)}};
        implementation.build = [name, cost](const mmopt::TileAssignment& tile)
            -> std::optional<Candidate> {
            const auto k = tile.at("singleCoreK");
            auto candidate = MakeCandidate(name + std::to_string(k),
                                           {{"compute", "cube", "cube",
                                             cost(k), "flop", {}}});
            candidate.family = name;
            candidate.tile["singleCoreK"] = k;
            return candidate;
        };
        if (bounded) {
            implementation.lower_bound_us = [cost](const mmopt::TileAssignment& tile) {
                const auto found = tile.find("singleCoreK");
                return found == tile.end() ? 0.0 : cost(found->second);
            };
        }
        return implementation;
    };
    const auto base_cost = [](std::uint64_t k) { return 9.0 - static_cast<double>(k); };
    const auto split_cost = [](std::uint64_t k) { return 4.0 - static_cast<double>(k); };
    const auto base = make_family("BASE", {4, 2, 1}, base_cost, true);
    const auto split = make_family("DETERMINISTIC_SPLIT_K", {1, 2}, split_cost, false);
    const auto nested = mmopt::OptimizeImplementations(workload, hardware, {base, split});
    assert(nested.best && nested.best->family == "DETERMINISTIC_SPLIT_K");
    assert(nested.best_plan && nested.best_plan->tile.at("singleCoreK") == 2);
    assert(nested.best_tiling.at("singleCoreK") == 2);
    assert(std::abs(nested.best->score.runtime_us - 2.0) < 1e-12);
    assert(nested.implementations[0].best_tiling.at("singleCoreK") == 4);
    assert(nested.implementations[0].pruned_subtrees == 2);
    assert(nested.implementations[1].scored == 2);

    auto base_without_bound = base;
    base_without_bound.lower_bound_us = {};
    const auto exhaustive = mmopt::OptimizeImplementations(workload, hardware,
                                                             {base_without_bound, split});
    assert(exhaustive.best->id == nested.best->id);
    assert(exhaustive.implementations[0].scored == 3);

    // Shape-derived BASE work includes four output tails and three K tails.
    auto base_plan = MakeCandidate("generated-base", {});
    base_plan.tile["singleCoreK"] = 33;
    base_plan.tile["baseK"] = 16;
    const auto generated_base = mmopt::BuildMatmulEvents({17, 18, 33, "fp16"}, base_plan);
    double base_read = 0.0, base_write = 0.0, base_flops = 0.0;
    for (const auto& event : generated_base.events) {
        if (event.primitive == "gm_read") base_read += event.work_units;
        if (event.primitive == "gm_write") base_write += event.work_units;
        if (event.primitive == "cube") base_flops += event.work_units;
    }
    assert(generated_base.events.size() == 41);
    assert(base_read == 4620.0 && base_write == 612.0);
    assert(base_flops == 98304.0);
    auto wider_base = base_plan;
    wider_base.tile["baseM"] = 8;
    const auto generated_wider = mmopt::BuildMatmulEvents({17, 18, 33, "fp16"}, wider_base);
    assert(generated_wider.events.size() > generated_base.events.size());
    auto longer_plan = base_plan;
    longer_plan.tile["singleCoreK"] = 49;
    const auto generated_longer_k = mmopt::BuildMatmulEvents({17, 18, 49, "fp16"},
                                                              longer_plan);
    double longer_read = 0.0;
    for (const auto& event : generated_longer_k.events)
        if (event.primitive == "gm_read") longer_read += event.work_units;
    assert(longer_read > base_read);

    // Deterministic Split-K creates one FP32 partial per core, reads both
    // partials for the merge, and writes the FP16 output.
    auto split_plan = MakeCandidate("generated-split", {});
    split_plan.family = "DETERMINISTIC_SPLIT_K";
    split_plan.tile["singleCoreK"] = 16;
    split_plan.tile["baseK"] = 16;
    split_plan.workspace_bytes = 2048;
    const auto generated_split = mmopt::BuildMatmulEvents({16, 16, 32, "fp16"}, split_plan);
    double split_read = 0.0, split_write = 0.0, split_add = 0.0;
    for (const auto& event : generated_split.events) {
        if (event.primitive == "gm_read") split_read += event.work_units;
        if (event.primitive == "gm_write") split_write += event.work_units;
        if (event.primitive == "vector_add") split_add += event.work_units;
    }
    assert(generated_split.events.size() == 13);
    assert(split_read == 4096.0 && split_write == 2560.0 && split_add == 256.0);
    auto transposed = mmopt::Workload{16, 16, 32, "fp16"};
    transposed.trans_a = true;
    bool unsupported_layout = false;
    try {
        mmopt::BuildMatmulEvents(transposed, split_plan);
    } catch (const std::invalid_argument&) {
        unsupported_layout = true;
    }
    assert(unsupported_layout);
    auto unmodeled = split_plan;
    unmodeled.tile["depthA1"] = 2;
    bool unsupported_tile = false;
    try {
        mmopt::BuildMatmulEvents({16, 16, 32, "fp16"}, unmodeled);
    } catch (const std::invalid_argument&) {
        unsupported_tile = true;
    }
    assert(unsupported_tile);
    split_plan.workspace_bytes = 2047;
    bool insufficient_workspace = false;
    try {
        mmopt::BuildMatmulEvents({16, 16, 32, "fp16"}, split_plan);
    } catch (const std::invalid_argument&) {
        insufficient_workspace = true;
    }
    assert(insufficient_workspace);

    // Source-backed plan preflight derives the key tuple, launch dimensions,
    // and the host tiler's RPC plus deterministic partial workspace formula.
    const mmopt::Workload large_k{16, 16, 102400, "fp16"};
    mmopt::Hardware b3;
    b3.aic_count = 20;
    b3.workspace_limit_bytes = 32ULL * 1024 * 1024;
    mmopt::PlanDecision decision;
    decision.used_cores = 1;
    decision.single_core_m = 16;
    decision.single_core_n = 16;
    decision.single_core_k = large_k.k;
    decision.base_m = 16;
    decision.base_n = 16;
    decision.base_k = 128;
    auto finalized = mmopt::FinalizePlan(large_k, b3, decision, "base");
    assert(finalized.block_dim == 1 && finalized.schedule_mode == 1);
    assert(finalized.key.split_core_mode == 0 && finalized.key.mix_nd2nz == 0);
    assert(finalized.nd2nz_a && !finalized.nd2nz_b);
    assert(finalized.nd2nz_a_workspace_bytes == 16 * 102400 * 2);
    assert(finalized.workspace_bytes == 20ULL * 1024 * 1024 + 16 * 102400 * 2);
    assert(finalized.candidate.backend_verified);
    assert(finalized.candidate.tiling_key == "(0,0,0,0,0,0)");
    assert(finalized.candidate.tile.size() == 7);
    auto converted_base = mmopt::BuildMatmulEvents(large_k, finalized.candidate);
    assert(converted_base.events[1].primitive == "gm_read");
    assert(converted_base.events[2].primitive == "nd2nz");
    assert(converted_base.events[3].primitive == "gm_write");
    assert(converted_base.events[4].primitive == "sync");
    assert(converted_base.events[1].work_units == 16.0 * 102400 * 2);
    assert(converted_base.events[2].work_units == 16.0 * 102400);

    decision.family = mmopt::KernelFamily::DeterministicSplitK;
    decision.used_cores = 20;
    decision.single_core_k = 256;
    decision.step_k = 2;
    finalized = mmopt::FinalizePlan(large_k, b3, decision, "det");
    assert(finalized.key.split_core_mode == 3);
    assert(finalized.candidate.tiling_key == "(0,3,0,0,0,0)");
    assert(finalized.workspace_bytes == 20ULL * 1024 * 1024 +
                                         20 * 16 * 16 * 8 + 16 * 102400 * 2);
    assert(finalized.block_dim == 20);
    auto converted_split = mmopt::BuildMatmulEvents(large_k, finalized.candidate);
    assert(converted_split.events[2].primitive == "nd2nz");

    auto rejected_plan = [&](const auto& shape, const auto& platform,
                             const auto& tile) {
        try {
            (void)mmopt::FinalizePlan(shape, platform, tile, "bad");
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };
    auto wrong_step = decision;
    wrong_step.step_k = 3;
    assert(rejected_plan(large_k, b3, wrong_step));
    auto too_many_cores = decision;
    too_many_cores.used_cores = 21;
    assert(rejected_plan(large_k, b3, too_many_cores));
    auto huge_base = decision;
    huge_base.base_m = 256;
    assert(rejected_plan(large_k, b3, huge_base));
    auto small_budget = b3;
    small_budget.workspace_limit_bytes = 20ULL * 1024 * 1024;
    assert(rejected_plan(large_k, small_budget, decision));
    auto bad_dtype = large_k;
    bad_dtype.dtype = "fp32";
    assert(rejected_plan(bad_dtype, b3, decision));
    auto tail = large_k;
    tail.n = 17;
    assert(rejected_plan(tail, b3, decision));
    auto no_conversion = mmopt::Workload{16, 128, 32768, "fp16"};
    auto no_conversion_decision = decision;
    no_conversion_decision.single_core_n = 128;
    const auto direct = mmopt::FinalizePlan(no_conversion, b3,
                                            no_conversion_decision, "direct");
    assert(!direct.nd2nz_a && direct.key.mix_nd2nz == 1);
    auto trans = large_k;
    trans.trans_a = true;
    assert(rejected_plan(trans, b3, decision));
    auto overflow = decision;
    overflow.base_m = std::numeric_limits<std::uint64_t>::max() - 15;
    assert(rejected_plan(large_k, b3, overflow));

    std::cout << "mmopt_tests passed\n";
}
