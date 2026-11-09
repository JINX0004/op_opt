#pragma once

#include "optimizer.h"
#include "plan_finalizer.h"

#include <optional>
#include <vector>

namespace mmopt {

// Discrete block domains. Empty base vectors use a small L0 tile grid; empty
// single-core vectors additionally include full-axis and core-partition
// extents derived from the shape and allocated AIC count. Nonempty vectors
// override that dimension's default domain. All values must be 16-aligned.
// The solver selects all four dimensions independently subject
// to baseM<=singleCoreM and baseN<=singleCoreN.
struct MipTileDomain {
    std::vector<std::uint64_t> single_core_m;
    std::vector<std::uint64_t> single_core_n;
    std::vector<std::uint64_t> base_m;
    std::vector<std::uint64_t> base_n;
    // For the tested plain BASE key 0x10000, the source kernel's single
    // Iterate call writes only one base tile. Enable this in device studies
    // to exclude outer tiles larger than baseM/N; other families are free.
    bool base_core_equals_base = false;
    // Use bounded, source-informed FP32 BASE, deterministic Split-K and
    // single-core mode-2/mode-5 geometry domains. The FP32 mode-5
    // K=1536 NN channel keeps N=384 as one output tile because its
    // narrower N partitions need a repeated-A transfer model.
    // BASE candidates are capacity
    // admissible, not a prediction of the source tiler's one default tile.
    // Ignored for FP16/BF16; false retains the unrestricted standalone grid.
    bool fp32_source_patterns = false;
};

// A reduced geometry-selection search. BASE K tiling is eliminated analytically;
// deterministic Split-K checks bounded K-step/core choices, and supported
// single-core Split-K modes check source block patterns/core choices. The best
// feasible choice per family and then across families is selected directly.
// It minimizes a service-demand surrogate, not the full event-graph makespan.
struct MipPlan {
    PlanDecision decision;
    Candidate candidate;
    double surrogate_us = 0.0;
    // The online selector does not build a detailed event graph.
    std::optional<Score> event_graph_score;
};

struct MipResult {
    std::optional<MipPlan> base;
    std::optional<MipPlan> deterministic_split_k;
    std::optional<MipPlan> single_core_split_k_mode2;
    std::optional<MipPlan> single_core_split_k_mode5;
    std::optional<MipPlan> single_core_split_k_mode6;
    std::optional<MipPlan> winner;
};

// No external solver is required. Each family enumerates its bounded discrete
// choices and the smallest reduced objective wins.
// Optional single-core primitives: sc_fp32_store, sc_fp32_store_steady,
// sc_atomic_extra (bytes), sc_cast (elements), and paired sc_gm_to_l1_a/b
// (bytes). sc_fp32_store_steady is the total FP32 atomic-output demand
// for mode-6 packets and direct FP32 mode-2/mode-5 packets after the first; it
// supersedes sc_atomic_extra there. Without it generic gm_write plus
// sc_atomic_extra is an
// exploratory proxy. They are caller supplied; no device measurements
// are installed as defaults.
// Optional FP32 BASE primitives: base_fp32_packet_issue (count),
// base_fp32_call (count), and base_fp32_atomic_output (bytes). The packet
// term is a provisional in-context slope from a fixed source tile; the
// call and atomic terms require independent calibration before activation.
// Optional experimental FP32 mode-2 pipeline primitives: sc_mode2_l0a/b
// (bytes). Supplying both replaces only that family's per-slice timing
// expression; the default online profile omits them.
// Throws on malformed hardware data or an invalid candidate. Infeasible families
// are absent from the result.
MipResult SolveReducedMip(const Workload& workload, const Hardware& hardware,
                          const CoreMemory& memory = {},
                          const MipTileDomain& domain = {});

// Recompute the reduced objective for a complete plan. This is also useful
// for checking the MILP's integer solution and documenting its exact scope.
double ReducedObjectiveUs(const Workload& workload, const Hardware& hardware,
                          const PlanDecision& decision, const CoreMemory& memory = {},
                          bool source_emitted_base = false);

}  // namespace mmopt
