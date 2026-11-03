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
};

// A reduced geometry-selection MILP. BASE K tiling is eliminated analytically;
// deterministic Split-K checks bounded K-step/core choices, and FP16/BF16
// single-core Split-K checks source block patterns/core choices, for each
// spatial geometry before the one-hot geometry master is solved.
// It minimizes a service-demand surrogate, not the full event-graph makespan.
struct MipPlan {
    PlanDecision decision;
    Candidate candidate;
    double surrogate_us = 0.0;
    // Absent when the selected plan exceeds the explicit event-graph cap or
    // its whole-output Split-K partial representation exceeds workspace.
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

// Requires the optional HiGHS C library at build time. Each family is solved
// as an integer program; the smallest certified reduced objective wins.
// Optional single-core primitives: sc_fp32_store, sc_fp32_store_steady,
// sc_atomic_extra (bytes), sc_cast (elements), and paired sc_gm_to_l1_a/b
// (bytes). For the measured FP16 mode-5 path, supply both
// sc_mode5_fixpipe (total FP32 output bytes) and sc_mode5_iterate
// (one source IterateAll call); other dtypes retain the original mode-5
// objective. sc_fp32_store_steady is the total FP32 atomic-output demand
// for mode-6 K packets after the first; it supersedes sc_atomic_extra on
// those packets. Without it generic gm_write plus sc_atomic_extra is an
// exploratory proxy. They are caller supplied; no device measurements
// are installed as defaults.
// Throws on malformed hardware data or a solver failure. Infeasible families
// are absent from the result.
MipResult SolveReducedMip(const Workload& workload, const Hardware& hardware,
                          const CoreMemory& memory = {},
                          const MipTileDomain& domain = {});

// Recompute the reduced objective for a complete plan. This is also useful
// for checking the MILP's integer solution and documenting its exact scope.
double ReducedObjectiveUs(const Workload& workload, const Hardware& hardware,
                          const PlanDecision& decision, const CoreMemory& memory = {});

}  // namespace mmopt
