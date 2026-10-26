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
// deterministic Split-K checks bounded K-step and occupied-core choices for
// each spatial geometry before the one-hot geometry master is solved.
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
    std::optional<MipPlan> winner;
};

// Requires the optional HiGHS C library at build time. Each family is solved
// as an integer program; the smaller certified surrogate objective wins.
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
