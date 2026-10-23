#pragma once

#include "optimizer.h"
#include "plan_finalizer.h"

#include <optional>
#include <vector>

namespace mmopt {

// Discrete block domains. Empty vectors use the documented default grid;
// nonempty vectors specify exact admissible aligned tile extents for one
// dimension. The solver selects all four dimensions independently subject
// to baseM<=singleCoreM and baseN<=singleCoreN.
struct MipTileDomain {
    std::vector<std::uint64_t> single_core_m;
    std::vector<std::uint64_t> single_core_n;
    std::vector<std::uint64_t> base_m;
    std::vector<std::uint64_t> base_n;
};

// A reduced MILP with block geometry, K tile/step, and AIC count decisions.
// It minimizes a service-demand surrogate, not the full event-graph makespan.
struct MipPlan {
    PlanDecision decision;
    Candidate candidate;
    double surrogate_us = 0.0;
    // Absent when the selected plan exceeds the explicit event-graph cap.
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
