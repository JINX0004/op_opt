#include "mip_solver.h"

#include <iomanip>
#include <iostream>

namespace {

void Print(const char* name, const std::optional<mmopt::MipPlan>& plan) {
    if (!plan) {
        std::cout << name << " infeasible\n";
        return;
    }
    const auto& d = plan->decision;
    std::cout << name << " surrogate_us=" << plan->surrogate_us
              << " event_graph_us=";
    if (plan->event_graph_score)
        std::cout << plan->event_graph_score->runtime_us;
    else std::cout << "unavailable(event_cap)";
    std::cout
              << " singleCoreM=" << d.single_core_m
              << " singleCoreN=" << d.single_core_n
              << " baseM=" << d.base_m
              << " baseN=" << d.base_n
              << " usedCoreNum=" << d.used_cores
              << " singleCoreK=" << d.single_core_k
              << " baseK=" << d.base_k
              << " stepK=" << d.step_k
              << " key=" << plan->candidate.tiling_key
              << " workspace_bytes=" << plan->candidate.workspace_bytes << '\n';
}

}  // namespace

int main() {
    const mmopt::Workload workload{16, 16, 102400, "fp16"};
    // Hypothetical rates used to demonstrate the formulation, not a measured
    // 910B3 timing profile.
    const mmopt::Hardware hardware{
        20, 32ULL * 1024 * 1024,
        {{"launch", {"count", 0.2, 1.0}},
         {"gm_read", {"bytes", 0.005, 1600000.0}},
         {"gm_write", {"bytes", 0.005, 1600000.0}},
         {"cube", {"flop", 0.0, 14745600.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}}}};
    const auto result = mmopt::SolveReducedMip(workload, hardware);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Selection: optimal reduced MILP objective within the declared tile grid\n";
    Print("BASE", result.base);
    Print("DETERMINISTIC_SPLIT_K", result.deterministic_split_k);
    Print("WINNER", result.winner);
    return result.winner ? 0 : 3;
}
