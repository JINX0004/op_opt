#include "event_builder.h"
#include "optimizer.h"
#include "plan_finalizer.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

mmopt::Candidate Plan(const mmopt::Workload& workload, const mmopt::Hardware& hardware,
                      const std::string& family,
                      const mmopt::TileAssignment& choices) {
    std::string id = family;
    for (const auto& [name, value] : choices)
        id += "_" + name + std::to_string(value);
    mmopt::PlanDecision decision;
    decision.family = family == "BASE" ? mmopt::KernelFamily::Base
                                        : mmopt::KernelFamily::DeterministicSplitK;
    decision.used_cores = 1;
    decision.single_core_m = workload.m;
    decision.single_core_n = workload.n;
    decision.single_core_k = workload.k;
    decision.base_m = 16;
    decision.base_n = 16;
    decision.base_k = 128;
    if (const auto it = choices.find("baseK"); it != choices.end())
        decision.base_k = it->second;
    if (family == "DETERMINISTIC_SPLIT_K") {
        decision.used_cores = choices.at("usedCoreNum");
        decision.single_core_k = choices.at("singleCoreK");
        decision.step_k = decision.single_core_k / decision.base_k;
    }
    const auto plan = mmopt::FinalizePlan(workload, hardware, decision, id);
    return mmopt::BuildMatmulEvents(workload, plan.candidate);
}

mmopt::Implementation Family(const mmopt::Workload& workload, const mmopt::Hardware& hardware,
                             const std::string& name,
                             std::vector<mmopt::TileVariable> variables) {
    mmopt::Implementation implementation;
    implementation.id = name;
    implementation.variables = std::move(variables);
    implementation.build = [workload, hardware, name](const mmopt::TileAssignment& choices)
        -> std::optional<mmopt::Candidate> {
        try {
            return Plan(workload, hardware, name, choices);
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }
    };
    return implementation;
}

}  // namespace

int main() {
    const mmopt::Workload workload{16, 16, 102400, "fp16"};
    // Illustration only: these are not measured 910B3 primitive parameters.
    const mmopt::Hardware hardware{
        20, 32 * 1048576,
        {{"launch", {"count", 0.2, 1.0}},
         {"gm_read", {"bytes", 0.0, 1600000.0}},
         {"gm_write", {"bytes", 0.0, 1600000.0}},
         {"cube", {"flop", 0.0, 14745600.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}}}};
    const std::vector<mmopt::Implementation> families = {
        Family(workload, hardware, "BASE", {{"baseK", {128, 256}}}),
        Family(workload, hardware, "DETERMINISTIC_SPLIT_K",
               {{"usedCoreNum", {10, 20}}, {"singleCoreK", {256, 512}}}),
    };
    const auto result = mmopt::OptimizeImplementations(workload, hardware, families);
    for (const auto& family : result.implementations) {
        if (!family.best) continue;
        std::cout << "implementation=" << family.id
                  << " predicted_us=" << family.best->score.runtime_us
                  << " scored=" << family.scored
                  << " rejected=" << family.rejected
                  << " pruned=" << family.pruned_subtrees << '\n';
    }
    if (!result.best) return 3;
    std::cout << "winner=" << result.best->family
              << " predicted_us=" << result.best->score.runtime_us << '\n';
    std::cout << "selected_plan key=" << result.best_plan->tiling_key
              << " workspace_bytes=" << result.best_plan->workspace_bytes
              << " event_count=" << result.best_plan->events.size();
    for (const auto& [name, value] : result.best_plan->tile)
        std::cout << ' ' << name << '=' << value;
    std::cout << '\n';
}
