#include "hardware_profile.h"

#include <iomanip>
#include <iostream>
#include <optional>

int main() {
    const mmopt::Workload w{2560, 1024, 73728, "fp16"};
    const auto h = mmopt::NarrowB910B3();
    const auto d = mmopt::SourceCompatibleDomain();
    const auto r = mmopt::SolveReducedMip(w, h, {}, d);
    std::cout << std::fixed << std::setprecision(6);
    const auto show = [](const char* name, const std::optional<mmopt::MipPlan>& plan) {
        if (!plan) {
            std::cout << name << ": infeasible\n";
            return;
        }
        const auto& p = *plan;
        const auto& x = p.decision;
        std::cout << name << ": objective_us=" << p.surrogate_us
                  << " used_cores=" << x.used_cores
                  << " core=" << x.single_core_m << 'x' << x.single_core_n
                  << 'x' << x.single_core_k
                  << " base=" << x.base_m << 'x' << x.base_n
                  << 'x' << x.base_k
                  << " steps=" << x.step_m << 'x' << x.step_n
                  << 'x' << x.step_k << '\n';
    };
    show("BASE", r.base);
    show("deterministic_split_k", r.deterministic_split_k);
    show("single_core_mode2", r.single_core_split_k_mode2);
    show("single_core_mode5", r.single_core_split_k_mode5);
    show("single_core_mode6", r.single_core_split_k_mode6);
    const auto source_score = [&](const char* name, mmopt::PlanDecision decision,
                                  mmopt::Hardware hardware = mmopt::NarrowB910B3(),
                                  mmopt::CoreMemory memory = {}) {
        try {
            std::cout << name << ": source_geometry_objective_us="
                      << mmopt::ReducedObjectiveUs(w, hardware, decision, memory) << '\n';
        } catch (const std::exception& error) {
            std::cout << name << ": source_geometry_rejected=" << error.what() << '\n';
        }
    };
    mmopt::PlanDecision base;
    base.family = mmopt::KernelFamily::Base;
    base.used_cores = 20;
    base.single_core_m = base.base_m = 128;
    base.single_core_n = base.base_n = 256;
    base.single_core_k = 73728;
    base.base_k = 64;
    mmopt::CoreMemory base_memory;
    base_memory.base_l0c_buffers = 1;
    source_score("source_BASE", base, h, base_memory);
    mmopt::PlanDecision split;
    split.family = mmopt::KernelFamily::DeterministicSplitK;
    split.used_cores = 20;
    split.single_core_m = 2560;
    split.single_core_n = 384;
    split.single_core_k = 384;
    split.base_m = split.base_n = split.base_k = 128;
    split.step_k = 3;
    auto expanded_workspace = h;
    expanded_workspace.workspace_limit_bytes = 2ULL * 1024 * 1024 * 1024;
    source_score("source_SplitK", split, expanded_workspace);
    for (const auto k : {65408ULL, 65536ULL, 73728ULL}) {
        auto wk = w;
        wk.k = k;
        auto bk = base;
        bk.single_core_k = k;
        std::cout << "source_BASE_K=" << k << ": objective_us="
                  << mmopt::ReducedObjectiveUs(wk, h, bk, base_memory) << '\n';
        std::cout << "source_SplitK_K=" << k << ": objective_us="
                  << mmopt::ReducedObjectiveUs(wk, expanded_workspace, split) << '\n';
    }
}
