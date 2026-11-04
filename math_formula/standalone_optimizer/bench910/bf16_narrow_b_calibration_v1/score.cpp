#include "model/hardware_profile.h"

#include <cstdint>
#include <iostream>

namespace {

void Report(const char* label, const mmopt::Workload& workload,
            const mmopt::Hardware& hardware) {
    const auto result = mmopt::SolveReducedMip(
        workload, hardware, {}, mmopt::SourceCompatibleDomain());
    const auto print = [&](const char* family, const std::optional<mmopt::MipPlan>& plan) {
        if (!plan) return;
        std::cout << label << ',' << family << ',' << plan->surrogate_us << ','
                  << plan->decision.used_cores << ',' << plan->decision.base_k << '\n';
    };
    print("winner", result.winner);
    print("BASE", result.base);
    print("deterministic", result.deterministic_split_k);
    print("mode2", result.single_core_split_k_mode2);
    print("mode5", result.single_core_split_k_mode5);
    print("mode6", result.single_core_split_k_mode6);

    mmopt::PlanDecision source_base;
    source_base.family = mmopt::KernelFamily::Base;
    source_base.used_cores = 1;
    source_base.single_core_m = 16;
    source_base.single_core_n = 16;
    source_base.single_core_k = workload.k;
    source_base.base_m = 16;
    source_base.base_n = 16;
    source_base.base_k = 64;  // Actual BF16 tiler output for 16x16x65408.
    std::cout << label << ",source_BASE_64,"
              << mmopt::ReducedObjectiveUs(workload, hardware, source_base)
              << ",1,64\n";
    source_base.base_k = 1024;  // Generic model's winning hypothetical tile.
    std::cout << label << ",model_BASE_1024,"
              << mmopt::ReducedObjectiveUs(workload, hardware, source_base)
              << ",1,1024\n";
}

}  // namespace

int main() {
    const mmopt::Workload workload{16, 16, 65408, "bf16"};
    const auto generic = mmopt::ExperimentalBf16910B3();
    auto fp16_rows = generic;
    const auto measured_fp16 = mmopt::NarrowB910B3();
    for (const auto* name : {"base_mte2_b_narrow_rows_one",
                             "base_mte2_b_narrow_rows_full"}) {
        fp16_rows.primitives[name] = measured_fp16.primitives.at(name);
    }
    Report("BF16_generic", workload, generic);
    Report("BF16_with_FP16_rows", workload, fp16_rows);
}
