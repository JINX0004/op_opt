#pragma once

#include "optimizer.h"

#include <cstdint>
#include <string>

namespace mmopt {

enum class KernelFamily {
    Base, DeterministicSplitK, SingleCoreSplitKMode2,
    SingleCoreSplitKMode5, SingleCoreSplitKMode6
};

// Shared plan fields for the reduced objectives. The original event builder
// uses the seven M/N/K tile fields; the FP16/BF16 single-core models also use
// step_m and step_n. For Split-K, single_core_k=base_k*step_k. BASE uses
// step_k=1 and single_core_k covering the full K axis.
struct PlanDecision {
    KernelFamily family = KernelFamily::Base;
    std::uint64_t used_cores = 0;
    std::uint64_t single_core_m = 0;
    std::uint64_t single_core_n = 0;
    std::uint64_t single_core_k = 0;
    std::uint64_t base_m = 0;
    std::uint64_t base_n = 0;
    std::uint64_t base_k = 0;
    std::uint64_t step_k = 1;
    // Source single-core Split-K block: MK33=(3,1,3), 2x4=(2,1,4),
    // short-side=(1,1,4), and NK33=(1,3,3). Legacy families ignore these.
    std::uint64_t step_m = 1;
    std::uint64_t step_n = 1;
};

// Usable capacities queried on an Ascend 910B3 with CANN 9.0.0. Pass fresh
// queried capacities for another installation or partition. L1/UB are 256 B
// below their nominal 512/192 KiB sizes on this installed platform API.
struct CoreMemory {
    std::uint64_t l1_bytes = 524032;
    std::uint64_t l0a_bytes = 65536;
    std::uint64_t l0b_bytes = 65536;
    std::uint64_t l0c_bytes = 131072;
    std::uint64_t l2_bytes = 201326592;
    // Plain BASE may emit dbL0C=1; keep the conservative two-buffer
    // preflight by default and set this to 1 only for that source path.
    std::uint64_t base_l0c_buffers = 2;
    bool support_l0c2out = true;
    bool support_l12bt_bf16 = false;
};

// Template arguments from mat_mul_v3_tiling_key.h. This is the tuple, not the
// packed GET_TPL_TILING_KEY value (that macro comes from CANN).
struct KernelKeyTuple {
    std::uint32_t load_mode = 0;
    std::uint32_t split_core_mode = 0;
    std::uint32_t fix_opti = 0;
    std::uint32_t mix_nd2nz = 1;
    std::uint32_t special_opti = 0;
    std::uint32_t fp32_addmm = 0;
};

struct FinalizedPlan {
    PlanDecision tiling;
    KernelKeyTuple key;
    std::uint64_t block_dim = 0;
    std::uint32_t schedule_mode = 1;
    std::uint64_t workspace_bytes = 0;
    bool nd2nz_a = false;
    bool nd2nz_b = false;
    std::uint64_t nd2nz_a_workspace_bytes = 0;
    std::uint64_t nd2nz_b_workspace_bytes = 0;
    // The current scorer accepts a Candidate; it is a model plan, not
    // serialized CANN MatmulTilingData or a claim of device executability.
    Candidate candidate;
};

// Narrow standalone preflight for ND/ND/ND NN, FP16/BF16, no bias, aligned
// shape and base tiles. FP16/BF16 single-core Split-K modes 2/5/6 additionally
// require source block patterns and their mode-specific gates. Includes the
// source's head ND2NZ conversion branch for long K (e.g. 16x16x102400).
// Throws std::invalid_argument for an unsupported or infeasible decision.
FinalizedPlan FinalizePlan(const Workload& workload, const Hardware& hardware,
                           const PlanDecision& decision, const std::string& id,
                           const CoreMemory& memory = {});

}  // namespace mmopt
