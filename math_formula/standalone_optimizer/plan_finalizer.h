#pragma once

#include "optimizer.h"

#include <cstdint>
#include <string>

namespace mmopt {

enum class KernelFamily { Base, DeterministicSplitK };

// The seven dimensions represented by the current analytical event builder.
// step_k is a derived consistency check: single_core_k = base_k * step_k for
// Split-K. It is one for BASE, whose single_core_k covers the full K axis.
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
};

// Reference capacities for a full Ascend 910B3. Pass queried capacities
// instead when the target exposes a smaller partition or different revision.
struct CoreMemory {
    std::uint64_t l1_bytes = 524288;
    std::uint64_t l0a_bytes = 65536;
    std::uint64_t l0b_bytes = 65536;
    std::uint64_t l0c_bytes = 131072;
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
// shape and base tiles, and BASE or deterministic Split-K. Includes the
// source's head ND2NZ conversion branch for long K (e.g. 16x16x102400).
// Throws std::invalid_argument for an unsupported or infeasible decision.
FinalizedPlan FinalizePlan(const Workload& workload, const Hardware& hardware,
                           const PlanDecision& decision, const std::string& id,
                           const CoreMemory& memory = {});

}  // namespace mmopt
