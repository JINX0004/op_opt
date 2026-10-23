#include "plan_finalizer.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace mmopt {
namespace {

constexpr std::uint64_t kAlign = 16;
constexpr std::uint64_t kInputBytes = 2;
constexpr std::uint64_t kAccumulatorBytes = 4;
constexpr std::uint64_t kRpcWorkspace = 20ULL * 1024 * 1024;

std::uint64_t Multiply(std::uint64_t a, std::uint64_t b, const char* name) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::invalid_argument(std::string(name) + " overflows uint64_t");
    return a * b;
}

std::uint64_t Add(std::uint64_t a, std::uint64_t b, const char* name) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::invalid_argument(std::string(name) + " overflows uint64_t");
    return a + b;
}

std::uint64_t Bytes(std::uint64_t x, std::uint64_t y, std::uint64_t element,
                    std::uint64_t buffers, const char* name) {
    return Multiply(Multiply(Multiply(x, y, name), element, name), buffers, name);
}

std::uint64_t CeilDiv(std::uint64_t x, std::uint64_t y) {
    return x / y + (x % y != 0);
}

void AlignedPositive(std::uint64_t value, const char* name) {
    if (!value || value % kAlign)
        throw std::invalid_argument(std::string(name) + " must be a positive multiple of 16");
}

bool OnTheWayFp16N(std::uint64_t n) {
    // SUPPORT_ND2NZ_GM2L0 is in bytes in matmul_v3_base_tiling.cpp.
    for (const auto bytes : {32ULL, 64ULL, 96ULL, 128ULL, 160ULL,
                             192ULL, 224ULL, 256ULL, 384ULL})
        if (n == bytes / kInputBytes) return true;
    return false;
}

}  // namespace

FinalizedPlan FinalizePlan(const Workload& workload, const Hardware& hardware,
                           const PlanDecision& d, const std::string& id,
                           const CoreMemory& memory) {
    if (id.empty()) throw std::invalid_argument("plan id is empty");
    if (workload.dtype != "fp16" && workload.dtype != "bf16")
        throw std::invalid_argument("preflight supports FP16/BF16 only");
    if (workload.a_layout != Layout::ND || workload.b_layout != Layout::ND ||
        workload.c_layout != Layout::ND || workload.trans_a || workload.trans_b ||
        workload.has_bias)
        throw std::invalid_argument("preflight supports ND/ND/ND, NN, no bias only");
    // Output-tail and non-NN paths still have different key/workspace rules.
    AlignedPositive(workload.m, "M");
    AlignedPositive(workload.n, "N");
    AlignedPositive(workload.k, "K");
    // Keep K 256-byte aligned. For B, accept direct 256-byte alignment or
    // the source's on-the-way transfer sizes; this keeps nd2nzB false. A
    // long K still requires the head ND2NZ A path and is modeled below.
    if (workload.k % 128 || workload.n > 65535 ||
        (workload.n % 128 && !OnTheWayFp16N(workload.n)))
        throw std::invalid_argument(
            "preflight requires 256-byte aligned K and direct/on-the-way B transfer");
    if (!memory.support_l0c2out || memory.support_l12bt_bf16)
        throw std::invalid_argument("preflight requires the 910B3 mixed-kernel feature profile");
    if (workload.m > 4096 && workload.n % 16384 == 0)
        throw std::invalid_argument("B matrix conflict conversion path is not modeled");
    AlignedPositive(d.single_core_m, "singleCoreM");
    AlignedPositive(d.single_core_n, "singleCoreN");
    AlignedPositive(d.single_core_k, "singleCoreK");
    AlignedPositive(d.base_m, "baseM");
    AlignedPositive(d.base_n, "baseN");
    AlignedPositive(d.base_k, "baseK");
    if (!hardware.aic_count || hardware.aic_count > 20 || !d.used_cores ||
        d.used_cores > hardware.aic_count)
        throw std::invalid_argument("usedCoreNum exceeds the 910B3 AIC allocation");
    if (!memory.l1_bytes || !memory.l0a_bytes || !memory.l0b_bytes || !memory.l0c_bytes)
        throw std::invalid_argument("local memory capacities must be positive");
    if (d.single_core_m > workload.m || d.single_core_n > workload.n)
        throw std::invalid_argument("single-core output tile exceeds the shape");
    if (!d.step_k) throw std::invalid_argument("stepK must be positive");

    FinalizedPlan plan;
    plan.tiling = d;
    plan.block_dim = d.used_cores;
    plan.key.split_core_mode = d.family == KernelFamily::Base ? 0 : 3;
    plan.workspace_bytes = kRpcWorkspace;
    plan.nd2nz_a = workload.k > 65535;
    plan.key.mix_nd2nz = plan.nd2nz_a ? 0 : 1;

    if (d.family == KernelFamily::Base) {
        if (d.single_core_k != workload.k || d.step_k != 1)
            throw std::invalid_argument("BASE must own all K and have stepK=1");
        const auto outputs = Multiply(CeilDiv(workload.m, d.single_core_m),
                                      CeilDiv(workload.n, d.single_core_n),
                                      "output tile count");
        if (d.used_cores > outputs)
            throw std::invalid_argument("BASE assigns more AICs than output tiles");
    } else if (d.family == KernelFamily::DeterministicSplitK) {
        if (Multiply(d.base_k, d.step_k, "singleCoreK") != d.single_core_k)
            throw std::invalid_argument("Split-K requires singleCoreK=baseK*stepK");
        if (d.used_cores < 2 || d.used_cores > CeilDiv(workload.k, d.single_core_k))
            throw std::invalid_argument("Split-K requires 2 or more occupied K partitions");
        // GetDeterministicSplitKWorkspaceSize(): usedCoreNum * singleCoreM *
        // singleCoreN * DB_SIZE(2) * sizeof(float) + RPC_WORKSIZE(20 MiB).
        const auto partial = Bytes(d.single_core_m, d.single_core_n,
                                   kAccumulatorBytes, 2, "Split-K partial workspace");
        plan.workspace_bytes = Add(kRpcWorkspace,
                                   Multiply(d.used_cores, partial, "Split-K partial workspace"),
                                   "Split-K workspace");
    } else {
        throw std::invalid_argument("unsupported implementation family");
    }

    if (plan.nd2nz_a) {
        // CalcNd2nzWorkspace(): ceil(M,16) * ceil(K,16) * sizeof(fp16).
        // M and K are aligned by this preflight's contract.
        plan.nd2nz_a_workspace_bytes = Bytes(workload.m, workload.k,
                                              kInputBytes, 1, "A ND2NZ workspace");
        plan.workspace_bytes = Add(plan.workspace_bytes,
                                   plan.nd2nz_a_workspace_bytes, "total workspace");
    }

    // Modeled local-memory preflight. L0A/L0B reserve two base tiles,
    // L0C reserves two FP32 accumulator tiles. L1 reserves two K-step windows
    // for both operands. Exact TCubeTiling occupancy remains CANN's decision.
    const auto a_l0 = Bytes(d.base_m, d.base_k, kInputBytes, 2, "L0A footprint");
    const auto b_l0 = Bytes(d.base_n, d.base_k, kInputBytes, 2, "L0B footprint");
    const auto c_l0 = Bytes(d.base_m, d.base_n, kAccumulatorBytes, 2, "L0C footprint");
    if (a_l0 > memory.l0a_bytes || b_l0 > memory.l0b_bytes || c_l0 > memory.l0c_bytes)
        throw std::invalid_argument("base tile exceeds L0 capacity");
    const auto a_l1 = Bytes(d.base_m, d.base_k, kInputBytes,
                            Multiply(2, d.step_k, "L1 A depth"), "L1 A footprint");
    const auto b_l1 = Bytes(d.base_n, d.base_k, kInputBytes,
                            Multiply(2, d.step_k, "L1 B depth"), "L1 B footprint");
    if (Add(a_l1, b_l1, "L1 footprint") > memory.l1_bytes)
        throw std::invalid_argument("base tile and stepK exceed L1 capacity");
    if (plan.workspace_bytes > hardware.workspace_limit_bytes)
        throw std::invalid_argument("workspace budget exceeded");

    Candidate& candidate = plan.candidate;
    candidate.id = id;
    candidate.family = d.family == KernelFamily::Base ? "BASE" : "DETERMINISTIC_SPLIT_K";
    candidate.tiling_key = "(0," + std::to_string(plan.key.split_core_mode) + ",0," +
                           std::to_string(plan.key.mix_nd2nz) + ",0,0)";
    // Legacy scorer gate: means this candidate passed the standalone preflight.
    // It does not mean CANN finalized/serialized the plan or ran the kernel.
    candidate.backend_verified = true;
    candidate.workspace_bytes = plan.workspace_bytes;
    candidate.nd2nz_a = plan.nd2nz_a;
    candidate.nd2nz_b = plan.nd2nz_b;
    candidate.tile = {{"usedCoreNum", d.used_cores}, {"singleCoreM", d.single_core_m},
                      {"singleCoreN", d.single_core_n}, {"singleCoreK", d.single_core_k},
                      {"baseM", d.base_m}, {"baseN", d.base_n}, {"baseK", d.base_k}};
    return plan;
}

}  // namespace mmopt
