#include "plan_finalizer.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace mmopt {
namespace {

constexpr std::uint64_t kAlign = 16;
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

bool OnTheWayN(std::uint64_t n, std::uint64_t b_bytes) {
    // SUPPORT_ND2NZ_GM2L0 is in bytes in matmul_v3_base_tiling.cpp.
    for (const auto bytes : {32ULL, 64ULL, 96ULL, 128ULL, 160ULL,
                             192ULL, 224ULL, 256ULL, 384ULL})
        if (n == bytes / b_bytes) return true;
    return false;
}

}  // namespace

FinalizedPlan FinalizePlan(const Workload& workload, const Hardware& hardware,
                           const PlanDecision& d, const std::string& id,
                           const CoreMemory& memory) {
    if (id.empty()) throw std::invalid_argument("plan id is empty");
    const auto types = WorkloadTypes(workload);
    // FP32 modes 2 and 5 use direct FP32 Cube output. FP16/BF16
    // single-core modes additionally run a vector cast/output stage.
    if ((types.a != "fp16" && types.a != "bf16" && types.a != "fp32") ||
        types.b != types.a || types.c != types.a ||
        types.accumulator != "fp32" || workload.dtype != types.a)
        throw std::invalid_argument("preflight supports same-type FP16/BF16/FP32 with FP32 accumulation");
    const auto a_bytes = ElementBytes(types.a);
    const auto b_bytes = ElementBytes(types.b);
    const auto accumulator_bytes = ElementBytes(types.accumulator);
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
    const auto transfer_alignment = 256 / a_bytes;
    if (workload.k % transfer_alignment || workload.n > 65535 ||
        (workload.n % (256 / b_bytes) && !OnTheWayN(workload.n, b_bytes)))
        throw std::invalid_argument(
            "preflight requires 256-byte aligned K and direct/on-the-way B transfer");
    // FP32 K>65535 invokes the source's ND->NZ conversion path. Its FP32
    // staging and transfer objective still need a separate device audit.
    if (types.a == "fp32" && workload.k > 65535)
        throw std::invalid_argument("FP32 converted-A path is not modeled yet");
    if (!memory.support_l0c2out || memory.support_l12bt_bf16)
        throw std::invalid_argument("preflight requires the 910B3 mixed-kernel feature profile");
    if (workload.m > 4096 && workload.n % 16384 == 0)
        throw std::invalid_argument("B matrix conflict conversion path is not modeled");
    // The source aligns FP32 single-core M partitions to 32 bytes (8 elements).
    if (types.a == "fp32" &&
        (d.family == KernelFamily::SingleCoreSplitKMode2 ||
         d.family == KernelFamily::SingleCoreSplitKMode5)) {
        if (!d.single_core_m || d.single_core_m % 8)
            throw std::invalid_argument("FP32 single-core singleCoreM must be 8-aligned");
    } else {
        AlignedPositive(d.single_core_m, "singleCoreM");
    }
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
    if (memory.base_l0c_buffers < 1 || memory.base_l0c_buffers > 2)
        throw std::invalid_argument("BASE L0C buffer count must be 1 or 2");
    // Source deterministic Split-K can use the 3x128 outer M/N tile even
    // when the corresponding output axis is only 128 or 256. BASE still
    // requires a tile within the shape for its spatial AIC assignment.
    const bool single_core_split = d.family == KernelFamily::SingleCoreSplitKMode2 ||
        d.family == KernelFamily::SingleCoreSplitKMode5 ||
        d.family == KernelFamily::SingleCoreSplitKMode6;
    if (types.a == "fp32" && d.family == KernelFamily::SingleCoreSplitKMode6)
        throw std::invalid_argument("FP32 mode 6 is absent from this source host path");
    const bool fp32_base_n_padding = types.a == "fp32" && d.family == KernelFamily::Base &&
        workload.n >= 64 && workload.n < 128 && d.single_core_n == 128 && d.base_n == 128;
    if ((d.family == KernelFamily::Base || single_core_split) &&
        (d.single_core_m > workload.m ||
         (d.single_core_n > workload.n && !fp32_base_n_padding)))
        throw std::invalid_argument("spatial single-core output tile exceeds the shape");
    if (!d.step_k) throw std::invalid_argument("stepK must be positive");

    FinalizedPlan plan;
    plan.tiling = d;
    plan.block_dim = d.used_cores;
    switch (d.family) {
        case KernelFamily::Base: plan.key.split_core_mode = 0; break;
        case KernelFamily::DeterministicSplitK: plan.key.split_core_mode = 3; break;
        case KernelFamily::SingleCoreSplitKMode2: plan.key.split_core_mode = 2; break;
        case KernelFamily::SingleCoreSplitKMode5: plan.key.split_core_mode = 5; break;
        case KernelFamily::SingleCoreSplitKMode6: plan.key.split_core_mode = 6; break;
    }
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
        if (d.base_k != (types.a == "fp32" ? 64 : 128))
            throw std::invalid_argument("Split-K baseK does not match the dtype path");
        if (Multiply(d.base_k, d.step_k, "singleCoreK") != d.single_core_k)
            throw std::invalid_argument("Split-K requires singleCoreK=baseK*stepK");
        if (d.used_cores < 2 || d.used_cores > CeilDiv(workload.k, d.single_core_k))
            throw std::invalid_argument("Split-K requires 2 or more occupied K partitions");
        // GetDeterministicSplitKWorkspaceSize(): usedCoreNum * singleCoreM *
        // singleCoreN * DB_SIZE(2) * sizeof(float) + RPC_WORKSIZE(20 MiB).
        const auto partial = Bytes(d.single_core_m, d.single_core_n,
                                   accumulator_bytes, 2, "Split-K partial workspace");
        plan.workspace_bytes = Add(kRpcWorkspace,
                                   Multiply(d.used_cores, partial, "Split-K partial workspace"),
                                   "Split-K workspace");
    } else if (single_core_split) {
        const auto sc_base_k = types.a == "fp32" ? 64ULL : 128ULL;
        const auto n_alignment = 256ULL / b_bytes;
        if (workload.n % n_alignment ||
            d.base_m != 128 || d.base_n != 128 || d.base_k != sc_base_k ||
            Multiply(d.base_k, d.step_k, "singleCoreK") != d.single_core_k)
            throw std::invalid_argument("single-core Split-K requires dtype-specific aligned N and base tiles");
        const auto outputs = Multiply(CeilDiv(workload.m, d.single_core_m),
                                      CeilDiv(workload.n, d.single_core_n), "output tile count");
        if (d.used_cores > outputs)
            throw std::invalid_argument("single-core Split-K assigns more AICs than spatial tiles");
        const bool mkn_small_k = workload.k == 1536 && workload.m == 384 &&
            workload.n >= 49152 &&
            static_cast<long double>(workload.m) * workload.n * accumulator_bytes <=
                0.65L * memory.l2_bytes;
        const bool nkm_small_k = workload.k == 1536 && workload.n == 384 &&
            workload.m >= 49152 &&
            static_cast<long double>(workload.m) * workload.n * accumulator_bytes <=
                0.65L * memory.l2_bytes;
        if (d.family == KernelFamily::SingleCoreSplitKMode5) {
            if (!nkm_small_k || workload.m % 128 || workload.n % 128 ||
                plan.nd2nz_a || d.step_m != 1 ||
                d.step_n != 3 || d.step_k != 3)
                throw std::invalid_argument("NN/ND mode 5 requires the K=1536 NK33 channel");
        } else {
            const bool mk33 = d.step_m == 3 && d.step_n == 1 && d.step_k == 3;
            const bool block24 = (d.step_m == 1 || d.step_m == 2) &&
                                 d.step_n == 1 && d.step_k == 4;
            if (!mk33 && !block24)
                throw std::invalid_argument("FP16/BF16 modes 2/6 require MK33 or 2x4 source block");
            const bool short_side = workload.m < 384 || workload.n < 384;
            if ((short_side && d.step_m != 1) ||
                (!short_side && d.step_m == 1))
                throw std::invalid_argument("FP16/BF16 single-core Split-K block does not match the short-side rule");
            if (mkn_small_k && !mk33)
                throw std::invalid_argument("FP16/BF16 K=1536 MKN channel forces MK33");
            if (workload.k < 27392 && !mkn_small_k)
                throw std::invalid_argument("mode 2/6 requires long K or K=1536 MKN gate");
            if (d.family == KernelFamily::SingleCoreSplitKMode6 &&
                (workload.k > 65535 || mkn_small_k))
                throw std::invalid_argument("FP16/BF16 mode 6 requires its GM-to-L1 shape gate");
            if (!mkn_small_k) {
                const long double load = static_cast<long double>(workload.m) * workload.n /
                    (static_cast<long double>(d.single_core_m) * d.single_core_n * hardware.aic_count);
                if (d.single_core_n < 512 || load < 0.7L ||
                    (workload.n >= 896 && workload.n <= 2048 &&
                     (d.single_core_n <= 640 || load < 0.85L)))
                    throw std::invalid_argument("FP16/BF16 single-core Split-K spatial host gate failed");
            }
        }
        // CalcSplitCoreWorkspace: one aligned FP32 output image plus RPC.
        const auto aligned_n = Multiply(CeilDiv(workload.n, n_alignment), n_alignment,
                                        "aligned single-core N");
        plan.workspace_bytes = Add(kRpcWorkspace,
            Bytes(workload.m, aligned_n, accumulator_bytes, 1,
                  "single-core Split-K output workspace"),
            "single-core Split-K workspace");
    } else {
        throw std::invalid_argument("unsupported implementation family");
    }

    if (plan.nd2nz_a) {
        // CalcNd2nzWorkspace(): ceil(M,16) * ceil(K,16) * sizeof(A).
        // M and K are aligned by this preflight's contract.
        plan.nd2nz_a_workspace_bytes = Bytes(workload.m, workload.k,
                                              a_bytes, 1, "A ND2NZ workspace");
        plan.workspace_bytes = Add(plan.workspace_bytes,
                                   plan.nd2nz_a_workspace_bytes, "total workspace");
    }

    // Modeled local-memory preflight. L0A/L0B reserve two base tiles,
    // BASE L0C uses the selected one/two-buffer policy; Split-K reserves
    // two FP32 accumulator tiles. L1 reserves two K-step windows for both
    // operands. Exact TCubeTiling occupancy remains CANN's decision.
    const auto a_l0 = Bytes(d.base_m, d.base_k, a_bytes, 2, "L0A footprint");
    const auto b_l0 = Bytes(d.base_n, d.base_k, b_bytes, 2, "L0B footprint");
    const auto c_l0 = Bytes(d.base_m, d.base_n, accumulator_bytes,
                            d.family == KernelFamily::Base
                                ? memory.base_l0c_buffers : 2,
                            "L0C footprint");
    if (a_l0 > memory.l0a_bytes || b_l0 > memory.l0b_bytes || c_l0 > memory.l0c_bytes)
        throw std::invalid_argument("base tile exceeds L0 capacity");
    std::uint64_t l1_footprint = 0;
    if (single_core_split) {
        // The source tiler reserves 480 KiB for 3x3 and 512 KiB for
        // 2x4, including its short-side variant. The mode-6 kernel may
        // actively use only 384 KiB on that short-side path.
        // The queried usable L1 reports 256 bytes less than the nominal
        // allocation; the source reserves that 256-byte margin itself.
        l1_footprint = d.family == KernelFamily::SingleCoreSplitKMode5 ||
                       d.step_m == 3 ? 480ULL * 1024 : 512ULL * 1024;
    } else {
        const auto a_l1 = Bytes(d.base_m, d.base_k, a_bytes,
                                Multiply(2, d.step_k, "L1 A depth"), "L1 A footprint");
        const auto b_l1 = Bytes(d.base_n, d.base_k, b_bytes,
                                Multiply(2, d.step_k, "L1 B depth"), "L1 B footprint");
        l1_footprint = Add(a_l1, b_l1, "L1 footprint");
    }
    if (l1_footprint > memory.l1_bytes + (single_core_split ? 256 : 0))
        throw std::invalid_argument("base tile and stepK exceed L1 capacity");
    if (plan.workspace_bytes > hardware.workspace_limit_bytes)
        throw std::invalid_argument("workspace budget exceeded");

    Candidate& candidate = plan.candidate;
    candidate.id = id;
    switch (d.family) {
        case KernelFamily::Base: candidate.family = "BASE"; break;
        case KernelFamily::DeterministicSplitK: candidate.family = "DETERMINISTIC_SPLIT_K"; break;
        case KernelFamily::SingleCoreSplitKMode2: candidate.family = "SINGLE_CORE_SPLIT_K_MODE2"; break;
        case KernelFamily::SingleCoreSplitKMode5: candidate.family = "SINGLE_CORE_SPLIT_K_MODE5"; break;
        case KernelFamily::SingleCoreSplitKMode6: candidate.family = "SINGLE_CORE_SPLIT_K_MODE6"; break;
    }
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
    if (single_core_split) {
        candidate.tile["stepK"] = d.step_k;
        candidate.tile["stepM"] = d.step_m;
        candidate.tile["stepN"] = d.step_n;
    }
    return plan;
}

}  // namespace mmopt
