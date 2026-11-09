#include "hardware_profile.h"
#include "matmul_v3_mip_online.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

int main() {
    using namespace mmopt;
    const Workload fp16{16, 16, 102400, "fp16"};
    const Workload bf16{16, 16, 102400, "bf16"};
    assert(WorkloadTypes(fp16).a == "fp16");
    assert(WorkloadTypes(fp16).c == "fp16");
    assert(WorkloadTypes(fp16).accumulator == "fp32");
    assert(ElementBytes("fp16") == 2 && ElementBytes("bf16") == 2);
    assert(ElementBytes("fp32") == 4);

    const auto fp16_hardware = Calibrated910B3();
    const auto bf16_hardware = ExperimentalBf16910B3();
    assert(fp16_hardware.primitives.count("sc_mode5_fixpipe") == 1);
    assert(bf16_hardware.primitives.count("sc_mode5_fixpipe") == 1);
    assert(bf16_hardware.primitives.at("sc_mode5_fixpipe").rate_per_us == 108000.0);
    assert(bf16_hardware.primitives.at("sc_mode5_iterate").fixed_latency_us == 1.76);
    assert(fp16_hardware.primitives.count("base_mte2_b_narrow_rows_one") == 0);
    assert(bf16_hardware.primitives.count("base_mte2_b_narrow_rows_one") == 1);
    const auto near_k_limit = SolveReducedMip({16, 16, 65408, "bf16"},
                                              bf16_hardware, {}, SourceCompatibleDomain());
    assert(near_k_limit.winner);
    assert(near_k_limit.winner->decision.family == KernelFamily::DeterministicSplitK);
    // The source's 64-slice L1 step is one read startup, not 64 baseK
    // startups. This score previously exceeded 750 us for the emitted tile.
    PlanDecision emitted_base;
    emitted_base.family = KernelFamily::Base;
    emitted_base.used_cores = 1;
    emitted_base.single_core_m = emitted_base.base_m = 16;
    emitted_base.single_core_n = emitted_base.base_n = 16;
    emitted_base.single_core_k = 65408;
    emitted_base.base_k = 64;
    const auto emitted_score = ReducedObjectiveUs({16, 16, 65408, "bf16"},
                                                   bf16_hardware, emitted_base);
    assert(emitted_score > 500.0 && emitted_score < 600.0);
    const auto domain = SourceCompatibleDomain();
    const auto a = SolveReducedMip(fp16, fp16_hardware, {}, domain);
    const auto b = SolveReducedMip(bf16, bf16_hardware, {}, domain);
    assert(a.base && a.deterministic_split_k && a.winner);
    assert(b.base && b.deterministic_split_k && b.winner);
    assert(a.base->candidate.workspace_bytes == b.base->candidate.workspace_bytes);

    const auto nkm = SolveReducedMip({49152, 384, 1536, "bf16"},
                                     bf16_hardware, {}, domain);
    assert(nkm.single_core_split_k_mode5 && nkm.winner);
    assert(nkm.winner->decision.family == KernelFamily::SingleCoreSplitKMode5);
    assert(nkm.winner->decision.single_core_m == 2464);
    assert(nkm.winner->decision.single_core_n == 384);
    const auto mkn = SolveReducedMip({384, 49152, 1536, "bf16"},
                                     bf16_hardware, {}, domain);
    assert(mkn.single_core_split_k_mode2);
    const auto long_k = SolveReducedMip({1792, 768, 32768, "bf16"},
                                        bf16_hardware, {}, domain);
    assert(long_k.single_core_split_k_mode6);

    Workload mixed{16, 16, 102400, "bf16"};
    mixed.types = {"bf16", "bf16", "fp32", "fp32"};
    bool rejected = false;
    try { (void)SolveReducedMip(mixed, bf16_hardware, {}, domain); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    unsetenv("MMV3_ONLINE_BF16_EXPERIMENT");
    assert(optiling::matmul_v3::LookupOnlineMipMode(
        16, 16, 102400, "bf16").mode == -1);
    setenv("MMV3_ONLINE_BF16_EXPERIMENT", "1", 1);
    const auto model_only = optiling::matmul_v3::LookupOnlineMipMode(
        256, 512, 16384, "bf16");
    PlanDecision source_base;
    source_base.family = KernelFamily::Base;
    source_base.used_cores = 4;
    source_base.single_core_m = source_base.base_m = 128;
    source_base.single_core_n = source_base.base_n = 256;
    source_base.single_core_k = 16384;
    source_base.base_k = 64;
    const auto source_scored = optiling::matmul_v3::LookupOnlineMipMode(
        256, 512, 16384, "bf16", &source_base);
    assert(model_only.mode == 0 && source_scored.mode == 3);
    assert(optiling::matmul_v3::LookupOnlineMipMode(
        16, 16, 102400, "bf16").mode >= 0);
    assert(optiling::matmul_v3::LookupOnlineMipMode(
        16, 16, 102400, mixed.types).mode == -1);

    // Device-audited FP32 source tuples exercise the 4-byte L0/L1
    // footprint and dtype-specific K tiles without changing FP16/BF16.
    const Workload fp32_base{2048, 2048, 4096, "fp32"};
    PlanDecision emitted_fp32_base;
    emitted_fp32_base.family = KernelFamily::Base;
    emitted_fp32_base.used_cores = 20;
    emitted_fp32_base.single_core_m = emitted_fp32_base.base_m = 128;
    emitted_fp32_base.single_core_n = emitted_fp32_base.base_n = 256;
    emitted_fp32_base.single_core_k = 4096;
    emitted_fp32_base.base_k = 32;
    CoreMemory fp32_base_memory;
    fp32_base_memory.base_l0c_buffers = 1;
    const auto fp32_base_plan = FinalizePlan(fp32_base, fp16_hardware,
        emitted_fp32_base, "fp32_source_base", fp32_base_memory);
    assert(fp32_base_plan.key.split_core_mode == 0);
    assert(fp32_base_plan.workspace_bytes == 20ULL * 1024 * 1024);
    assert(std::isfinite(ReducedObjectiveUs(fp32_base, fp16_hardware,
        emitted_fp32_base, fp32_base_memory)));

    const Workload fp32_split{128, 128, 32768, "fp32"};
    PlanDecision emitted_fp32_split;
    emitted_fp32_split.family = KernelFamily::DeterministicSplitK;
    emitted_fp32_split.used_cores = 20;
    emitted_fp32_split.single_core_m = 384;
    emitted_fp32_split.single_core_n = 128;
    emitted_fp32_split.single_core_k = 192;
    emitted_fp32_split.base_m = emitted_fp32_split.base_n = 128;
    emitted_fp32_split.base_k = 64;
    emitted_fp32_split.step_k = 3;
    const auto fp32_split_plan = FinalizePlan(fp32_split, fp16_hardware,
        emitted_fp32_split, "fp32_source_split");
    assert(fp32_split_plan.key.split_core_mode == 3);
    assert(fp32_split_plan.workspace_bytes == 28835840);
    assert(std::isfinite(ReducedObjectiveUs(fp32_split, fp16_hardware,
        emitted_fp32_split)));
    MipTileDomain split_domain;
    split_domain.single_core_m = {384};
    split_domain.single_core_n = {128};
    split_domain.base_m = split_domain.base_n = {128};
    const auto fp32_result = SolveReducedMip(fp32_split, fp16_hardware,
        {}, split_domain);
    assert(fp32_result.deterministic_split_k);
    assert(fp32_result.deterministic_split_k->decision.base_k == 64);
    assert(!fp32_result.single_core_split_k_mode2 &&
           !fp32_result.single_core_split_k_mode5 &&
           !fp32_result.single_core_split_k_mode6);
    // The NN/ND FP32 source emits mode 2 with baseK=64, K step 4 and
    // direct FP32 cube output. The M extent is 32-byte, not 16-element,
    // aligned; the large-M source tuple therefore needs M=824.
    const Workload fp32_sc{2816, 1024, 36864, "fp32"};
    PlanDecision emitted_fp32_sc;
    emitted_fp32_sc.family = KernelFamily::SingleCoreSplitKMode2;
    emitted_fp32_sc.used_cores = 20;
    emitted_fp32_sc.single_core_m = 144;
    emitted_fp32_sc.single_core_n = 1024;
    emitted_fp32_sc.single_core_k = 256;
    emitted_fp32_sc.base_m = emitted_fp32_sc.base_n = 128;
    emitted_fp32_sc.base_k = 64;
    emitted_fp32_sc.step_m = 2;
    emitted_fp32_sc.step_n = 1;
    emitted_fp32_sc.step_k = 4;
    const auto fp32_sc_plan = FinalizePlan(fp32_sc, ExperimentalFp32910B3(),
        emitted_fp32_sc, "fp32_source_mode2");
    assert(fp32_sc_plan.key.split_core_mode == 2);
    const auto fp32_mode2_score = ReducedObjectiveUs(fp32_sc,
        ExperimentalFp32910B3(), emitted_fp32_sc);
    assert(std::isfinite(fp32_mode2_score));
    // The optional source-level pipeline uses nominal MTE1 A/B rates and
    // leaves the baseline selector profile untouched. The 910B3 sweep
    // measured this emitted tile at about 3943 us.
    const auto mode2_pipeline = ExperimentalFp32Mode2Pipeline910B3();
    const auto pipeline_score = ReducedObjectiveUs(fp32_sc,
        mode2_pipeline, emitted_fp32_sc);
    assert(std::abs(fp32_mode2_score - 3098.38319) < 0.01);
    assert(std::abs(pipeline_score - 3934.87950) < 0.01);
    auto missing_mode2_b = mode2_pipeline;
    missing_mode2_b.primitives.erase("sc_mode2_l0b");
    bool incomplete_mode2_profile_rejected = false;
    try { (void)ReducedObjectiveUs(fp32_sc, missing_mode2_b,
                                   emitted_fp32_sc); }
    catch (const std::invalid_argument&) {
        incomplete_mode2_profile_rejected = true;
    }
    assert(incomplete_mode2_profile_rejected);
    auto slow_vector = ExperimentalFp32910B3();
    slow_vector.primitives["sync"].fixed_latency_us = 1000.0;
    slow_vector.primitives["vector_add"].rate_per_us = 1.0;
    slow_vector.primitives["sc_cast"].rate_per_us = 1.0;
    assert(std::abs(ReducedObjectiveUs(fp32_sc, slow_vector,
        emitted_fp32_sc) - fp32_mode2_score) < 1e-6);
    const auto fp32_sc_search = SolveReducedMip(fp32_sc,
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_sc_search.single_core_split_k_mode2);
    assert(fp32_sc_search.single_core_split_k_mode2->decision.base_k == 64);
    assert(fp32_sc_search.single_core_split_k_mode2->decision.single_core_k ==
           fp32_sc_search.single_core_split_k_mode2->decision.base_k *
           fp32_sc_search.single_core_split_k_mode2->decision.step_k);
    assert(!fp32_sc_search.single_core_split_k_mode5 &&
           !fp32_sc_search.single_core_split_k_mode6);
    // FP32 K=1536 with large M and N=384 enters the source's NKM
    // small-K route. Its AIC writes FP32 directly, without an AIV cast.
    const Workload fp32_nkm{49152, 384, 1536, "fp32"};
    PlanDecision emitted_fp32_nkm;
    emitted_fp32_nkm.family = KernelFamily::SingleCoreSplitKMode5;
    emitted_fp32_nkm.used_cores = 20;
    emitted_fp32_nkm.single_core_m = 2464;
    emitted_fp32_nkm.single_core_n = 384;
    emitted_fp32_nkm.single_core_k = 192;
    emitted_fp32_nkm.base_m = emitted_fp32_nkm.base_n = 128;
    emitted_fp32_nkm.base_k = 64;
    emitted_fp32_nkm.step_m = 1;
    emitted_fp32_nkm.step_n = 3;
    emitted_fp32_nkm.step_k = 3;
    const auto fp32_nkm_plan = FinalizePlan(fp32_nkm, ExperimentalFp32910B3(),
        emitted_fp32_nkm, "fp32_source_mode5");
    assert(fp32_nkm_plan.key.split_core_mode == 5);
    assert(fp32_nkm_plan.workspace_bytes ==
           20ULL * 1024 * 1024 + 4ULL * 49152 * 384);
    const auto fp32_nkm_score = ReducedObjectiveUs(fp32_nkm,
        ExperimentalFp32910B3(), emitted_fp32_nkm);
    assert(std::isfinite(fp32_nkm_score));
    assert(std::abs(ReducedObjectiveUs(fp32_nkm, slow_vector,
        emitted_fp32_nkm) - fp32_nkm_score) < 1e-6);
    const auto fp32_nkm_search = SolveReducedMip(fp32_nkm,
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_nkm_search.single_core_split_k_mode5);
    assert(fp32_nkm_search.single_core_split_k_mode5->decision.base_k == 64);
    assert(fp32_nkm_search.single_core_split_k_mode5->decision.single_core_n == 384);
    assert(fp32_nkm_search.single_core_split_k_mode5->decision.step_n == 3);
    assert(fp32_nkm_search.single_core_split_k_mode5->decision.step_k == 3);
    assert(!fp32_nkm_search.single_core_split_k_mode6);
    const auto fp32_nkm_large = SolveReducedMip({73728, 384, 1536, "fp32"},
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_nkm_large.single_core_split_k_mode5);
    assert(fp32_nkm_large.single_core_split_k_mode5->decision.single_core_n == 384);
    bool fp32_nkm_bad_k = false;
    try { (void)ReducedObjectiveUs({49152, 384, 1600, "fp32"},
                                   ExperimentalFp32910B3(), emitted_fp32_nkm); }
    catch (const std::invalid_argument&) { fp32_nkm_bad_k = true; }
    assert(fp32_nkm_bad_k);
    auto large_fp32_sc = emitted_fp32_sc;
    large_fp32_sc.single_core_m = 824;
    large_fp32_sc.single_core_n = 2048;
    large_fp32_sc.step_m = 3;
    large_fp32_sc.step_k = 3;
    large_fp32_sc.single_core_k = 192;
    assert(std::isfinite(ReducedObjectiveUs({8192, 4096, 32768, "fp32"},
        ExperimentalFp32910B3(), large_fp32_sc)));
    // The audited 512x512x27328 source BASE tile fills all 20 AICs;
    // deterministic Split-K uses MK 3x3 rather than a free stepK=1 tile.
    const auto fp32_boundary = SolveReducedMip({512, 512, 27328, "fp32"},
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_boundary.base && fp32_boundary.deterministic_split_k &&
           fp32_boundary.winner);
    assert(fp32_boundary.base->decision.base_m == 112 &&
           fp32_boundary.base->decision.base_n == 128 &&
           fp32_boundary.base->decision.used_cores == 20);
    assert(fp32_boundary.deterministic_split_k->decision.single_core_m == 384 &&
           fp32_boundary.deterministic_split_k->decision.single_core_n == 512 &&
           fp32_boundary.deterministic_split_k->decision.step_k == 3);
    assert(fp32_boundary.winner->decision.family == KernelFamily::Base);
    auto fp32_unrestricted = SourceCompatibleDomain();
    fp32_unrestricted.fp32_source_patterns = false;
    const auto fp32_free = SolveReducedMip({512, 512, 27328, "fp32"},
        ExperimentalFp32910B3(), {}, fp32_unrestricted);
    assert(fp32_free.winner &&
           fp32_free.winner->decision.family == KernelFamily::DeterministicSplitK &&
           fp32_free.winner->decision.step_k == 1);
    const auto fp32_small = SolveReducedMip({16, 16, 6144, "fp32"},
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_small.deterministic_split_k &&
           fp32_small.deterministic_split_k->decision.single_core_m == 384 &&
           fp32_small.deterministic_split_k->decision.single_core_n == 16 &&
           fp32_small.deterministic_split_k->decision.used_cores == 10);
    const auto fp32_narrow = SolveReducedMip({2048, 64, 27392, "fp32"},
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_narrow.deterministic_split_k &&
           fp32_narrow.deterministic_split_k->decision.single_core_m == 2048 &&
           fp32_narrow.deterministic_split_k->decision.single_core_n == 384);
    const auto fp32_l2_split = SolveReducedMip({1024, 8192, 27392, "fp32"},
        ExperimentalFp32910B3(), {}, SourceCompatibleDomain());
    assert(fp32_l2_split.deterministic_split_k &&
           fp32_l2_split.deterministic_split_k->decision.single_core_m == 384 &&
           fp32_l2_split.deterministic_split_k->decision.single_core_n == 2816);
    PlanDecision padded_fp32_base = fp32_narrow.base->decision;
    padded_fp32_base.single_core_m = padded_fp32_base.base_m = 112;
    padded_fp32_base.single_core_n = padded_fp32_base.base_n = 128;
    padded_fp32_base.base_k = 64;
    padded_fp32_base.used_cores = 19;
    assert(std::isfinite(ReducedObjectiveUs({2048, 64, 27392, "fp32"},
        ExperimentalFp32910B3(), padded_fp32_base)));
    // Held-out K=65408 source BASE measurement is 145.464 us. The FP32
    // source L1 steps and exposed baseK-packet term must stay separate from
    // the optimized hypothetical baseK tile and from FP16/BF16 scoring.
    PlanDecision narrow_fp32_base;
    narrow_fp32_base.family = KernelFamily::Base;
    narrow_fp32_base.used_cores = 1;
    narrow_fp32_base.single_core_m = narrow_fp32_base.base_m = 16;
    narrow_fp32_base.single_core_n = narrow_fp32_base.base_n = 16;
    narrow_fp32_base.single_core_k = 65408;
    narrow_fp32_base.base_k = 32;
    CoreMemory narrow_fp32_memory;
    narrow_fp32_memory.base_l0c_buffers = 1;
    const double narrow_fp32_score = ReducedObjectiveUs(
        {16, 16, 65408, "fp32"}, ExperimentalFp32910B3(),
        narrow_fp32_base, narrow_fp32_memory, true);
    assert(narrow_fp32_score > 145.0 && narrow_fp32_score < 150.0);
    // The opt-in source-BASE MTE2 bound activates only when A exceeds L2,
    // the source cannot form a cache region, and the physical A row pitch
    // falls in a measured alignment class. Other source scores are exact.
    PlanDecision stride_base;
    stride_base.family = KernelFamily::Base;
    stride_base.used_cores = 20;
    stride_base.single_core_m = stride_base.base_m = 128;
    stride_base.single_core_n = stride_base.base_n = 256;
    stride_base.single_core_k = 32768;
    stride_base.base_k = 32;
    const auto old_fp32 = ExperimentalFp32Mode2Pipeline910B3();
    const auto mte2_fp32 = ExperimentalFp32BaseMte2910B3();
    const auto score_source_base = [&](std::uint64_t m, std::uint64_t n,
                                       std::uint64_t k, const Hardware& h) {
        auto d = stride_base;
        d.single_core_k = k;
        return ReducedObjectiveUs({m, n, k, "fp32"}, h, d,
                                  narrow_fp32_memory, true);
    };
    assert(std::abs(score_source_base(2560, 768, 32512, old_fp32) -
                    score_source_base(2560, 768, 32512, mte2_fp32)) < 1e-9);
    assert(score_source_base(2560, 768, 32768, mte2_fp32) >
           score_source_base(2560, 768, 32768, old_fp32) + 500.0);
    assert(std::abs(score_source_base(768, 2048, 32768, old_fp32) -
                    score_source_base(768, 2048, 32768, mte2_fp32)) < 1e-9);
    assert(std::abs(score_source_base(4096, 4096, 16384, old_fp32) -
                    score_source_base(4096, 4096, 16384, mte2_fp32)) < 1e-9);
    bool converted_fp32_rejected = false;
    try { (void)SolveReducedMip({16, 16, 65536, "fp32"},
                                fp16_hardware, {}, split_domain); }
    catch (const std::invalid_argument&) { converted_fp32_rejected = true; }
    assert(converted_fp32_rejected);
}
