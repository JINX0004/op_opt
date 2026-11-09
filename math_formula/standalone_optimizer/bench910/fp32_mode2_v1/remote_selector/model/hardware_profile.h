#pragma once

#include "mip_solver.h"

namespace mmopt {

// Shared measured 910B3 profile for the standalone diagnostic and host tiler.
inline Hardware Calibrated910B3() {
    return {20, 512ULL * 1024 * 1024,
            {{"launch", {"count", 2.68, 1.0}},
             {"gm_read", {"bytes", 0.109620, 3710212.0}},
             {"gm_write", {"bytes", 0.162586, 5229502.0}},
             {"cube", {"flop", 0.0, 14719432.0}},
             {"vector_add", {"ops", 0.0, 1000000.0}},
             {"nd2nz", {"elements", 0.0, 1000000.0}},
             {"sync", {"count", 0.1, 1.0}},
             {"split_partial", {"count", 1.28, 1.0}},
             {"sc_fp32_store", {"bytes", 0.2432, 34190.0}},
             {"sc_fp32_store_steady", {"bytes", 0.0, 91000.0}},
             // 49152x384x1536 source-in-context mode-5 measurements.
             {"sc_mode5_fixpipe", {"bytes", 0.0, 108000.0}},
             {"sc_mode5_iterate", {"count", 1.8, 1.0}},
             {"sc_atomic_extra", {"bytes", 0.0541, 106618.0}},
             {"sc_cast", {"elements", 0.0156, 116718.0}},
             {"sc_gm_to_l1_a", {"bytes", 0.3124, 270810.0}},
             {"sc_gm_to_l1_b", {"bytes", 0.0227, 115414.0}}}};
}

// BF16 uses the same transfer widths and FP32 accumulation as FP16. Most
// rates remain provisional. The mode-5 pair below is measured in the BF16
// source kernel at K=1536, N=384, with large and 128x128 core tiles.
inline Hardware ExperimentalBf16910B3() {
    auto hardware = Calibrated910B3();
    hardware.primitives["sc_mode5_fixpipe"] = {"bytes", 0.0, 108000.0};
    hardware.primitives["sc_mode5_iterate"] = {"count", 1.76, 1.0};
    // Paired BF16/FP16 910B3 packet probes measured equivalent 16-wide B reads.
    // Reuse the FP16 fitted narrow-row rates; wider rows keep generic costs.
    hardware.primitives["base_mte2_b_narrow_rows_one"] = {"rows", 0.0, 124.25456613790051};
    hardware.primitives["base_mte2_b_narrow_rows_full"] = {"rows", 0.0, 85.48932166157344};
    return hardware;
}

// First FP32 NN source-tile fit on CANN 9.0.0 / 910B3. The 2048x2048x4096
// BASE source tile anchors the effective Cube service; 512x512x4096 BASE
// and 128x128x32768 deterministic Split-K are held-out checks. FP32 modes
// 2 and 5 provisionally reuse the measured FP32 atomic-output service from
// mode 6; their own rates are not calibrated yet. Transfer rates are inherited from the
// two-byte profile. Keep this
// separate from the calibrated FP16/BF16 production selectors.
inline Hardware ExperimentalFp32910B3() {
    auto hardware = Calibrated910B3();
    hardware.primitives["cube"].rate_per_us = 3600000.0;
    // Controlled 16x16xK source BASE sweep, fixed 16x16x32 tile. The
    // residual after byte service and source L1-window startups grows by
    // 0.06636 us per baseK packet. This is an in-context effective service,
    // not an isolated MTE1 or call latency; broader tiles need validation.
    hardware.primitives["base_fp32_packet_issue"] = {"count", 0.06636304, 1.0};
    return hardware;
}

// Diagnostic FP32 mode-2 pipeline profile. Huawei's nominal 910B3
// L1->L0A/L0B links are 256/128 bytes per 1.8-GHz AIC cycle. They are
// throughput ceilings, not calibrated source-kernel primitives. This
// opt-in profile lets the mode-2 scorer test the additional stage and
// output-drain dependencies without changing online FP32 decisions.
inline Hardware ExperimentalFp32Mode2Pipeline910B3() {
    auto hardware = ExperimentalFp32910B3();
    hardware.primitives["sc_mode2_l0a"] = {"bytes", 0.0, 460800.0};
    hardware.primitives["sc_mode2_l0b"] = {"bytes", 0.0, 230400.0};
    return hardware;
}

// Diagnostic source-BASE transfer bound from 910B3/CANN 9.0.0 measurements.
// The rates are effective aggregate GM->L1 service under a Cube workload:
// 1.52 MB/us for a 32-KiB-aligned A pitch (K=24576/40960/57344); and 1.26 MB/us for a
// 64-KiB-aligned A pitch (K=32768/49152). The source's 20-AIC L2 grouping
// needs at least four tiles per axis and a region below its 100-MiB target.
// The regular path retains the old score because it already matches source
// measurements; this opt-in lower bound is not the online FP32 profile.
inline Hardware ExperimentalFp32BaseMte2910B3() {
    auto hardware = ExperimentalFp32Mode2Pipeline910B3();
    hardware.fp32_base_mte2_service = BaseMte2Service{
        1520000.0, 1260000.0,
        32ULL * 1024, 64ULL * 1024,
        192ULL * 1024 * 1024, 100ULL * 1024 * 1024,
        4, 4};
    // The direct-copy N=64/128 source BASE path has a second service regime
    // at unaligned A row pitches. These are 910B3/CANN 9.0.0 effective
    // primitive rates, measured with Cube work in flight. The 256-MiB A
    // footprint is the lower boundary of the validated cache regime; it is
    // not a claim about the physical L2 size.
    hardware.fp32_base_mte2_service->general_pitch =
        BaseMte2Service::GeneralPitch{
            1290000.0, 119000.0, 0.00405,
            256ULL * 1024 * 1024, 64, 128, 16ULL * 1024};
    return hardware;
}

// Device-4 FP16 ND->L1 probes identify an additional B-row cost when each
// source row is only 32 bytes. This conservative profile retains the
// established read objective for wider rows and other families.
inline Hardware NarrowB910B3() {
    auto hardware = Calibrated910B3();
    auto& p = hardware.primitives;
    p.emplace("base_mte2_b_narrow_rows_one", PrimitiveTiming{"rows", 0.0, 124.25456613790051});
    p.emplace("base_mte2_b_narrow_rows_full", PrimitiveTiming{"rows", 0.0, 85.48932166157344});
    return hardware;
}

// The broader packet fit is retained for diagnostic comparisons. Source-
// backed FP16 trials found large-shape family regressions, so do not make
// this the default online profile.
inline Hardware RowAware910B3() {
    auto hardware = NarrowB910B3();
    auto& p = hardware.primitives;
    p.emplace("base_mte2_a", PrimitiveTiming{"bytes", 0.1731666666666667, 261620.75848303392});
    p.emplace("base_mte2_b", PrimitiveTiming{"bytes", 0.11958333333333337, 235458.68263473056});
    p.emplace("base_mte2_a_rows", PrimitiveTiming{"rows", 0.0, 1.0e12});
    p.emplace("base_mte2_b_rows", PrimitiveTiming{"rows", 0.0, 1.0e12});
    p.emplace("base_mte2_a_full", PrimitiveTiming{"bytes", 0.12958333333333338, 188592.8057553957});
    p.emplace("base_mte2_b_full", PrimitiveTiming{"bytes", 0.12475000000000014, 195775.95220313675});
    return hardware;
}

inline MipTileDomain SourceCompatibleDomain() {
    MipTileDomain domain;
    domain.base_core_equals_base = true;
    domain.fp32_source_patterns = true;
    return domain;
}

}  // namespace mmopt
