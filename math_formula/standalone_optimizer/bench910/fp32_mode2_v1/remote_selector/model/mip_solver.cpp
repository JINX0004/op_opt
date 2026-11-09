#include "mip_solver.h"

#include "split_vector_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mmopt {
namespace {

constexpr double kRpcBytes = 20.0 * 1024.0 * 1024.0;

bool IsSingleCore(KernelFamily family) {
    return family == KernelFamily::SingleCoreSplitKMode2 ||
           family == KernelFamily::SingleCoreSplitKMode5 ||
           family == KernelFamily::SingleCoreSplitKMode6;
}

std::uint64_t CeilDiv(std::uint64_t x, std::uint64_t y) {
    return x / y + (x % y != 0);
}

struct Rates {
    PrimitiveTiming launch, read, write, cube, vector, sync, nd2nz;
    // Optional operand-specific BASE GM->L1 services. Without both measured
    // profiles the legacy generic read service applies to both operands.
    PrimitiveTiming read_a, read_b;
    struct BasePacketRead {
        PrimitiveTiming a_bytes, b_bytes, a_rows, b_rows;
        // Optional occupancy endpoint services and the narrow ND B-row path.
        // The original four-primitive packet profile remains valid alone.
        std::optional<PrimitiveTiming> a_bytes_full, b_bytes_full;
    };
    std::optional<BasePacketRead> base_packet_read;
    // Fixed MTE2→V / add / V→MTE2 cost of one extra Split-K partial.
    // Absent from a profile means zero; the reduction probe supplies it.
    double split_partial_us = 0.0;
    // The serial plain-store probe is used for mode 2/5 and the first
    // mode-6 K packet. The optional steady service is the *total* measured
    // Fixpipe/atomic output demand for later mode-6 K packets, so it must
    // not be combined with sc_atomic_extra.
    std::optional<PrimitiveTiming> sc_fp32_store, sc_fp32_store_steady;
    // Source-in-context mode-5 total output and per-IterateAll setup.
    std::optional<PrimitiveTiming> sc_mode5_fixpipe, sc_mode5_iterate;
    std::optional<PrimitiveTiming> sc_atomic_extra, sc_cast;
    std::optional<PrimitiveTiming> sc_gm_to_l1_a, sc_gm_to_l1_b;
    // Exposed FP32 BASE work per internal Iterate/GetTensorC call and per
    // subsequent atomic output update. Absent until independently measured.
    std::optional<PrimitiveTiming> base_fp32_call, base_fp32_atomic_output;
    // Exposed per-baseK packet service on the FP32 BASE path, measured by
    // holding the source M/N/baseK tile fixed and varying K.
    std::optional<PrimitiveTiming> base_fp32_packet_issue;
    std::optional<BaseMte2Service> base_fp32_mte2_service;
    // Can be used with either the generic read surrogate or packet profile.
    std::optional<PrimitiveTiming> b_narrow_rows_one, b_narrow_rows_full;
    double a_bytes = 2.0, b_bytes = 2.0, output_bytes = 2.0, accumulator_bytes = 4.0;
    // Optional experimental FP32 mode-2 L1->L0 services. These are kept
    // separate from the default profile until source-shaped probes validate
    // the nominal A/B rates and their overlap with GM->L1.
    std::optional<PrimitiveTiming> sc_mode2_l0a, sc_mode2_l0b;
};

PrimitiveTiming Get(const Hardware& hardware, const char* name, const char* unit) {
    const auto found = hardware.primitives.find(name);
    if (found == hardware.primitives.end())
        throw std::invalid_argument(std::string("missing primitive ") + name);
    const auto& p = found->second;
    if (p.unit != unit || !std::isfinite(p.fixed_latency_us) ||
        p.fixed_latency_us < 0 || !std::isfinite(p.rate_per_us) || p.rate_per_us <= 0)
        throw std::invalid_argument(std::string("invalid primitive ") + name);
    return p;
}

Rates ReadRates(const Hardware& hardware, const Workload& workload) {
    // Every primitive used by either family must have a finite nonnegative
    // fixed cost and a positive rate in the specified unit. Per-packet Cube
    // fixed latency is not represented by the reduced service objective.
    Rates r{Get(hardware, "launch", "count"),
            Get(hardware, "gm_read", "bytes"),
            Get(hardware, "gm_write", "bytes"),
            Get(hardware, "cube", "flop"),
            Get(hardware, "vector_add", "ops"),
            Get(hardware, "sync", "count"),
            Get(hardware, "nd2nz", "elements"), {}, {}, {},
            0.0, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
    const auto types = WorkloadTypes(workload);
    r.a_bytes = static_cast<double>(ElementBytes(types.a));
    r.b_bytes = static_cast<double>(ElementBytes(types.b));
    r.output_bytes = static_cast<double>(ElementBytes(types.c));
    r.accumulator_bytes = static_cast<double>(ElementBytes(types.accumulator));
    const bool has_a = hardware.primitives.count("gm_read_a") != 0;
    const bool has_b = hardware.primitives.count("gm_read_b") != 0;
    if (has_a != has_b)
        throw std::invalid_argument("gm_read_a and gm_read_b must be supplied together");
    r.read_a = has_a ? Get(hardware, "gm_read_a", "bytes") : r.read;
    r.read_b = has_b ? Get(hardware, "gm_read_b", "bytes") : r.read;
    constexpr const char* packet_names[] = {"base_mte2_a", "base_mte2_b",
                                            "base_mte2_a_rows", "base_mte2_b_rows"};
    int packet_count = 0;
    for (const auto* name : packet_names)
        packet_count += hardware.primitives.count(name) != 0;
    if (packet_count && packet_count != 4)
        throw std::invalid_argument("all four BASE MTE2 packet primitives are required");
    if (packet_count == 4) {
        r.base_packet_read = Rates::BasePacketRead{
            Get(hardware, packet_names[0], "bytes"),
            Get(hardware, packet_names[1], "bytes"),
            Get(hardware, packet_names[2], "rows"),
            Get(hardware, packet_names[3], "rows"), {}, {}};
        if (r.base_packet_read->a_rows.fixed_latency_us != 0 ||
            r.base_packet_read->b_rows.fixed_latency_us != 0)
            throw std::invalid_argument("BASE MTE2 row primitives must have zero fixed latency");
    }
    constexpr const char* occupancy_names[] = {
        "base_mte2_a_full", "base_mte2_b_full"};
    int occupancy_count = 0;
    for (const auto* name : occupancy_names)
        occupancy_count += hardware.primitives.count(name) != 0;
    if (occupancy_count && (occupancy_count != 2 || !r.base_packet_read))
        throw std::invalid_argument("both full-occupancy BASE MTE2 primitives are required");
    if (occupancy_count == 2) {
        auto& p = *r.base_packet_read;
        p.a_bytes_full = Get(hardware, occupancy_names[0], "bytes");
        p.b_bytes_full = Get(hardware, occupancy_names[1], "bytes");
    }
    constexpr const char* narrow_names[] = {
        "base_mte2_b_narrow_rows_one", "base_mte2_b_narrow_rows_full"};
    const int narrow_count = (hardware.primitives.count(narrow_names[0]) != 0) +
                             (hardware.primitives.count(narrow_names[1]) != 0);
    if (narrow_count && narrow_count != 2)
        throw std::invalid_argument("both narrow B-row primitives are required");
    if (narrow_count == 2) {
        r.b_narrow_rows_one = Get(hardware, narrow_names[0], "rows");
        r.b_narrow_rows_full = Get(hardware, narrow_names[1], "rows");
        if (r.b_narrow_rows_one->fixed_latency_us != 0 ||
            r.b_narrow_rows_full->fixed_latency_us != 0)
            throw std::invalid_argument("narrow B-row primitives must have zero fixed latency");
    }
    if (r.cube.fixed_latency_us != 0.0)
        throw std::invalid_argument("reduced model currently requires zero fixed Cube latency");
    const auto partial = hardware.primitives.find("split_partial");
    if (partial != hardware.primitives.end()) {
        if (partial->second.unit != "count" ||
            !std::isfinite(partial->second.fixed_latency_us) ||
            partial->second.fixed_latency_us < 0)
            throw std::invalid_argument("invalid primitive split_partial");
        r.split_partial_us = partial->second.fixed_latency_us;
    }
    if (hardware.primitives.count("sc_atomic_extra"))
        r.sc_atomic_extra = Get(hardware, "sc_atomic_extra", "bytes");
    if (hardware.primitives.count("sc_fp32_store"))
        r.sc_fp32_store = Get(hardware, "sc_fp32_store", "bytes");
    if (hardware.primitives.count("sc_fp32_store_steady"))
        r.sc_fp32_store_steady = Get(hardware, "sc_fp32_store_steady", "bytes");
    const bool mode5_fix = hardware.primitives.count("sc_mode5_fixpipe") != 0;
    const bool mode5_iterate = hardware.primitives.count("sc_mode5_iterate") != 0;
    if (mode5_fix != mode5_iterate)
        throw std::invalid_argument("sc_mode5_fixpipe and sc_mode5_iterate must be supplied together");
    if (mode5_fix) {
        r.sc_mode5_fixpipe = Get(hardware, "sc_mode5_fixpipe", "bytes");
        r.sc_mode5_iterate = Get(hardware, "sc_mode5_iterate", "count");
    }
    if (hardware.primitives.count("sc_cast"))
        r.sc_cast = Get(hardware, "sc_cast", "elements");
    const bool has_sc_a = hardware.primitives.count("sc_gm_to_l1_a") != 0;
    const bool has_sc_b = hardware.primitives.count("sc_gm_to_l1_b") != 0;
    if (has_sc_a != has_sc_b)
        throw std::invalid_argument("sc_gm_to_l1_a and sc_gm_to_l1_b must be supplied together");
    if (has_sc_a) {
        r.sc_gm_to_l1_a = Get(hardware, "sc_gm_to_l1_a", "bytes");
        r.sc_gm_to_l1_b = Get(hardware, "sc_gm_to_l1_b", "bytes");
    }
    if (hardware.primitives.count("base_fp32_call"))
        r.base_fp32_call = Get(hardware, "base_fp32_call", "count");
    if (hardware.primitives.count("base_fp32_atomic_output"))
        r.base_fp32_atomic_output = Get(hardware, "base_fp32_atomic_output", "bytes");
    if (hardware.primitives.count("base_fp32_packet_issue"))
        r.base_fp32_packet_issue = Get(hardware, "base_fp32_packet_issue", "count");
    if (hardware.fp32_base_mte2_service) {
        const auto& p = *hardware.fp32_base_mte2_service;
        if (!std::isfinite(p.aligned_rate_per_us) || p.aligned_rate_per_us <= 0 ||
            !std::isfinite(p.strongly_aligned_rate_per_us) ||
            p.strongly_aligned_rate_per_us <= 0 ||
            !p.aligned_pitch_bytes || p.aligned_pitch_bytes % 4 ||
            !p.strongly_aligned_pitch_bytes || p.strongly_aligned_pitch_bytes % 4 ||
            p.strongly_aligned_pitch_bytes % p.aligned_pitch_bytes ||
            !p.l2_bytes || !p.region_limit_bytes ||
            !p.minimum_region_m_tiles || !p.minimum_region_n_tiles)
            throw std::invalid_argument("invalid FP32 BASE MTE2 service profile");
        if (p.general_pitch) {
            const auto& q = *p.general_pitch;
            if (!std::isfinite(q.shared_bytes_per_us) || q.shared_bytes_per_us <= 0 ||
                !std::isfinite(q.core_bytes_per_us) || q.core_bytes_per_us <= 0 ||
                !std::isfinite(q.a_row_window_us) || q.a_row_window_us < 0 ||
                !q.minimum_a_footprint_bytes || !q.n_multiple ||
                q.maximum_n < q.n_multiple ||
                !q.excluded_pitch_alignment_bytes ||
                q.excluded_pitch_alignment_bytes % 4)
                throw std::invalid_argument("invalid FP32 BASE general-pitch service profile");
        }
        r.base_fp32_mte2_service = p;
    }
    const bool mode2_a = hardware.primitives.count("sc_mode2_l0a") != 0;
    const bool mode2_b = hardware.primitives.count("sc_mode2_l0b") != 0;
    if (mode2_a != mode2_b)
        throw std::invalid_argument("sc_mode2_l0a and sc_mode2_l0b must be supplied together");
    if (mode2_a) {
        r.sc_mode2_l0a = Get(hardware, "sc_mode2_l0a", "bytes");
        r.sc_mode2_l0b = Get(hardware, "sc_mode2_l0b", "bytes");
    }
    return r;
}

double ConversionUs(const Workload& w, const Rates& r) {
    if (w.k <= 65535) return 0.0;
    const double elements = static_cast<double>(w.m) * static_cast<double>(w.k);
    const double bytes = r.a_bytes * elements;
    return r.read.fixed_latency_us + bytes / r.read.rate_per_us +
           r.nd2nz.fixed_latency_us + elements / r.nd2nz.rate_per_us +
           r.write.fixed_latency_us + bytes / r.write.rate_per_us +
           r.sync.fixed_latency_us;
}

double CommonUs(const Workload& w, const Rates& r) {
    return r.launch.fixed_latency_us + ConversionUs(w, r);
}

double OutputElements(const Workload& w) {
    return static_cast<double>(w.m) * static_cast<double>(w.n);
}

std::uint64_t Bound(double numerator, double denominator) {
    if (denominator <= 0 || !std::isfinite(denominator)) return 0;
    const double quotient = std::floor(numerator / denominator);
    if (!std::isfinite(quotient) || quotient < 0) return 0;
    return static_cast<std::uint64_t>(std::min(quotient, 1000000.0));
}

struct Geometry {
    // One discrete M/N choice g. Counts include output and base-tile tails.
    // Workload-derived counts are precomputed for each geometry.
    std::uint64_t core_m, core_n, base_m, base_n;
    std::uint64_t output_tiles, spatial_packets;
    double operand_bytes, a_bytes, b_bytes;
};

struct BaseCoreLoad {
    double elements = 0.0;
    double spatial_packets = 0.0;
    double output_tiles = 0.0;
    // Sum of actual A-M and B-N extents across spatial base packets.
    // These keep tail transfer volumes exact without expanding each tile.
    double a_axis_sum = 0.0;
    double b_axis_sum = 0.0;
};

std::vector<BaseCoreLoad> BaseCoreLoads(const Workload& w,
                                        std::uint64_t core_m,
                                        std::uint64_t core_n,
                                        std::uint64_t base_m,
                                        std::uint64_t base_n,
                                        std::uint64_t cores) {
    // Abstract the source's rotation of surplus tile work without replaying
    // its L2 regions or per-tile block indices. Each of the at most four
    // full/tail tile classes gives every core floor(count/cores) tiles and
    // gives one extra tile to count%cores consecutive cores. The extra-tile
    // window rotates between classes. This is a balanced-work assumption,
    // NOT an exact MatMulV3 assignment or a fitted hardware correction.
    // Work is O(cores), independent of the number of output tiles.
    const auto full_m = w.m / core_m, tail_m = w.m % core_m;
    const auto full_n = w.n / core_n, tail_n = w.n % core_n;
    std::vector<BaseCoreLoad> loads(cores);
    std::uint64_t phase = 0;
    const auto add_class = [&](std::uint64_t count, std::uint64_t m_extent,
                               std::uint64_t n_extent) {
        if (!count) return;
        const auto each = count / cores;
        const auto extra = count % cores;
        const auto m_packets = CeilDiv(m_extent, base_m);
        const auto n_packets = CeilDiv(n_extent, base_n);
        const double spatial = static_cast<double>(m_packets) * n_packets;
        for (std::uint64_t core = 0; core < cores; ++core) {
            const auto relative = (core + cores - phase) % cores;
            const double assigned = static_cast<double>(each + (relative < extra));
            auto& load = loads[core];
            load.output_tiles += assigned;
            load.spatial_packets += assigned * spatial;
            load.a_axis_sum += assigned * spatial * m_extent;
            load.b_axis_sum += assigned * spatial * n_extent;
            load.elements += assigned * m_extent * n_extent;
        }
        phase = (phase + extra) % cores;
    };
    add_class(full_m * full_n, core_m, core_n);
    if (tail_n) add_class(full_m, core_m, tail_n);
    if (tail_m) add_class(full_n, tail_m, core_n);
    if (tail_m && tail_n) add_class(1, tail_m, tail_n);
    return loads;
}

struct BaseL1Steps { std::uint64_t a, b; };

BaseL1Steps SourceBaseL1Steps(const Workload& w, const Geometry& g,
                             std::uint64_t base_k, const CoreMemory& mem) {
    // Plain BASE supportL0c2out branch of CalL1Tiling() in
    // matmul_v3_base_tiling.cpp. The source adds back its 256-byte RPC
    // reservation before splitting L1 between A and B double buffers.
    const auto l1 = mem.l1_bytes + 256;
    const auto types = WorkloadTypes(w);
    const auto a_width = ElementBytes(types.a);
    const auto b_width = ElementBytes(types.b);
    const auto packet_a = a_width * g.base_m * base_k;
    const auto packet_b = b_width * g.base_n * base_k;
    auto depth_a = l1 / 2 / packet_a;
    auto depth_b = l1 / 2 / packet_b;
    if (depth_a * packet_a + depth_b * packet_b > l1) {
        if (g.base_m <= g.base_n) depth_a /= 2;
        else depth_b /= 2;
    }
    auto step_a = depth_a / 2;
    auto step_b = depth_b / 2;
    if (!step_a || !step_b)
        throw std::invalid_argument("no source BASE L1 step fits");
    const auto align = [&](std::uint64_t& step, std::uint64_t element_width) {
        if (step * base_k >= w.k) return;
        const auto span_bytes = step * base_k * element_width;
        const auto quantum = span_bytes > 512 ? 512ULL :
                             span_bytes > 256 ? 256ULL : 0ULL;
        if (quantum && quantum % (base_k * element_width) == 0)
            while (step > 1 && (step * base_k * element_width) % quantum) --step;
    };
    align(step_a, a_width);
    align(step_b, b_width);
    if (step_a >= step_b) step_a = step_a / step_b * step_b;
    else step_b = step_b / step_a * step_a;
    return {step_a, step_b};
}

double BaseStageUs(const Workload& w, const Rates& r,
                   const Geometry& g, std::uint64_t base_k,
                   std::uint64_t cores, const CoreMemory& mem,
                   std::uint64_t max_cores,
                   const std::vector<BaseCoreLoad>* cached_loads = nullptr,
                   bool source_emitted_base = false) {
    const auto computed_loads = cached_loads ? std::vector<BaseCoreLoad>{} :
        BaseCoreLoads(w, g.core_m, g.core_n, g.base_m, g.base_n, cores);
    const auto& loads = cached_loads ? *cached_loads : computed_loads;
    const double occupancy = max_cores > 1 ?
        static_cast<double>(cores - 1) / (max_cores - 1) : 0.0;
    double narrow_b_row_us = 0.0;
    if (r.b_bytes * g.base_n < 64 && r.b_narrow_rows_one) {
        const double one = 1.0 / r.b_narrow_rows_one->rate_per_us;
        const double full = 1.0 / r.b_narrow_rows_full->rate_per_us;
        narrow_b_row_us = one + occupancy * occupancy * (full - one);
    }
    if (r.base_packet_read) {
        // One packet is the source's entire L1 K step, not a baseK slice.
        // Fit: fixed packet service + requested bytes + ND source rows.
        // The probes saturate 20 AICs, so this is an exploratory per-core
        // service objective; Cube/MTE2 overlap is approximated by max.
        const auto steps = SourceBaseL1Steps(w, g, base_k, mem);
        const auto& p = *r.base_packet_read;
        const auto fixed = [&](const PrimitiveTiming& one,
                               const std::optional<PrimitiveTiming>& full) {
            return one.fixed_latency_us +
                (full ? occupancy * (full->fixed_latency_us - one.fixed_latency_us) : 0.0);
        };
        const auto inverse_rate = [&](const PrimitiveTiming& one,
                                      const std::optional<PrimitiveTiming>& full) {
            return 1.0 / one.rate_per_us +
                (full ? occupancy * (1.0 / full->rate_per_us - 1.0 / one.rate_per_us) : 0.0);
        };
        const double count_a = static_cast<double>(CeilDiv(w.k, steps.a * base_k));
        const double count_b = static_cast<double>(CeilDiv(w.k, steps.b * base_k));
        const double shared_write = r.output_bytes * OutputElements(w) /
                                    r.write.rate_per_us;
        double critical = 0.0;
        for (const auto& load : loads) {
            const double cube = 2.0 * load.elements * w.k / r.cube.rate_per_us;
            const double read =
                load.spatial_packets * (count_a * fixed(p.a_bytes, p.a_bytes_full) +
                                        count_b * fixed(p.b_bytes, p.b_bytes_full)) +
                r.a_bytes * w.k * load.a_axis_sum * inverse_rate(p.a_bytes, p.a_bytes_full) +
                r.b_bytes * w.k * load.b_axis_sum * inverse_rate(p.b_bytes, p.b_bytes_full) +
                count_a * load.a_axis_sum / p.a_rows.rate_per_us +
                w.k * load.spatial_packets *
                    (1.0 / p.b_rows.rate_per_us + narrow_b_row_us);
            critical = std::max(critical, std::max(
                cube, read + shared_write +
                      load.output_tiles * r.write.fixed_latency_us));
        }
        return critical;
    }
    // Source FP32 BASE and measured narrow-B BF16 charge one startup per
    // actual L1 K window. Other two-byte paths retain their prior surrogate.
    double a_startups = static_cast<double>(CeilDiv(w.k, base_k));
    double b_startups = a_startups;
    if ((w.dtype == "fp32" && source_emitted_base) ||
        (w.dtype == "bf16" && narrow_b_row_us > 0.0)) {
        const auto steps = SourceBaseL1Steps(w, g, base_k, mem);
        a_startups = static_cast<double>(CeilDiv(w.k, steps.a * base_k));
        b_startups = static_cast<double>(CeilDiv(w.k, steps.b * base_k));
    }
    const auto fp32_calls = w.dtype == "fp32" ? CeilDiv(w.k, 8192) : 1;
    double cube_critical = 0.0, packet_critical = 0.0;
    double call_critical = 0.0, atomic_critical = 0.0;
    for (const auto& load : loads) {
        cube_critical = std::max(cube_critical,
            2.0 * load.elements * static_cast<double>(w.k) /
                r.cube.rate_per_us);
        packet_critical = std::max(packet_critical,
            load.spatial_packets *
                (a_startups * r.read_a.fixed_latency_us +
                 b_startups * r.read_b.fixed_latency_us) +
                load.output_tiles * r.write.fixed_latency_us +
                w.k * load.spatial_packets * narrow_b_row_us +
                (source_emitted_base && r.base_fp32_packet_issue ?
                    load.spatial_packets * CeilDiv(w.k, base_k) *
                        r.base_fp32_packet_issue->fixed_latency_us : 0.0));
        if (source_emitted_base && r.base_fp32_call)
            call_critical = std::max(call_critical,
                load.output_tiles * fp32_calls *
                    r.base_fp32_call->fixed_latency_us);
        if (source_emitted_base && r.base_fp32_atomic_output && fp32_calls > 1)
            atomic_critical = std::max(atomic_critical,
                (fp32_calls - 1) *
                    (load.output_tiles * r.base_fp32_atomic_output->fixed_latency_us +
                     r.accumulator_bytes * load.elements /
                         r.base_fp32_atomic_output->rate_per_us));
    }
    const double shared_bytes = g.a_bytes / r.read_a.rate_per_us +
        g.b_bytes / r.read_b.rate_per_us +
        r.output_bytes * OutputElements(w) / r.write.rate_per_us;
    double source_mte2_bound = 0.0;
    if (w.dtype == "fp32" &&
        r.a_bytes == 4.0 && r.b_bytes == 4.0 &&
        g.core_m == g.base_m && g.core_n == g.base_n &&
        r.base_fp32_mte2_service) {
        // The source requests A once per N tile and B once per M tile. The
        // L2-hot generic GM rate misses a lower-rate, pitch-sensitive MTE2
        // regime for large A matrices when CalcTile has no cache region.
        // Keep this optional until the access classes have held-out support.
        const auto& p = *r.base_fp32_mte2_service;
        const double a_footprint = r.a_bytes * w.m * static_cast<double>(w.k);
        const double minimum_region_bytes = static_cast<double>(w.k) *
            (r.a_bytes * p.minimum_region_m_tiles * g.base_m +
             r.b_bytes * p.minimum_region_n_tiles * g.base_n) +
            r.output_bytes * p.minimum_region_m_tiles * g.base_m *
            p.minimum_region_n_tiles * g.base_n;
        const bool source_region_possible =
            CeilDiv(w.m, g.base_m) >= p.minimum_region_m_tiles &&
            CeilDiv(w.n, g.base_n) >= p.minimum_region_n_tiles &&
            minimum_region_bytes <= p.region_limit_bytes;
        double rate = 0.0;
        if (a_footprint > p.l2_bytes && !source_region_possible) {
            if (w.k % (p.strongly_aligned_pitch_bytes / 4) == 0)
                rate = p.strongly_aligned_rate_per_us;
            else if (w.k % (p.aligned_pitch_bytes / 4) == 0)
                rate = p.aligned_rate_per_us;
        }
        if (rate > 0.0) source_mte2_bound = (g.a_bytes + g.b_bytes) / rate;
        if (p.general_pitch &&
            a_footprint > p.general_pitch->minimum_a_footprint_bytes &&
            w.n % p.general_pitch->n_multiple == 0 &&
            w.n <= p.general_pitch->maximum_n &&
            w.k % (p.general_pitch->excluded_pitch_alignment_bytes / 4) != 0 &&
            !source_region_possible) {
            // The aggregate requested bytes constrain the shared MTE2 path.
            // Independently, a core must serve its own bytes and A row
            // windows. SourceBaseL1Steps gives the A-copy window granularity.
            const auto& q = *p.general_pitch;
            const auto steps = SourceBaseL1Steps(w, g, base_k, mem);
            const double a_windows = static_cast<double>(
                CeilDiv(w.k, steps.a * base_k));
            double core_service = 0.0;
            for (const auto& load : loads) {
                const double core_bytes = w.k *
                    (r.a_bytes * load.a_axis_sum +
                     r.b_bytes * load.b_axis_sum);
                core_service = std::max(core_service,
                    core_bytes / q.core_bytes_per_us +
                    a_windows * load.a_axis_sum * q.a_row_window_us);
            }
            source_mte2_bound = std::max({source_mte2_bound,
                (g.a_bytes + g.b_bytes) / q.shared_bytes_per_us,
                core_service});
        }
    }
    return std::max({cube_critical, shared_bytes + packet_critical,
                     source_mte2_bound}) +
           call_critical + atomic_critical;
}

std::uint64_t AxisPackets(std::uint64_t shape, std::uint64_t core,
                          std::uint64_t base);

double SplitStageUs(const Workload& w, const Rates& r,
                    const Geometry& g, std::uint64_t base_k,
                    std::uint64_t step,
                    std::uint64_t cores) {
    // Panel pipeline from SOLVE_SPLIT_FORMULATION.md. The source NK path
    // (M > N) ping-pongs N panels; otherwise it ping-pongs M panels. Each
    // panel repeats the same K split. Operand reads overlap that panel's
    // Cube work. The vector chain then adds one accumulated partial per AIC.
    const auto q = CeilDiv(w.k, base_k);
    const auto p = CeilDiv(q, step);
    const auto rounds = CeilDiv(p, cores);
    const auto q_crit = std::min(q, rounds * step);
    const auto k_crit = std::min(w.k, q_crit * base_k);
    const bool n_outer = w.m > w.n;
    const auto outer_shape = n_outer ? w.n : w.m;
    const auto outer_core = n_outer ? g.core_n : g.core_m;
    std::vector<std::uint64_t> extents;
    if (outer_core >= outer_shape) {
        extents.push_back(outer_shape);
    } else {
        extents.assign(outer_shape / outer_core, outer_core);
        if (const auto tail = outer_shape % outer_core) extents.push_back(tail);
    }

    std::vector<double> cube_stages;
    std::vector<double> vector_stages;
    cube_stages.reserve(extents.size());
    vector_stages.reserve(extents.size());
    for (const auto extent : extents) {
        const auto m_ext = n_outer ? w.m : extent;
        const auto n_ext = n_outer ? extent : w.n;
        const auto m_packets = n_outer ? AxisPackets(w.m, g.core_m, g.base_m)
                                       : CeilDiv(m_ext, g.base_m);
        const auto n_packets = n_outer ? CeilDiv(n_ext, g.base_n)
                                       : AxisPackets(w.n, g.core_n, g.base_n);
        const double spatial = static_cast<double>(m_packets) * n_packets;
        const double elements = static_cast<double>(m_ext) * n_ext;
        const double operand_bytes = static_cast<double>(w.k) *
            (r.a_bytes * static_cast<double>(m_ext) * n_packets +
             r.b_bytes * static_cast<double>(n_ext) * m_packets);
        const double cube = 2.0 * elements * k_crit / r.cube.rate_per_us;
        // The measured converted-A path groups baseK slices into the
        // source's L1 K step. Retain the established per-baseK surrogate
        // below the conversion threshold: grouping that path regressed
        // held-out family decisions.
        const auto read_startups = w.k > 65535 ? CeilDiv(q_crit, step) : q_crit;
        const double operand_gm = operand_bytes / r.read.rate_per_us +
            2.0 * spatial * static_cast<double>(read_startups) *
                r.read.fixed_latency_us;
        const double shard = static_cast<double>(
            SplitVectorMaxElements(m_ext * n_ext, cores));
        cube_stages.push_back(std::max(cube, operand_gm));
        vector_stages.push_back(
            r.sync.fixed_latency_us +
            static_cast<double>(cores - 1) *
                (r.split_partial_us + shard / r.vector.rate_per_us) +
            r.write.fixed_latency_us);
    }
    double stage = cube_stages.front();
    for (std::size_t i = 1; i < cube_stages.size(); ++i)
        stage += std::max(cube_stages[i], vector_stages[i - 1]);
    return stage + vector_stages.back();
}

// A panel transforms Cube and AIV completion times as
// C'=C+c, V'=max(C+a,V+b). Composition and binary powers let the source's
// contiguous spatial assignment be scored without expanding large M/N grids.
struct ScPipe {
    double c = 0.0;
    double a = -std::numeric_limits<double>::infinity();
    double b = 0.0;
};

ScPipe Then(const ScPipe& first, const ScPipe& second) {
    return {first.c + second.c,
            std::max(first.a + second.b, first.c + second.a),
            first.b + second.b};
}

ScPipe Power(ScPipe unit, std::uint64_t count) {
    ScPipe result;
    while (count) {
        if (count & 1) result = Then(result, unit);
        count >>= 1;
        if (count) unit = Then(unit, unit);
    }
    return result;
}

double ScAtomicExtraUs(const Rates& r, double packets, double bytes) {
    if (!r.sc_atomic_extra) return 0.0;
    return packets * r.sc_atomic_extra->fixed_latency_us +
           bytes / r.sc_atomic_extra->rate_per_us;
}

double ScFp32StoreUs(const Rates& r, double packets, double bytes) {
    const auto& p = r.sc_fp32_store ? *r.sc_fp32_store : r.write;
    return packets * p.fixed_latency_us + bytes / p.rate_per_us;
}

double ScMeasuredAtomicOutputUs(const Rates& r, double packets, double bytes) {
    // A source-in-context Fixpipe counter supplies total atomic output
    // demand for mode 6 and is provisionally reused for direct FP32
    // modes 2 and 5.
    // The standalone serial store probe spends much longer in Fixpipe than
    // the source kernel at the same geometry. Without an in-context profile,
    // retain the exploratory generic-write + atomic proxy.
    if (r.sc_fp32_store_steady) {
        const auto& p = *r.sc_fp32_store_steady;
        return packets * p.fixed_latency_us + bytes / p.rate_per_us;
    }
    return packets * r.write.fixed_latency_us + bytes / r.write.rate_per_us +
           ScAtomicExtraUs(r, packets, bytes);
}

double ScSliceUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                 std::uint64_t k, std::uint64_t base_k, bool atomic,
                 bool fp32_direct) {
    const double mb = static_cast<double>(CeilDiv(m, 128));
    const double nb = static_cast<double>(CeilDiv(n, 128));
    const double qk = static_cast<double>(CeilDiv(k, base_k));
    const double cube = 2.0 * m * static_cast<double>(n) * k / r.cube.rate_per_us;
    const double input = static_cast<double>(k) *
        (r.a_bytes * m * nb + r.b_bytes * n * mb);
    const double output_bytes = r.accumulator_bytes * m * n;
    const double output = fp32_direct && atomic ?
        ScMeasuredAtomicOutputUs(r, mb * nb, output_bytes) :
        ScFp32StoreUs(r, mb * nb, output_bytes) +
        (atomic ? ScAtomicExtraUs(r, mb * nb, output_bytes) : 0.0);
    const double memory = input / r.read.rate_per_us +
        2.0 * mb * nb * qk * r.read.fixed_latency_us + output;
    return std::max(cube, memory);
}

double ScMode2PipelineSliceUs(const Rates& r, std::uint64_t m,
                              std::uint64_t n, std::uint64_t k,
                              std::uint64_t base_k, bool atomic) {
    // One IterateAll has a GM->L1 path, then L1->L0 A and B loads, Cube,
    // and Fixpipe. The two operand loads are serialized on the conservative
    // source-level path; Cube and the input path may overlap across base
    // output pieces. The last full output piece must still drain through
    // Fixpipe after the final input/Cube wave. This is an experimental
    // max-plus reduction, not a claim about the Matmul engine's hidden
    // base-piece issue order. It is enabled only by the optional A/B rates.
    const double mb = static_cast<double>(CeilDiv(m, 128));
    const double nb = static_cast<double>(CeilDiv(n, 128));
    const double qk = static_cast<double>(CeilDiv(k, base_k));
    const double a_bytes = r.a_bytes * static_cast<double>(k) * m * nb;
    const double b_bytes = r.b_bytes * static_cast<double>(k) * n * mb;
    const double gm_to_l1 = (a_bytes + b_bytes) / r.read.rate_per_us +
        2.0 * mb * nb * qk * r.read.fixed_latency_us;
    const double l1_to_l0 =
        a_bytes / r.sc_mode2_l0a->rate_per_us +
        b_bytes / r.sc_mode2_l0b->rate_per_us +
        mb * nb * qk * (r.sc_mode2_l0a->fixed_latency_us +
                        r.sc_mode2_l0b->fixed_latency_us);
    const double mac = 2.0 * m * static_cast<double>(n) * k /
        r.cube.rate_per_us;
    const double output = atomic ?
        ScMeasuredAtomicOutputUs(r, mb * nb, r.accumulator_bytes * m * n) :
        ScFp32StoreUs(r, mb * nb, r.accumulator_bytes * m * n);
    const double one_piece_bytes = r.accumulator_bytes *
        std::min<std::uint64_t>(m, 128) * std::min<std::uint64_t>(n, 128);
    const double one_piece = atomic ?
        ScMeasuredAtomicOutputUs(r, 1.0, one_piece_bytes) :
        ScFp32StoreUs(r, 1.0, one_piece_bytes);
    return std::max({mac + one_piece,
                     gm_to_l1 + l1_to_l0 + one_piece,
                     output});
}

double ScMode5SliceUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                      std::uint64_t k) {
    // Source-in-context Fixpipe demand includes repeated FP32 updates.
    // MTE2, MAC and Fixpipe overlap; each IterateAll exposes setup work.
    const double mb = static_cast<double>(CeilDiv(m, 128));
    const double nb = static_cast<double>(CeilDiv(n, 128));
    const double qk = static_cast<double>(CeilDiv(k, 128));
    const double cube = 2.0 * m * static_cast<double>(n) * k /
        r.cube.rate_per_us;
    const double input = static_cast<double>(k) *
        (r.a_bytes * m * nb + r.b_bytes * n * mb);
    const double read = input / r.read.rate_per_us +
        2.0 * mb * nb * qk * r.read.fixed_latency_us;
    const double fixpipe = r.sc_mode5_fixpipe->fixed_latency_us * mb * nb +
        r.accumulator_bytes * m * static_cast<double>(n) /
            r.sc_mode5_fixpipe->rate_per_us;
    return std::max({cube, read, fixpipe}) +
        r.sc_mode5_iterate->fixed_latency_us;
}

double ScMode6ReadUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                     std::uint64_t k) {
    if (r.sc_gm_to_l1_a) {
        return r.a_bytes * static_cast<double>(k) * m /
                   r.sc_gm_to_l1_a->rate_per_us +
               CeilDiv(m, 128) * r.sc_gm_to_l1_a->fixed_latency_us +
               r.b_bytes * static_cast<double>(k) * n /
                   r.sc_gm_to_l1_b->rate_per_us +
               CeilDiv(n, 128) * r.sc_gm_to_l1_b->fixed_latency_us;
    }
    return static_cast<double>(k) * (r.a_bytes * m + r.b_bytes * n) / r.read.rate_per_us +
        (CeilDiv(m, 128) + CeilDiv(n, 128)) * r.read.fixed_latency_us;
}

double ScMode6MacUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                    std::uint64_t k, bool atomic) {
    const double cube = 2.0 * m * static_cast<double>(n) * k / r.cube.rate_per_us;
    const auto store_service = atomic ? ScMeasuredAtomicOutputUs : ScFp32StoreUs;
    const double store = store_service(r,
        CeilDiv(m, 128) * CeilDiv(n, 128), r.accumulator_bytes * m * n);
    const double max_m = std::min<std::uint64_t>(m, 128);
    const double max_n = std::min<std::uint64_t>(n, 128);
    const double one_piece = 2.0 * max_m * max_n * k / r.cube.rate_per_us +
        store_service(r, 1.0, r.accumulator_bytes * max_m * max_n);
    return std::max({cube, store, one_piece});
}

double ScKChainUs(const Workload& w, const Rates& r, const PlanDecision& d,
                  std::uint64_t m, std::uint64_t n) {
    const auto packets = CeilDiv(w.k, d.single_core_k);
    const auto tail = w.k - (packets - 1) * d.single_core_k;
    if ((w.dtype == "fp16" || w.dtype == "bf16") &&
        d.family == KernelFamily::SingleCoreSplitKMode5 &&
        r.sc_mode5_fixpipe) {
        const auto full = ScMode5SliceUs(r, m, n, d.single_core_k);
        return (packets - 1) * full + ScMode5SliceUs(r, m, n, tail);
    }
    if (d.family != KernelFamily::SingleCoreSplitKMode6) {
        const bool fp32_direct = w.dtype == "fp32";
        if (fp32_direct && d.family == KernelFamily::SingleCoreSplitKMode2 &&
            r.sc_mode2_l0a) {
            const auto slice = [&](std::uint64_t k, bool atomic) {
                return ScMode2PipelineSliceUs(r, m, n, k, d.base_k, atomic);
            };
            if (packets == 1) return slice(tail, false);
            return slice(d.single_core_k, false) +
                   (packets - 2) * slice(d.single_core_k, true) +
                   slice(tail, true);
        }
        if (packets == 1)
            return ScSliceUs(r, m, n, tail, d.base_k, false, fp32_direct);
        return ScSliceUs(r, m, n, d.single_core_k, d.base_k, false, fp32_direct) +
               (packets - 2) * ScSliceUs(r, m, n, d.single_core_k, d.base_k,
                                        true, fp32_direct) +
               ScSliceUs(r, m, n, tail, d.base_k, true, fp32_direct);
    }
    const auto full_k = d.single_core_k;
    const auto read_full = ScMode6ReadUs(r, m, n, full_k);
    const auto mac_plain = ScMode6MacUs(r, m, n, full_k, false);
    if (packets == 1) return ScMode6ReadUs(r, m, n, tail) +
                             ScMode6MacUs(r, m, n, tail, false);
    const auto mac_atomic = ScMode6MacUs(r, m, n, full_k, true);
    const auto n_tail = n - (CeilDiv(n, 128) - 1) * 128;
    const double overlap = 2.0 * m * static_cast<double>(n_tail) * full_k /
                           r.cube.rate_per_us;
    const auto read_tail = ScMode6ReadUs(r, m, n, tail);
    return read_full + mac_plain + (packets - 2) *
        (mac_atomic + std::max(0.0, read_full - overlap)) +
        std::max(0.0, read_tail - overlap) +
        ScMode6MacUs(r, m, n, tail, true);
}

ScPipe ScPanel(const Workload& w, const Rates& r, const PlanDecision& d,
               std::uint64_t m, std::uint64_t n) {
    const double cube = ScKChainUs(w, r, d, m, n);
    // FP32 mode 2 writes directly from AIC with IterateAll/atomic updates;
    // unlike FP16/BF16 mode 2, it does not run an AIV cast/output stage.
    if (w.dtype == "fp32") return {cube, cube, 0.0};
    const double shard = static_cast<double>(CeilDiv(m, 2)) * n;
    const double vector = r.sync.fixed_latency_us +
        r.accumulator_bytes * shard / r.read.rate_per_us +
        r.output_bytes * shard / r.write.rate_per_us + r.write.fixed_latency_us +
        (r.sc_cast ? r.sc_cast->fixed_latency_us +
                         shard / r.sc_cast->rate_per_us : 0.0);
    return {cube, cube + vector, vector};
}

ScPipe ScTile(const Workload& w, const Rates& r, const PlanDecision& d,
              std::uint64_t m, std::uint64_t n) {
    if (d.family == KernelFamily::SingleCoreSplitKMode5)
        return ScPanel(w, r, d, m, n);
    const auto panel_m = 128 * d.step_m;
    const auto count = CeilDiv(m, panel_m);
    const auto tail = m - (count - 1) * panel_m;
    return Then(Power(ScPanel(w, r, d, panel_m, n), count - 1),
                ScPanel(w, r, d, tail, n));
}

double ScStageUs(const Workload& w, const Rates& r, const PlanDecision& d) {
    const auto n_m = CeilDiv(w.m, d.single_core_m);
    const auto n_n = CeilDiv(w.n, d.single_core_n);
    const auto tiles = n_m * n_n;
    const auto rounds = CeilDiv(tiles, d.used_cores);
    const auto pre = tiles % d.used_cores ? tiles % d.used_cores : d.used_cores;
    const auto m_tail = w.m % d.single_core_m;
    const auto n_tail = w.n % d.single_core_n;
    const auto full_m = w.m / d.single_core_m;
    const auto full_n = w.n / d.single_core_n;
    const auto full = ScTile(w, r, d, d.single_core_m, d.single_core_n);
    const auto m_edge = m_tail ? ScTile(w, r, d, m_tail, d.single_core_n) : ScPipe{};
    const auto n_edge = n_tail ? ScTile(w, r, d, d.single_core_m, n_tail) : ScPipe{};
    const auto corner = m_tail && n_tail ? ScTile(w, r, d, m_tail, n_tail) : ScPipe{};
    const auto full_row = Then(Power(full, full_m), m_tail ? m_edge : ScPipe{});
    const auto tail_row = Then(Power(n_edge, full_m), m_tail ? corner : ScPipe{});
    double critical = 0.0;
    for (std::uint64_t q = 0; q < d.used_cores; ++q) {
        std::uint64_t index = q < pre ? q * rounds :
                              q * (rounds - 1) + pre;
        std::uint64_t remaining = q < pre ? rounds : rounds - 1;
        ScPipe sequence;
        const auto append_partial_row = [&](std::uint64_t row,
                                            std::uint64_t column,
                                            std::uint64_t take) {
            const bool n_is_tail = n_tail && row == full_n;
            const auto& normal = n_is_tail ? n_edge : full;
            const auto& last_m = n_is_tail ? corner : m_edge;
            const auto normal_count = column < full_m ?
                std::min(take, full_m - column) : 0;
            sequence = Then(sequence, Power(normal, normal_count));
            if (take > normal_count) sequence = Then(sequence, last_m);
        };
        while (remaining) {
            const auto row = index / n_m;
            const auto column = index % n_m;
            if (column || remaining < n_m) {
                const auto take = std::min(remaining, n_m - column);
                append_partial_row(row, column, take);
                index += take;
                remaining -= take;
                continue;
            }
            // Whole identical N rows compose in logarithmic time. Only the
            // final N row can have a different width.
            const auto rows = remaining / n_m;
            const auto normal_rows = row < full_n ?
                std::min(rows, full_n - row) : 0;
            sequence = Then(sequence, Power(full_row, normal_rows));
            index += normal_rows * n_m;
            remaining -= normal_rows * n_m;
            if (normal_rows == rows) continue;
            sequence = Then(sequence, tail_row);
            index += n_m;
            remaining -= n_m;
        }
        critical = std::max(critical, std::max(sequence.a, sequence.b));
    }
    return critical;
}

std::vector<std::uint64_t> TileValues(const std::vector<std::uint64_t>& supplied,
                                      std::uint64_t shape, bool core_tile,
                                      std::uint64_t aic_count,
                                      bool include_split_384) {
    std::vector<std::uint64_t> values = supplied;
    if (values.empty()) {
        // L0 base tiles remain small. singleCoreM/N are outer work extents:
        // matmul_v3_base_tiling.cpp can use the whole M or N axis, or an
        // aligned ceil(shape / core_partition) tile. Include factor pairs
        // useful for distributing work over 910B3 AICs without coupling the
        // single-core extent to the L0 base tile size.
        for (const std::uint64_t v : {16ULL, 32ULL, 64ULL, 128ULL, 256ULL})
            if (v <= shape) values.push_back(v);
        if (core_tile) {
            // The source 3x3 deterministic Split-K pattern can emit a
            // 384-row/column core tile even when that output axis is 128
            // or 256. Only Split-K may use this oversized geometry.
            if (include_split_384 && shape >= 128 && shape < 384)
                values.push_back(384);
            for (const std::uint64_t v : {512ULL, 1024ULL, 2048ULL})
                if (v <= shape) values.push_back(v);
            values.push_back(shape);
            const std::vector<std::uint64_t> partition_counts =
                {1, 2, 4, 5, 10, aic_count};
            for (const auto partitions : partition_counts) {
                if (!partitions || partitions > aic_count) continue;
                const auto extent = 16 * CeilDiv(CeilDiv(shape, partitions), 16);
                values.push_back(std::min(extent, shape));
            }
        } else if (shape <= 256) {
            values.push_back(shape);
        }
    }
    for (const auto v : values)
        if (!v || v % 16 || (v > shape && (!core_tile || v != 384 ||
                                         shape < 128)))
            throw std::invalid_argument("tile domain requires aligned values within shape, except source Split-K 384");
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::uint64_t AxisPackets(std::uint64_t shape, std::uint64_t core,
                          std::uint64_t base) {
    // Sum ceil(output_extent/base) over every output tile on one axis.
    // The final, possibly shorter output tile is counted separately.
    const auto full = shape / core;
    const auto tail = shape % core;
    return full * CeilDiv(core, base) + (tail ? CeilDiv(tail, base) : 0);
}

std::vector<Geometry> Geometries(const Workload& w, const Hardware& h,
                                 const MipTileDomain& domain) {
    const auto types = WorkloadTypes(w);
    const double a_element_bytes = ElementBytes(types.a);
    const double b_element_bytes = ElementBytes(types.b);
    const auto cm = TileValues(domain.single_core_m, w.m, true, h.aic_count,
                               w.k > 65535);
    const auto cn = TileValues(domain.single_core_n, w.n, true, h.aic_count,
                               w.k > 65535);
    const auto bm = TileValues(domain.base_m, w.m, false, h.aic_count, false);
    const auto bn = TileValues(domain.base_n, w.n, false, h.aic_count, false);
    std::vector<Geometry> result;
    // Source deterministic Split-K may use an outer 384 tile above a small
    // output axis; BASE filters that geometry before selection. Other
    // capacity tests discard infeasible geometries in each family.
    for (const auto m : cm) for (const auto n : cn)
        for (const auto a : bm) for (const auto b : bn) {
            if (a > m || b > n) continue;
            const auto pm = AxisPackets(w.m, m, a);
            const auto pn = AxisPackets(w.n, n, b);
            const auto outputs = CeilDiv(w.m, m) * CeilDiv(w.n, n);
            const auto packets = pm * pn;
            const double a_bytes = a_element_bytes * static_cast<double>(w.k) * w.m * pn;
            const double b_bytes = b_element_bytes * static_cast<double>(w.k) * w.n * pm;
            result.push_back({m, n, a, b, outputs, packets,
                              a_bytes + b_bytes, a_bytes, b_bytes});
            if (result.size() > 4096)
                throw std::invalid_argument("tile domain exceeds 4096 spatial geometries");
        }
    return result;
}

Geometry SourceGeometry(const Workload& w, std::uint64_t core_m,
                        std::uint64_t core_n, std::uint64_t base_m,
                        std::uint64_t base_n) {
    const auto types = WorkloadTypes(w);
    const auto pm = AxisPackets(w.m, core_m, base_m);
    const auto pn = AxisPackets(w.n, core_n, base_n);
    const double a_bytes = ElementBytes(types.a) * static_cast<double>(w.k) * w.m * pn;
    const double b_bytes = ElementBytes(types.b) * static_cast<double>(w.k) * w.n * pm;
    return {core_m, core_n, base_m, base_n,
            CeilDiv(w.m, core_m) * CeilDiv(w.n, core_n), pm * pn,
            a_bytes + b_bytes, a_bytes, b_bytes};
}

std::vector<Geometry> SourceFp32BaseGeometries(const Workload& w) {
    // Plain FP32 BASE writes one base tile per AIC. Include every aligned
    // local tile up to 256, including 112x128 (five M tiles x four N tiles
    // at M=N=512). The source may pad a short N=64 to a 128-wide tile.
    // L0/L1 capacity and the reduced objective prune this bounded grid.
    std::vector<Geometry> result;
    const auto max_m = std::min<std::uint64_t>(w.m, 256);
    const auto max_n = std::min<std::uint64_t>(
        w.n >= 64 ? std::max<std::uint64_t>(w.n, 128) : w.n, 256);
    for (std::uint64_t m = 16; m <= max_m; m += 16)
        for (std::uint64_t n = 16; n <= max_n; n += 16)
            result.push_back(SourceGeometry(w, m, n, m, n));
    return result;
}

bool SourceFp32NkOrder(const Workload& w) {
    // IsNkOrder() for the audited FP32 NN/ND path. N and M are already
    // 16-aligned by the preflight; 256-byte alignment is 64 FP32 elements.
    return w.m > w.n && w.m >= 128 && w.n % 8 == 0 &&
        !(w.n % 64 == 0 && w.m % 64 != 0) &&
        !(w.m % 64 != 0 && w.n <= 64) &&
        !(w.m >= 2048 && w.n == 16);
}

std::vector<Geometry> SourceFp32SplitGeometries(const Workload& w,
                                                const Hardware& h,
                                                const CoreMemory& mem) {
    // DoDeterministicMultiCoreSplitKTiling() first chooses MK/NK 3x3:
    // base=128x128x64, stepK=3, and one 384-wide outer output axis.
    // It may then split the other output axis to fit 70% of L2 across AICs.
    if (!h.aic_count || mem.l2_bytes < 1024) return {};
    constexpr std::uint64_t kSlice = 192, kBlock = 128, kBytes = 4;
    const auto per_core_l2 = (mem.l2_bytes * 7 / 10) / h.aic_count;
    const auto panel_bound = [&](std::uint64_t fixed_extent) {
        const auto fixed = kSlice * std::min<std::uint64_t>(fixed_extent, 384) * kBytes;
        if (per_core_l2 <= fixed) return std::uint64_t{0};
        return (per_core_l2 - fixed) /
            (kSlice * kBytes + std::min<std::uint64_t>(fixed_extent, 384) * kBytes);
    };
    std::uint64_t core_m = 384, core_n = w.n;
    if (SourceFp32NkOrder(w)) {
        core_m = w.m;
        core_n = 384;
        const auto bound = panel_bound(w.n);
        if (!bound) return {};
        if (w.m > bound) {
            const auto panel = kBlock * CeilDiv(bound, kBlock);
            core_m = kBlock * CeilDiv(CeilDiv(w.m, CeilDiv(w.m, panel)), kBlock);
        }
    } else {
        const auto bound = panel_bound(w.m);
        if (!bound) return {};
        if (w.n > bound) {
            const auto panel = kBlock * CeilDiv(bound, kBlock);
            core_n = kBlock * CeilDiv(CeilDiv(w.n, CeilDiv(w.n, panel)), kBlock);
        }
    }
    return {SourceGeometry(w, core_m, core_n, 128, 128)};
}

std::vector<Geometry> SourceFp32SingleCoreGeometries(const Workload& w,
                                                     const Hardware& h) {
    // The source partitions the output over at most 20 AICs, rounds M to
    // 32 bytes and N to 256 bytes, then uses 128x128x64 base tiles. Search
    // those source-aligned partition extents instead of BF16's 16-aligned
    // geometry grid. The host's spatial load and shape gates are checked
    // by FinalizePlan for each candidate.
    std::vector<Geometry> result;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seen;
    for (std::uint64_t m_parts = 1; m_parts <= h.aic_count; ++m_parts) {
        const auto core_m = std::min(w.m, 8 * CeilDiv(CeilDiv(w.m, m_parts), 8));
        if (core_m < 128) continue;
        for (std::uint64_t n_parts = 1; n_parts <= h.aic_count; ++n_parts) {
            const auto core_n = std::min(w.n, 64 * CeilDiv(CeilDiv(w.n, n_parts), 64));
            if (core_n < 128) continue;
            const auto key = std::make_pair(core_m, core_n);
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
            seen.push_back(key);
            result.push_back(SourceGeometry(w, core_m, core_n, 128, 128));
        }
    }
    return result;
}

std::vector<Geometry> SourceFp32Mode5Geometries(const Workload& w,
                                                const Hardware& h) {
    // The NN K=1536 NKM host route keeps N=384 in one output tile and
    // partitions only M. A forced N=128 tile was correct on 910B3 but
    // 18% slower at 73728x384x1536; the generic byte model tied them
    // because it misses repeated A reads across separate N tiles.
    if (w.k != 1536 || w.n != 384) return {};
    std::vector<Geometry> result;
    for (std::uint64_t parts = 1; parts <= h.aic_count; ++parts) {
        const auto core_m = std::min(w.m, 8 * CeilDiv(CeilDiv(w.m, parts), 8));
        if (core_m < 128) continue;
        if (!result.empty() && result.back().core_m == core_m) continue;
        result.push_back(SourceGeometry(w, core_m, 384, 128, 128));
    }
    return result;
}

std::uint64_t BaseKUnitsUpper(const Workload& w, const Geometry& g,
                              const CoreMemory& mem) {
    // BASE uses baseK=16b and stepK=1. Its modeled double-buffer footprint
    // gives 64*baseM*b <= L0A, 64*baseN*b <= L0B, and
    // 64*(baseM+baseN)*b <= L1; also baseK may not exceed K.
    const double m = static_cast<double>(g.base_m), n = static_cast<double>(g.base_n);
    const auto types = WorkloadTypes(w);
    const double a_bytes = ElementBytes(types.a), b_bytes = ElementBytes(types.b);
    return std::min({w.k / 16,
        Bound(mem.l0a_bytes, 32.0 * a_bytes * m),
        Bound(mem.l0b_bytes, 32.0 * b_bytes * n),
        Bound(mem.l1_bytes, 32.0 * (a_bytes * m + b_bytes * n))});
}

std::uint64_t StepUpper(const Workload& w, const Geometry& g,
                        const CoreMemory& mem, std::uint64_t base_k) {
    // Two operand L1 windows. FP16/BF16 use baseK=128; the audited FP32
    // deterministic path uses baseK=64.
    const auto types = WorkloadTypes(w);
    return std::min(w.k / base_k,
                    Bound(mem.l1_bytes, 2.0 * static_cast<double>(base_k) *
                         (ElementBytes(types.a) * static_cast<double>(g.base_m) +
                          ElementBytes(types.b) * static_cast<double>(g.base_n))));
}

bool FixedTileFits(const Workload& w, const Geometry& g, std::uint64_t base_k,
                   const CoreMemory& mem) {
    // Two L0A/L0B base tiles and two FP32 L0C accumulator tiles.
    const double m = static_cast<double>(g.base_m), n = static_cast<double>(g.base_n);
    const auto types = WorkloadTypes(w);
    return 2.0 * ElementBytes(types.a) * m * base_k <= mem.l0a_bytes &&
           2.0 * ElementBytes(types.b) * n * base_k <= mem.l0b_bytes &&
           2.0 * ElementBytes(types.accumulator) * m * n <= mem.l0c_bytes;
}

MipPlan Finish(const Workload& w, const Hardware& h, const CoreMemory& mem,
               PlanDecision decision, const std::string& id,
               double solver_objective) {
    // FinalizePlan is a post-selection legality cross-check. The expanded
    // event graph is unnecessary for online family selection.
    const auto finalized = FinalizePlan(w, h, decision, id, mem);
    const double recomputed = ReducedObjectiveUs(w, h, decision, mem);
    if (std::abs(recomputed - solver_objective) >
        1e-4 * std::max(1.0, std::abs(recomputed)))
        throw std::runtime_error("selected plan does not match the reduced objective");
    return {decision, finalized.candidate, recomputed, std::nullopt};
}

std::optional<MipPlan> SolveBase(const Workload& w, const Hardware& h,
                                 const CoreMemory& mem, const Rates& r,
                                 const std::vector<Geometry>& geometries,
                                 bool core_equals_base, bool allow_fp32_padding = false) {
    const auto types = WorkloadTypes(w);
    const double conversion_bytes = w.k > 65535 ?
        ElementBytes(types.a) * static_cast<double>(w.m) * w.k : 0.0;
    // mat_mul_v3 reserves the entire converted A matrix, independently of
    // spatial tiling or occupied Cube cores.
    if (kRpcBytes + conversion_bytes > h.workspace_limit_bytes)
        return std::nullopt;

    // For the generic read surrogate, larger baseK cannot increase the
    // K-packet count, so eliminate it at its capacity bound. The optional
    // source-step packet model changes K-span discontinuously with baseK;
    // evaluate every legal aligned baseK there. Core count is checked: the
    // balanced full/tail class assignment can still be imbalanced, and the
    // busiest core determines Cube and packet service. At most 20 counts
    // are evaluated per geometry before taking the minimum across geometries.
    struct Choice {
        Geometry geometry;
        std::uint64_t base_k, cores;
        double stage_us;
    };
    std::vector<Choice> choices;
    for (const auto& g : geometries) {
        if ((g.core_m > w.m || g.core_n > w.n) && !allow_fp32_padding) continue;
        if (core_equals_base &&
            (g.core_m != g.base_m || g.core_n != g.base_n)) continue;
        if (ElementBytes(types.accumulator) * mem.base_l0c_buffers * g.base_m * g.base_n >
            mem.l0c_bytes) continue;
        const auto bmax = BaseKUnitsUpper(w, g, mem);
        const auto cmax = std::min(h.aic_count, g.output_tiles);
        if (!bmax || !cmax) continue;
        double stage = std::numeric_limits<double>::infinity();
        std::uint64_t best_cores = 0;
        std::uint64_t best_base_k = 0;
        const auto first_b = r.base_packet_read ? 1ULL : bmax;
        for (std::uint64_t c = 1; c <= cmax; ++c) {
            // Spatial assignment is independent of baseK. Reuse it across
            // the optional source-step packet candidates.
            const auto loads = BaseCoreLoads(w, g.core_m, g.core_n,
                                             g.base_m, g.base_n, c);
            for (auto b = first_b; b <= bmax; ++b) {
                const auto base_k = 16 * b;
                const double candidate = BaseStageUs(w, r, g, base_k, c,
                                                     mem, h.aic_count, &loads);
                if (candidate <= stage) {
                    stage = candidate;
                    best_cores = c;
                    best_base_k = base_k;
                }
            }
        }
        choices.push_back({g, best_base_k, best_cores, stage});
    }
    if (choices.empty()) return std::nullopt;
    const auto selected = std::min_element(choices.begin(), choices.end(),
        [](const Choice& a, const Choice& b) { return a.stage_us < b.stage_us; });
    PlanDecision d;
    d.family = KernelFamily::Base;
    d.used_cores = selected->cores;
    d.single_core_m = selected->geometry.core_m;
    d.single_core_n = selected->geometry.core_n;
    d.base_m = selected->geometry.base_m;
    d.base_n = selected->geometry.base_n;
    d.single_core_k = w.k;
    d.base_k = selected->base_k;
    return Finish(w, h, mem, d, "MIP_BASE", selected->stage_us + CommonUs(w, r));
}

std::optional<MipPlan> SolveSplit(const Workload& w, const Hardware& h,
                                  const CoreMemory& mem, const Rates& r,
                                  const std::vector<Geometry>& geometries,
                                  bool source_fp32_pattern = false) {
    const std::uint64_t kBaseK = w.dtype == "fp32" ? 64 : 128;
    const auto types = WorkloadTypes(w);
    const double conversion_bytes = w.k > 65535 ?
        ElementBytes(types.a) * static_cast<double>(w.m) * w.k : 0.0;
    const double fixed_workspace = kRpcBytes + conversion_bytes;
    if (fixed_workspace > h.workspace_limit_bytes) return std::nullopt;

    // This is the standalone reduced Split-K domain. mat_mul_v3's host
    // tiler starts from MK/NK 3x3 patterns with branch-specific L1 depths;
    // arbitrary geometry/step choices here still need source-aware legality
    // and CANN finalization before they can be considered emittable.
    // A larger step changes the busiest AIC's K packets because the source
    // assigns whole slices. The vector chain length depends on the core
    // count, not on the step. Enumerate the bounded legal pairs. The
    // Take the minimum of the feasible per-geometry choices.
    struct Choice {
        Geometry geometry;
        std::uint64_t step, cores;
        double local_us;
    };
    std::vector<Choice> choices;
    const auto k_packets = CeilDiv(w.k, kBaseK);
    for (const auto& g : geometries) {
        if (!FixedTileFits(w, g, kBaseK, mem)) continue;
        const auto smax = StepUpper(w, g, mem, kBaseK);
        const auto workspace_cores = Bound(
            static_cast<double>(h.workspace_limit_bytes) - fixed_workspace,
            2.0 * ElementBytes(types.accumulator) * static_cast<double>(g.core_m) * g.core_n);
        const auto cmax = std::min<std::uint64_t>({h.aic_count, workspace_cores,
                                                   k_packets});
        if (!smax || cmax < 2) continue;

        double best_us = std::numeric_limits<double>::infinity();
        std::uint64_t best_step = 0, best_cores = 0;
        const auto source_cores = std::min(h.aic_count, CeilDiv(w.k, 3 * kBaseK));
        const auto small_mn = w.m <= 64 && w.n <= 64 && w.k >= 6144;
        const auto fixed_cores = small_mn ? std::min(h.aic_count,
            8 + (w.k - 6144) / 1024 + CeilDiv(w.m, 16) + CeilDiv(w.n, 16)) :
            source_cores;
        for (std::uint64_t c = source_fp32_pattern ? fixed_cores : 2;
             c <= (source_fp32_pattern ? fixed_cores : cmax); ++c) {
            for (std::uint64_t step = source_fp32_pattern ? 3 : 1;
                 step <= (source_fp32_pattern ? 3 : smax); ++step) {
                if (c > cmax || step > smax || c < 2) continue;
                if (CeilDiv(k_packets, step) < c) continue;
                const double local = SplitStageUs(w, r, g, kBaseK, step, c);
                if (local < best_us) {
                    best_us = local;
                    best_step = step;
                    best_cores = c;
                }
            }
        }
        if (!best_step) continue;
        choices.push_back({g, best_step, best_cores, best_us});
    }
    if (choices.empty()) return std::nullopt;
    const auto selected = std::min_element(choices.begin(), choices.end(),
        [](const Choice& a, const Choice& b) { return a.local_us < b.local_us; });
    PlanDecision d;
    d.family = KernelFamily::DeterministicSplitK;
    d.used_cores = selected->cores;
    d.single_core_m = selected->geometry.core_m;
    d.single_core_n = selected->geometry.core_n;
    d.base_m = selected->geometry.base_m;
    d.base_n = selected->geometry.base_n;
    d.base_k = kBaseK;
    d.step_k = selected->step;
    d.single_core_k = d.base_k * d.step_k;
    return Finish(w, h, mem, d, "MIP_DETERMINISTIC_SPLIT_K",
                  selected->local_us + CommonUs(w, r));
}

std::optional<MipPlan> SolveSingleCore(const Workload& w, const Hardware& h,
                                       const CoreMemory& mem, const Rates& r,
                                       const std::vector<Geometry>& geometries,
                                       KernelFamily family) {
    if (w.dtype == "fp32" && family == KernelFamily::SingleCoreSplitKMode6)
        return std::nullopt;
    if (w.dtype != "fp16" && w.dtype != "bf16" && w.dtype != "fp32")
        return std::nullopt;
    struct Choice { PlanDecision decision; double local_us; };
    std::vector<Choice> choices;
    for (const auto& g : geometries) {
        if (g.base_m != 128 || g.base_n != 128 ||
            g.core_m > w.m || g.core_n > w.n) continue;
        const auto cmax = std::min(h.aic_count, g.output_tiles);
        if (!cmax) continue;
        double best = std::numeric_limits<double>::infinity();
        PlanDecision best_d;
        const bool short_side = w.m < 384 || w.n < 384;
        for (std::uint64_t step_m = 1; step_m <= 3; ++step_m) {
            if (family == KernelFamily::SingleCoreSplitKMode5 && step_m != 1)
                continue;
            if (family != KernelFamily::SingleCoreSplitKMode5 &&
                ((short_side && step_m != 1) || (!short_side && step_m == 1)))
                continue;
            for (std::uint64_t cores = 1; cores <= cmax; ++cores) {
                PlanDecision d;
                d.family = family;
                d.used_cores = cores;
                d.single_core_m = g.core_m;
                d.single_core_n = g.core_n;
                d.base_m = d.base_n = 128;
                d.base_k = w.dtype == "fp32" ? 64 : 128;
                d.step_m = step_m;
                d.step_n = family == KernelFamily::SingleCoreSplitKMode5 ? 3 : 1;
                d.step_k = step_m == 3 || family == KernelFamily::SingleCoreSplitKMode5 ? 3 : 4;
                d.single_core_k = d.base_k * d.step_k;
                try {
                    (void)FinalizePlan(w, h, d, "single_core_preflight", mem);
                    const double local = ScStageUs(w, r, d);
                    if (local < best) { best = local; best_d = d; }
                } catch (const std::invalid_argument&) {
                    // The source block, host shape gate, or memory is illegal.
                }
            }
        }
        if (!std::isfinite(best)) continue;
        choices.push_back({best_d, best});
    }
    if (choices.empty()) return std::nullopt;
    const auto selected = std::min_element(choices.begin(), choices.end(),
        [](const Choice& a, const Choice& b) { return a.local_us < b.local_us; });
    const char* id = family == KernelFamily::SingleCoreSplitKMode2 ?
        "MIP_SINGLE_CORE_SPLIT_K_MODE2" :
        family == KernelFamily::SingleCoreSplitKMode5 ?
        "MIP_SINGLE_CORE_SPLIT_K_MODE5" : "MIP_SINGLE_CORE_SPLIT_K_MODE6";
    return Finish(w, h, mem, selected->decision, id,
                  selected->local_us + CommonUs(w, r));
}

}  // namespace

double ReducedObjectiveUs(const Workload& w, const Hardware& h,
                          const PlanDecision& d, const CoreMemory& mem,
                          bool source_emitted_base) {
    // Recompute the service-time objective for a complete plan. Finish()
    // checks that the selected cost matches this value.
    const auto r = ReadRates(h, w);
    (void)FinalizePlan(w, h, d, "reduced_objective", mem);
    if (IsSingleCore(d.family))
        return CommonUs(w, r) + ScStageUs(w, r, d);
    if (d.family == KernelFamily::Base &&
        (d.base_m > d.single_core_m || d.base_n > d.single_core_n))
        throw std::invalid_argument("base M/N tile exceeds single-core output tile");
    if (r.base_packet_read && d.family == KernelFamily::Base &&
        (d.single_core_m != d.base_m || d.single_core_n != d.base_n))
        throw std::invalid_argument("BASE packet profile requires source plain BASE core=base geometry");
    const auto pm = AxisPackets(w.m, d.single_core_m, d.base_m);
    const auto pn = AxisPackets(w.n, d.single_core_n, d.base_n);
    const double outputs = static_cast<double>(CeilDiv(w.m, d.single_core_m)) *
                           static_cast<double>(CeilDiv(w.n, d.single_core_n));
    const double spatial_packets = static_cast<double>(pm) * pn;
    const auto types = WorkloadTypes(w);
    const double a_bytes = ElementBytes(types.a) * static_cast<double>(w.k) * w.m * pn;
    const double b_bytes = ElementBytes(types.b) * static_cast<double>(w.k) * w.n * pm;
    const double operand_bytes = a_bytes + b_bytes;
    const double common = CommonUs(w, r);
    if (d.family == KernelFamily::Base) {
        const Geometry g{d.single_core_m, d.single_core_n,
                         d.base_m, d.base_n,
                         static_cast<std::uint64_t>(outputs),
                         static_cast<std::uint64_t>(spatial_packets),
                         operand_bytes, a_bytes, b_bytes};
        return common + BaseStageUs(w, r, g, d.base_k, d.used_cores,
                                    mem, h.aic_count, nullptr, source_emitted_base);
    }
    if (d.base_k != (w.dtype == "fp32" ? 64 : 128))
        throw std::invalid_argument("reduced Split-K baseK does not match dtype path");
    const Geometry g{d.single_core_m, d.single_core_n, d.base_m, d.base_n,
                     static_cast<std::uint64_t>(outputs),
                     static_cast<std::uint64_t>(spatial_packets),
                     operand_bytes, a_bytes, b_bytes};
    return common + SplitStageUs(w, r, g, d.base_k, d.step_k, d.used_cores);
}

MipResult SolveReducedMip(const Workload& w, const Hardware& h,
                          const CoreMemory& mem, const MipTileDomain& domain) {
    // These are API/domain bounds. Global layout, dtype,
    // feature and family preconditions remain in FinalizePlan; it rejects an
    // unsupported selected plan before any result is returned.
    if (!w.m || !w.n || !w.k || w.m > 1000000 || w.n > 1000000 ||
        w.k > 1000000000 || !h.aic_count || h.aic_count > 20)
        throw std::invalid_argument("unsupported reduced-model dimensions or AIC count");
    const auto types = WorkloadTypes(w);
    if ((types.a != "fp16" && types.a != "bf16" && types.a != "fp32") ||
        types.b != types.a || types.c != types.a ||
        types.accumulator != "fp32" || w.dtype != types.a)
        throw std::invalid_argument("reduced model supports same-type FP16/BF16/FP32 with FP32 accumulation");
    if (types.a == "fp32" && w.k > 65535)
        throw std::invalid_argument("FP32 converted-A path is not modeled yet");
    const auto rates = ReadRates(h, w);
    if (rates.base_packet_read && !domain.base_core_equals_base)
        throw std::invalid_argument("BASE packet profile requires base_core_equals_base domain");
    const bool source_fp32 = w.dtype == "fp32" && domain.fp32_source_patterns &&
        domain.single_core_m.empty() && domain.single_core_n.empty() &&
        domain.base_m.empty() && domain.base_n.empty();
    const auto geometries = source_fp32 ? std::vector<Geometry>{} : Geometries(w, h, domain);
    MipResult result;
    result.base = SolveBase(w, h, mem, rates,
        source_fp32 ? SourceFp32BaseGeometries(w) : geometries,
        domain.base_core_equals_base, source_fp32);
    result.deterministic_split_k = SolveSplit(w, h, mem, rates,
        source_fp32 ? SourceFp32SplitGeometries(w, h, mem) : geometries,
        source_fp32);
    result.single_core_split_k_mode2 = SolveSingleCore(
        w, h, mem, rates,
        source_fp32 ? SourceFp32SingleCoreGeometries(w, h) : geometries,
        KernelFamily::SingleCoreSplitKMode2);
    result.single_core_split_k_mode5 = SolveSingleCore(
        w, h, mem, rates,
        source_fp32 ? SourceFp32Mode5Geometries(w, h) : geometries,
        KernelFamily::SingleCoreSplitKMode5);
    result.single_core_split_k_mode6 = SolveSingleCore(
        w, h, mem, rates, geometries, KernelFamily::SingleCoreSplitKMode6);
    // The outer implementation choice compares reduced objectives.
    for (const auto* plan : {&result.base, &result.deterministic_split_k,
                             &result.single_core_split_k_mode2,
                             &result.single_core_split_k_mode5,
                             &result.single_core_split_k_mode6})
        if (*plan && (!result.winner || (*plan)->surrogate_us < result.winner->surrogate_us))
            result.winner = *plan;
    return result;
}

}  // namespace mmopt
