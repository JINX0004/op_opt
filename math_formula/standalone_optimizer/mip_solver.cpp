#include "mip_solver.h"

#include "event_builder.h"
#include "split_vector_model.h"
#include "highs/interfaces/highs_c_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mmopt {
namespace {

constexpr double kRpcBytes = 20.0 * 1024.0 * 1024.0;

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
    };
    std::optional<BasePacketRead> base_packet_read;
    // Fixed MTE2→V / add / V→MTE2 cost of one extra Split-K partial.
    // Absent from a profile means zero; the reduction probe supplies it.
    double split_partial_us = 0.0;
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

Rates ReadRates(const Hardware& hardware) {
    // Every primitive used by either family must have a finite nonnegative
    // fixed cost and a positive rate in the specified unit. Per-packet Cube
    // fixed latency is not represented by the reduced service objective.
    Rates r{Get(hardware, "launch", "count"),
            Get(hardware, "gm_read", "bytes"),
            Get(hardware, "gm_write", "bytes"),
            Get(hardware, "cube", "flop"),
            Get(hardware, "vector_add", "ops"),
            Get(hardware, "sync", "count"),
            Get(hardware, "nd2nz", "elements"), {}, {}, {}};
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
            Get(hardware, packet_names[3], "rows")};
        if (r.base_packet_read->a_rows.fixed_latency_us != 0 ||
            r.base_packet_read->b_rows.fixed_latency_us != 0)
            throw std::invalid_argument("BASE MTE2 row primitives must have zero fixed latency");
    }
    if (r.cube.fixed_latency_us != 0.0)
        throw std::invalid_argument("reduced MILP currently requires zero fixed Cube latency");
    const auto partial = hardware.primitives.find("split_partial");
    if (partial != hardware.primitives.end()) {
        if (partial->second.unit != "count" ||
            !std::isfinite(partial->second.fixed_latency_us) ||
            partial->second.fixed_latency_us < 0)
            throw std::invalid_argument("invalid primitive split_partial");
        r.split_partial_us = partial->second.fixed_latency_us;
    }
    return r;
}

double ConversionUs(const Workload& w, const Rates& r) {
    if (w.k <= 65535) return 0.0;
    const double elements = static_cast<double>(w.m) * static_cast<double>(w.k);
    const double bytes = 2.0 * elements;
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

// This class only wraps the official HiGHS C API; all rows below are the
// mathematical program, with no calls to the event-graph evaluator in solve.
class Model {
public:
    Model() : handle_(Highs_create(), Highs_destroy) {
        if (!handle_) throw std::runtime_error("HiGHS creation failed");
        Check(Highs_setBoolOptionValue(handle_.get(), "output_flag", 0));
        Check(Highs_setDoubleOptionValue(handle_.get(), "mip_rel_gap", 0.0));
        Check(Highs_setDoubleOptionValue(handle_.get(), "mip_abs_gap", 0.0));
    }

    int Column(double cost, double lower, double upper, bool integer) {
        const int index = columns_++;
        Check(Highs_addCol(handle_.get(), cost, lower, upper, 0, nullptr, nullptr));
        if (integer)
            Check(Highs_changeColIntegrality(handle_.get(), index, kHighsVarTypeInteger));
        return index;
    }

    void Row(double lower, double upper,
             std::initializer_list<std::pair<int, double>> coefficients) {
        Row(lower, upper,
            std::vector<std::pair<int, double>>(coefficients));
    }

    void Row(double lower, double upper,
             const std::vector<std::pair<int, double>>& coefficients) {
        std::vector<HighsInt> indices;
        std::vector<double> values;
        for (const auto& [index, value] : coefficients) {
            indices.push_back(index);
            values.push_back(value);
        }
        Check(Highs_addRow(handle_.get(), lower, upper,
                           static_cast<HighsInt>(indices.size()),
                           indices.data(), values.data()));
    }

    std::optional<std::vector<double>> Solve() {
        const auto status = Highs_run(handle_.get());
        if (status == kHighsStatusError)
            throw std::runtime_error("HiGHS run failed");
        const auto model_status = Highs_getModelStatus(handle_.get());
        if (model_status == kHighsModelStatusInfeasible) return std::nullopt;
        if (model_status != kHighsModelStatusOptimal)
            throw std::runtime_error("HiGHS did not prove an optimal solution; status=" +
                                     std::to_string(model_status));
        std::vector<double> values(columns_);
        Check(Highs_getSolution(handle_.get(), values.data(), nullptr, nullptr, nullptr));
        return values;
    }

private:
    static void Check(HighsInt status) {
        if (status != kHighsStatusOk) throw std::runtime_error("HiGHS API call failed");
    }
    std::unique_ptr<void, decltype(&Highs_destroy)> handle_;
    int columns_ = 0;
};

std::uint64_t Integer(double value, const char* name) {
    if (!std::isfinite(value) || value < 0 ||
        value > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ||
        std::abs(value - std::round(value)) > 1e-4)
        throw std::runtime_error(std::string("noninteger MILP result for ") + name);
    return static_cast<std::uint64_t>(std::llround(value));
}

std::uint64_t Bound(double numerator, double denominator) {
    if (denominator <= 0 || !std::isfinite(denominator)) return 0;
    const double quotient = std::floor(numerator / denominator);
    if (!std::isfinite(quotient) || quotient < 0) return 0;
    return static_cast<std::uint64_t>(std::min(quotient, 1000000.0));
}

struct Geometry {
    // One discrete M/N choice g. Counts include output and base-tile tails.
    // They are constants in the MILP, so no product of tile variables is
    // introduced in the linear rows.
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
    const auto packet_a = 2 * g.base_m * base_k;
    const auto packet_b = 2 * g.base_n * base_k;
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
    const auto align = [&](std::uint64_t& step) {
        if (step * base_k >= w.k) return;
        const auto span_bytes = step * base_k * 2;
        const auto quantum = span_bytes > 512 ? 512ULL :
                             span_bytes > 256 ? 256ULL : 0ULL;
        if (quantum && quantum % (base_k * 2) == 0)
            while (step > 1 && (step * base_k * 2) % quantum) --step;
    };
    align(step_a);
    align(step_b);
    if (step_a >= step_b) step_a = step_a / step_b * step_b;
    else step_b = step_b / step_a * step_a;
    return {step_a, step_b};
}

double BaseStageUs(const Workload& w, const Rates& r,
                   const Geometry& g, std::uint64_t base_k,
                   std::uint64_t cores, const CoreMemory& mem,
                   const std::vector<BaseCoreLoad>* cached_loads = nullptr) {
    const auto computed_loads = cached_loads ? std::vector<BaseCoreLoad>{} :
        BaseCoreLoads(w, g.core_m, g.core_n, g.base_m, g.base_n, cores);
    const auto& loads = cached_loads ? *cached_loads : computed_loads;
    if (r.base_packet_read) {
        // One packet is the source's entire L1 K step, not a baseK slice.
        // Fit: fixed packet service + requested bytes + ND source rows.
        // The probes saturate 20 AICs, so this is an exploratory per-core
        // service objective; Cube/MTE2 overlap is approximated by max.
        const auto steps = SourceBaseL1Steps(w, g, base_k, mem);
        const auto& p = *r.base_packet_read;
        const double count_a = static_cast<double>(CeilDiv(w.k, steps.a * base_k));
        const double count_b = static_cast<double>(CeilDiv(w.k, steps.b * base_k));
        const double shared_write = 2.0 * OutputElements(w) /
                                    r.write.rate_per_us;
        double critical = 0.0;
        for (const auto& load : loads) {
            const double cube = 2.0 * load.elements * w.k / r.cube.rate_per_us;
            const double read =
                load.spatial_packets * (count_a * p.a_bytes.fixed_latency_us +
                                        count_b * p.b_bytes.fixed_latency_us) +
                2.0 * w.k * load.a_axis_sum / p.a_bytes.rate_per_us +
                2.0 * w.k * load.b_axis_sum / p.b_bytes.rate_per_us +
                count_a * load.a_axis_sum / p.a_rows.rate_per_us +
                w.k * load.spatial_packets / p.b_rows.rate_per_us;
            critical = std::max(critical, std::max(
                cube, read + shared_write +
                      load.output_tiles * r.write.fixed_latency_us));
        }
        return critical;
    }
    const double k_packets = static_cast<double>(CeilDiv(w.k, base_k));
    double cube_critical = 0.0, packet_critical = 0.0;
    for (const auto& load : loads) {
        cube_critical = std::max(cube_critical,
            2.0 * load.elements * static_cast<double>(w.k) /
                r.cube.rate_per_us);
        packet_critical = std::max(packet_critical,
            load.spatial_packets * k_packets *
                (r.read_a.fixed_latency_us + r.read_b.fixed_latency_us) +
                load.output_tiles * r.write.fixed_latency_us);
    }
    const double shared_bytes = g.a_bytes / r.read_a.rate_per_us +
        g.b_bytes / r.read_b.rate_per_us +
        2.0 * OutputElements(w) / r.write.rate_per_us;
    return std::max(cube_critical, shared_bytes + packet_critical);
}

std::uint64_t AxisPackets(std::uint64_t shape, std::uint64_t core,
                          std::uint64_t base);

double SplitStageUs(const Workload& w, const Rates& r,
                    const Geometry& g, std::uint64_t step,
                    std::uint64_t cores) {
    // Panel pipeline from SOLVE_SPLIT_FORMULATION.md. The source NK path
    // (M > N) ping-pongs N panels; otherwise it ping-pongs M panels. Each
    // panel repeats the same K split. Operand reads overlap that panel's
    // Cube work. The vector chain then adds one accumulated partial per AIC.
    const auto q = CeilDiv(w.k, 128);
    const auto p = CeilDiv(q, step);
    const auto rounds = CeilDiv(p, cores);
    const auto q_crit = std::min(q, rounds * step);
    const auto k_crit = std::min(w.k, q_crit * 128);
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
        const double operand_bytes = 2.0 * static_cast<double>(w.k) *
            (static_cast<double>(m_ext) * n_packets +
             static_cast<double>(n_ext) * m_packets);
        const double cube = 2.0 * elements * k_crit / r.cube.rate_per_us;
        const double operand_gm = operand_bytes / r.read.rate_per_us +
            2.0 * spatial * q_crit * r.read.fixed_latency_us;
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
            throw std::invalid_argument("MILP tile domain requires aligned values within shape, except source Split-K 384");
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

bool CanExpandEvents(const Workload& w, const PlanDecision& d) {
    // The event builder has not been validated for source Split-K core
    // extents larger than an output axis. Preserve the reduced plan only.
    if (d.single_core_m > w.m || d.single_core_n > w.n) return false;
    const long double pm = AxisPackets(w.m, d.single_core_m, d.base_m);
    const long double pn = AxisPackets(w.n, d.single_core_n, d.base_n);
    const long double outputs = static_cast<long double>(CeilDiv(w.m, d.single_core_m)) *
                                CeilDiv(w.n, d.single_core_n);
    // Each base packet needs two reads and one Cube event. This is a
    // conservative lower bound; the builder enforces its own exact cap.
    const long double minimum = 3.0L * pm * pn * CeilDiv(w.k, d.base_k) +
                                outputs + 1.0L;
    return minimum <= 200000.0L;
}

std::vector<Geometry> Geometries(const Workload& w, const Hardware& h,
                                 const MipTileDomain& domain) {
    const auto cm = TileValues(domain.single_core_m, w.m, true, h.aic_count,
                               w.k > 65535);
    const auto cn = TileValues(domain.single_core_n, w.n, true, h.aic_count,
                               w.k > 65535);
    const auto bm = TileValues(domain.base_m, w.m, false, h.aic_count, false);
    const auto bn = TileValues(domain.base_n, w.n, false, h.aic_count, false);
    std::vector<Geometry> result;
    // Source deterministic Split-K may use an outer 384 tile above a small
    // output axis; BASE filters that geometry before its master. Other
    // capacity tests discard infeasible geometries in each family.
    for (const auto m : cm) for (const auto n : cn)
        for (const auto a : bm) for (const auto b : bn) {
            if (a > m || b > n) continue;
            const auto pm = AxisPackets(w.m, m, a);
            const auto pn = AxisPackets(w.n, n, b);
            const auto outputs = CeilDiv(w.m, m) * CeilDiv(w.n, n);
            const auto packets = pm * pn;
            const double a_bytes = 2.0 * static_cast<double>(w.k) * w.m * pn;
            const double b_bytes = 2.0 * static_cast<double>(w.k) * w.n * pm;
            result.push_back({m, n, a, b, outputs, packets,
                              a_bytes + b_bytes, a_bytes, b_bytes});
            if (result.size() > 4096)
                throw std::invalid_argument("MILP tile domain exceeds 4096 spatial geometries");
        }
    return result;
}

std::uint64_t BaseKUnitsUpper(const Workload& w, const Geometry& g,
                              const CoreMemory& mem) {
    // BASE uses baseK=16b and stepK=1. Its modeled double-buffer footprint
    // gives 64*baseM*b <= L0A, 64*baseN*b <= L0B, and
    // 64*(baseM+baseN)*b <= L1; also baseK may not exceed K.
    const double m = static_cast<double>(g.base_m), n = static_cast<double>(g.base_n);
    return std::min({w.k / 16,
        Bound(mem.l0a_bytes, 64.0 * m),
        Bound(mem.l0b_bytes, 64.0 * n),
        Bound(mem.l1_bytes, 64.0 * (m + n))});
}

std::uint64_t StepUpper(const Workload& w, const Geometry& g,
                        const CoreMemory& mem) {
    // Split-K fixes baseK=128. The two L1 step windows require
    // 4*128*(baseM+baseN)*stepK <= L1, with stepK <= K/128.
    return std::min(w.k / 128,
                    Bound(mem.l1_bytes, 4.0 * 128.0 *
                         (static_cast<double>(g.base_m) + static_cast<double>(g.base_n))));
}

bool FixedTileFits(const Geometry& g, std::uint64_t base_k,
                   const CoreMemory& mem) {
    // Two L0A/L0B base tiles and two FP32 L0C accumulator tiles.
    const double m = static_cast<double>(g.base_m), n = static_cast<double>(g.base_n);
    return 4.0 * m * base_k <= mem.l0a_bytes &&
           4.0 * n * base_k <= mem.l0b_bytes &&
           8.0 * m * n <= mem.l0c_bytes;
}

MipPlan Finish(const Workload& w, const Hardware& h, const CoreMemory& mem,
               PlanDecision decision, const std::string& id,
               double solver_objective) {
    // FinalizePlan is a post-solve legality cross-check. The event graph is
    // scored for reporting; neither call changes the MILP objective or the
    // choice of the winner between the two implementation families.
    const auto finalized = FinalizePlan(w, h, decision, id, mem);
    const double recomputed = ReducedObjectiveUs(w, h, decision, mem);
    if (std::abs(recomputed - solver_objective) >
        1e-4 * std::max(1.0, std::abs(recomputed)))
        throw std::runtime_error("MILP solution does not match the reduced objective");
    auto candidate = finalized.candidate;
    std::optional<Score> event_score;
    // The explicit deterministic event builder represents one FP32 partial
    // for every output element and occupied AIC, whereas the reduced model
    // and source workspace reuse storage by output tile. Do not let that
    // graph-only representation limit reject an otherwise finalized large
    // plan. Its detailed score is unavailable, as with the event cap.
    const bool graph_workspace_fits =
        decision.family != KernelFamily::DeterministicSplitK ||
        static_cast<long double>(decision.used_cores) * w.m * w.n * 4.0L <=
            static_cast<long double>(finalized.candidate.workspace_bytes);
    if (CanExpandEvents(w, decision) && graph_workspace_fits) {
        try {
            candidate = BuildMatmulEvents(w, std::move(candidate));
            const auto scored = Optimize(w, h, {candidate});
            if (!scored.has_winner())
                throw std::runtime_error("MILP plan rejected by event-graph scorer");
            event_score = scored.winner().score;
        } catch (const std::length_error&) {
            // The reduced MILP can cover shapes larger than the explicitly
            // expanded event graph. Preserve the optimized plan and mark its
            // detailed timeline unavailable, rather than inventing a score.
            candidate = finalized.candidate;
        }
    }
    return {decision, std::move(candidate), recomputed, std::move(event_score)};
}

std::optional<MipPlan> SolveBase(const Workload& w, const Hardware& h,
                                 const CoreMemory& mem, const Rates& r,
                                 const std::vector<Geometry>& geometries,
                                 bool core_equals_base) {
    const double conversion_bytes = w.k > 65535 ? 2.0 * w.m * w.k : 0.0;
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
    // are evaluated per geometry before the one-hot geometry master.
    Model master;
    struct Choice {
        Geometry geometry;
        int y;
        std::uint64_t base_k, cores;
        double stage_us;
    };
    std::vector<Choice> choices;
    std::vector<std::pair<int, double>> select_one;
    for (const auto& g : geometries) {
        if (g.core_m > w.m || g.core_n > w.n) continue;
        if (core_equals_base &&
            (g.core_m != g.base_m || g.core_n != g.base_n)) continue;
        if (4.0 * mem.base_l0c_buffers * g.base_m * g.base_n >
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
                                                     mem, &loads);
                if (candidate <= stage) {
                    stage = candidate;
                    best_cores = c;
                    best_base_k = base_k;
                }
            }
        }
        const int y = master.Column(stage, 0.0, 1.0, true);
        choices.push_back({g, y, best_base_k, best_cores, stage});
        select_one.emplace_back(y, 1.0);
    }
    if (choices.empty()) return std::nullopt;
    master.Row(1.0, 1.0, select_one);
    const auto solution = master.Solve();
    if (!solution) return std::nullopt;
    const auto selected = std::find_if(choices.begin(), choices.end(),
        [&](const Choice& choice) { return Integer(solution->at(choice.y), "geometry") == 1; });
    if (selected == choices.end())
        throw std::runtime_error("MILP returned no BASE geometry");
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
                                  const std::vector<Geometry>& geometries) {
    constexpr std::uint64_t kBaseK = 128;
    const double conversion_bytes = w.k > 65535 ? 2.0 * w.m * w.k : 0.0;
    const double fixed_workspace = kRpcBytes + conversion_bytes;
    if (fixed_workspace > h.workspace_limit_bytes) return std::nullopt;

    // This is the standalone reduced Split-K domain. mat_mul_v3's host
    // tiler starts from MK/NK 3x3 patterns with branch-specific L1 depths;
    // arbitrary geometry/step choices here still need source-aware legality
    // and CANN finalization before they can be considered emittable.
    // A larger step changes the busiest AIC's K packets because the source
    // assigns whole slices. The vector chain length depends on the core
    // count, not on the step. Enumerate the bounded legal pairs. The
    // master has one binary column per feasible geometry and one row.
    Model master;
    struct Choice {
        Geometry geometry;
        int y;
        std::uint64_t step, cores;
        double local_us;
    };
    std::vector<Choice> choices;
    std::vector<std::pair<int, double>> select_one;
    const auto k_packets = CeilDiv(w.k, kBaseK);
    for (const auto& g : geometries) {
        if (!FixedTileFits(g, kBaseK, mem)) continue;
        const auto smax = StepUpper(w, g, mem);
        const auto workspace_cores = Bound(
            static_cast<double>(h.workspace_limit_bytes) - fixed_workspace,
            8.0 * static_cast<double>(g.core_m) * g.core_n);
        const auto cmax = std::min<std::uint64_t>({h.aic_count, workspace_cores,
                                                   k_packets});
        if (!smax || cmax < 2) continue;

        double best_us = std::numeric_limits<double>::infinity();
        std::uint64_t best_step = 0, best_cores = 0;
        for (std::uint64_t c = 2; c <= cmax; ++c) {
            for (std::uint64_t step = 1; step <= smax; ++step) {
                if (CeilDiv(k_packets, step) < c) continue;
                const double local = SplitStageUs(w, r, g, step, c);
                if (local < best_us) {
                    best_us = local;
                    best_step = step;
                    best_cores = c;
                }
            }
        }
        if (!best_step) continue;
        const int y = master.Column(best_us, 0.0, 1.0, true);
        choices.push_back({g, y, best_step, best_cores, best_us});
        select_one.emplace_back(y, 1.0);
    }
    if (choices.empty()) return std::nullopt;
    master.Row(1.0, 1.0, select_one);
    const auto solution = master.Solve();
    if (!solution) return std::nullopt;
    const auto selected = std::find_if(choices.begin(), choices.end(),
        [&](const Choice& choice) { return Integer(solution->at(choice.y), "geometry") == 1; });
    if (selected == choices.end())
        throw std::runtime_error("MILP returned no Split-K geometry");
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

}  // namespace

double ReducedObjectiveUs(const Workload& w, const Hardware& h,
                          const PlanDecision& d, const CoreMemory& mem) {
    // Recompute the service-time objective for a complete plan. Finish()
    // checks that the selected master coefficient matches this value.
    const auto r = ReadRates(h);
    (void)FinalizePlan(w, h, d, "reduced_objective", mem);
    if (d.base_m > d.single_core_m || d.base_n > d.single_core_n)
        throw std::invalid_argument("base M/N tile exceeds single-core output tile");
    if (r.base_packet_read && d.family == KernelFamily::Base &&
        (d.single_core_m != d.base_m || d.single_core_n != d.base_n))
        throw std::invalid_argument("BASE packet profile requires source plain BASE core=base geometry");
    const auto pm = AxisPackets(w.m, d.single_core_m, d.base_m);
    const auto pn = AxisPackets(w.n, d.single_core_n, d.base_n);
    const double outputs = static_cast<double>(CeilDiv(w.m, d.single_core_m)) *
                           static_cast<double>(CeilDiv(w.n, d.single_core_n));
    const double spatial_packets = static_cast<double>(pm) * pn;
    const double a_bytes = 2.0 * static_cast<double>(w.k) * w.m * pn;
    const double b_bytes = 2.0 * static_cast<double>(w.k) * w.n * pm;
    const double operand_bytes = a_bytes + b_bytes;
    const double common = CommonUs(w, r);
    if (d.family == KernelFamily::Base) {
        const Geometry g{d.single_core_m, d.single_core_n,
                         d.base_m, d.base_n,
                         static_cast<std::uint64_t>(outputs),
                         static_cast<std::uint64_t>(spatial_packets),
                         operand_bytes, a_bytes, b_bytes};
        return common + BaseStageUs(w, r, g, d.base_k, d.used_cores, mem);
    }
    if (d.base_k != 128)
        throw std::invalid_argument("reduced Split-K fixes baseK=128");
    const Geometry g{d.single_core_m, d.single_core_n, d.base_m, d.base_n,
                     static_cast<std::uint64_t>(outputs),
                     static_cast<std::uint64_t>(spatial_packets),
                     operand_bytes, a_bytes, b_bytes};
    return common + SplitStageUs(w, r, g, d.step_k, d.used_cores);
}

MipResult SolveReducedMip(const Workload& w, const Hardware& h,
                          const CoreMemory& mem, const MipTileDomain& domain) {
    // These are API/domain bounds, not MILP rows. Global layout, dtype,
    // feature and family preconditions remain in FinalizePlan; it rejects an
    // unsupported selected plan before any result is returned.
    if (!w.m || !w.n || !w.k || w.m > 1000000 || w.n > 1000000 ||
        w.k > 1000000000 || !h.aic_count || h.aic_count > 20)
        throw std::invalid_argument("unsupported reduced-MILP dimensions or AIC count");
    const auto rates = ReadRates(h);
    if (rates.base_packet_read && !domain.base_core_equals_base)
        throw std::invalid_argument("BASE packet profile requires base_core_equals_base domain");
    const auto geometries = Geometries(w, h, domain);
    MipResult result;
    result.base = SolveBase(w, h, mem, rates, geometries,
                            domain.base_core_equals_base);
    result.deterministic_split_k = SolveSplit(w, h, mem, rates, geometries);
    // The outer implementation choice compares certified reduced objectives,
    // not the separately reported event-graph scores.
    if (result.base && result.deterministic_split_k)
        result.winner = result.base->surrogate_us <=
                        result.deterministic_split_k->surrogate_us
                            ? result.base : result.deterministic_split_k;
    else if (result.base) result.winner = result.base;
    else if (result.deterministic_split_k) result.winner = result.deterministic_split_k;
    return result;
}

}  // namespace mmopt
