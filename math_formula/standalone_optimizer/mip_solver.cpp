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
            Get(hardware, "nd2nz", "elements")};
    if (r.cube.fixed_latency_us != 0.0)
        throw std::invalid_argument("reduced MILP currently requires zero fixed Cube latency");
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

double Flops(const Workload& w) {
    return 2.0 * static_cast<double>(w.m) * static_cast<double>(w.n) *
           static_cast<double>(w.k);
}

double OutputElements(const Workload& w) {
    return static_cast<double>(w.m) * static_cast<double>(w.n);
}

double SplitVectorCriticalElements(const Workload& w, std::uint64_t core_m,
                                   std::uint64_t core_n, std::uint64_t cores) {
    // Each output tile is reduced by up to two AIVs per occupied AIC. The
    // outer tiles are processed in sequence, while AIV work within a tile
    // overlaps. Count the largest balanced shard for each full/tail tile.
    const auto full_m = w.m / core_m, full_n = w.n / core_n;
    const auto tail_m = w.m % core_m, tail_n = w.n % core_n;
    double critical = static_cast<double>(full_m) * full_n *
        SplitVectorMaxElements(core_m * core_n, cores);
    if (tail_m) critical += static_cast<double>(full_n) *
        SplitVectorMaxElements(tail_m * core_n, cores);
    if (tail_n) critical += static_cast<double>(full_m) *
        SplitVectorMaxElements(core_m * tail_n, cores);
    if (tail_m && tail_n)
        critical += SplitVectorMaxElements(tail_m * tail_n, cores);
    return critical;
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
    double operand_bytes;
};

std::vector<std::uint64_t> TileValues(const std::vector<std::uint64_t>& supplied,
                                      std::uint64_t shape, bool core_tile,
                                      std::uint64_t aic_count) {
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
        if (!v || v % 16 || v > shape)
            throw std::invalid_argument("MILP tile domain requires aligned values within shape");
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
    const auto cm = TileValues(domain.single_core_m, w.m, true, h.aic_count);
    const auto cn = TileValues(domain.single_core_n, w.n, true, h.aic_count);
    const auto bm = TileValues(domain.base_m, w.m, false, h.aic_count);
    const auto bn = TileValues(domain.base_n, w.n, false, h.aic_count);
    std::vector<Geometry> result;
    // Spatial feasibility: positive 16-aligned extents within the shape,
    // baseM <= singleCoreM and baseN <= singleCoreN. Other capacity tests
    // below discard infeasible geometries before creating MILP variables.
    for (const auto m : cm) for (const auto n : cn)
        for (const auto a : bm) for (const auto b : bn) {
            if (a > m || b > n) continue;
            const auto pm = AxisPackets(w.m, m, a);
            const auto pn = AxisPackets(w.n, n, b);
            const auto outputs = CeilDiv(w.m, m) * CeilDiv(w.n, n);
            const auto packets = pm * pn;
            const double bytes = 2.0 * static_cast<double>(w.k) *
                (static_cast<double>(w.m) * pn + static_cast<double>(w.n) * pm);
            result.push_back({m, n, a, b, outputs, packets, bytes});
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
    if (CanExpandEvents(w, decision)) {
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
                                 const std::vector<Geometry>& geometries) {
    const double conversion_bytes = w.k > 65535 ? 2.0 * w.m * w.k : 0.0;
    // mat_mul_v3 reserves the entire converted A matrix, independently of
    // spatial tiling or occupied Cube cores.
    if (kRpcBytes + conversion_bytes > h.workspace_limit_bytes)
        return std::nullopt;

    // For a fixed geometry, the BASE objective is nonincreasing in both
    // baseK=16*b and occupied cores c. Larger b reduces the K-packet count;
    // larger c reduces Cube time. Neither has a penalty elsewhere in the
    // reduced BASE objective. Set each to its largest feasible value, then
    // solve only the one-hot geometry master. This is an exact elimination,
    // not a relaxation of the previous per-geometry integer program.
    Model master;
    struct Choice {
        Geometry geometry;
        int y;
        std::uint64_t base_k, cores;
        double stage_us;
    };
    std::vector<Choice> choices;
    std::vector<std::pair<int, double>> select_one;
    const double mn = OutputElements(w);
    const double compute_alpha = Flops(w) / r.cube.rate_per_us;
    for (const auto& g : geometries) {
        if (8.0 * g.base_m * g.base_n > mem.l0c_bytes) continue;
        const auto bmax = BaseKUnitsUpper(w, g, mem);
        const auto cmax = std::min(h.aic_count, g.output_tiles);
        if (!bmax || !cmax) continue;
        const auto base_k = 16 * bmax;
        const double packets = static_cast<double>(CeilDiv(w.k, base_k));
        const double cube = compute_alpha / static_cast<double>(cmax);
        const double hbm = g.operand_bytes / r.read.rate_per_us +
            2.0 * mn / r.write.rate_per_us +
            static_cast<double>(g.output_tiles) * r.write.fixed_latency_us +
            2.0 * static_cast<double>(g.spatial_packets) * packets *
                r.read.fixed_latency_us;
        const double stage = std::max(cube, hbm);
        const int y = master.Column(stage, 0.0, 1.0, true);
        choices.push_back({g, y, base_k, cmax, stage});
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
    // For a fixed geometry and occupied core count c, the Split-K objective
    // is nonincreasing in stepK s: larger s produces fewer K slices P,
    // reducing GM service without changing Cube or vector service. The exact
    // largest legal step is min(L1 bound, floor((K/128-1)/(c-1))). We still
    // check c=2..cmax because Cube throughput and vector overhead trade off.
    // The master has one binary column per feasible geometry and one row.
    Model master;
    struct Choice {
        Geometry geometry;
        int y;
        std::uint64_t step, cores;
        double local_us;
    };
    std::vector<Choice> choices;
    std::vector<std::pair<int, double>> select_one;
    const double mn = OutputElements(w);
    const double compute_alpha = Flops(w) / r.cube.rate_per_us;
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
            const auto step = std::min(smax, (k_packets - 1) / (c - 1));
            if (!step) continue;
            const double p = static_cast<double>(CeilDiv(k_packets, step));
            const double cores = static_cast<double>(c);
            const double outputs = static_cast<double>(g.output_tiles);
            const double cube = compute_alpha / cores;
            const double hbm = g.operand_bytes / r.read.rate_per_us +
                4.0 * p * mn / r.read.rate_per_us +
                (4.0 * p * mn + 2.0 * mn) / r.write.rate_per_us +
                (2.0 * static_cast<double>(g.spatial_packets) *
                     static_cast<double>(k_packets) +
                 outputs * (p - cores + 1.0)) * r.read.fixed_latency_us +
                outputs * (p + 1.0) * r.write.fixed_latency_us;
            const double local = std::max(cube, hbm) +
                outputs * (r.sync.fixed_latency_us + r.vector.fixed_latency_us) +
                (cores - 1.0) *
                    SplitVectorCriticalElements(w, g.core_m, g.core_n, c) /
                    r.vector.rate_per_us;
            if (local < best_us) {
                best_us = local;
                best_step = step;
                best_cores = c;
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
    // Independent evaluation of the two service-time formulas used for
    // master coefficients. It is also the consistency check in Finish().
    const auto r = ReadRates(h);
    (void)FinalizePlan(w, h, d, "reduced_objective", mem);
    if (d.base_m > d.single_core_m || d.base_n > d.single_core_n)
        throw std::invalid_argument("base M/N tile exceeds single-core output tile");
    const auto pm = AxisPackets(w.m, d.single_core_m, d.base_m);
    const auto pn = AxisPackets(w.n, d.single_core_n, d.base_n);
    const double outputs = static_cast<double>(CeilDiv(w.m, d.single_core_m)) *
                           static_cast<double>(CeilDiv(w.n, d.single_core_n));
    const double spatial_packets = static_cast<double>(pm) * pn;
    const double operand_bytes = 2.0 * static_cast<double>(w.k) *
        (static_cast<double>(w.m) * pn + static_cast<double>(w.n) * pm);
    const double mn = OutputElements(w);
    const double common = CommonUs(w, r);
    if (d.family == KernelFamily::Base) {
        const double p = static_cast<double>(CeilDiv(w.k, d.base_k));
        const double cube = Flops(w) /
            (static_cast<double>(d.used_cores) * r.cube.rate_per_us);
        const double hbm = operand_bytes / r.read.rate_per_us +
            2.0 * mn / r.write.rate_per_us +
            outputs * r.write.fixed_latency_us +
            2.0 * spatial_packets * p * r.read.fixed_latency_us;
        return common + std::max(cube, hbm);
    }
    if (d.base_k != 128)
        throw std::invalid_argument("reduced Split-K fixes baseK=128");
    const double p = static_cast<double>(CeilDiv(w.k, d.single_core_k));
    const double c = static_cast<double>(d.used_cores);
    const double cube = Flops(w) / (c * r.cube.rate_per_us);
    const double hbm = operand_bytes / r.read.rate_per_us +
        4.0 * p * mn / r.read.rate_per_us +
        (4.0 * p * mn + 2.0 * mn) / r.write.rate_per_us +
        (2.0 * spatial_packets * static_cast<double>(CeilDiv(w.k, d.base_k)) +
         outputs * (p - c + 1.0)) * r.read.fixed_latency_us +
        outputs * (p + 1.0) * r.write.fixed_latency_us;
    return common + std::max(cube, hbm) +
        outputs * (r.sync.fixed_latency_us + r.vector.fixed_latency_us) +
        (c - 1.0) * SplitVectorCriticalElements(
            w, d.single_core_m, d.single_core_n, d.used_cores) /
            r.vector.rate_per_us;
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
    const auto geometries = Geometries(w, h, domain);
    MipResult result;
    result.base = SolveBase(w, h, mem, rates, geometries);
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
