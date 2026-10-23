#include "mip_solver.h"

#include "event_builder.h"
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

constexpr double kInf = 1e30;
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
    std::uint64_t core_m, core_n, base_m, base_n;
    std::uint64_t output_tiles, spatial_packets;
    double operand_bytes;
};

std::vector<std::uint64_t> TileValues(const std::vector<std::uint64_t>& supplied,
                                      std::uint64_t shape) {
    std::vector<std::uint64_t> values = supplied;
    if (values.empty()) {
        for (const std::uint64_t v : {16ULL, 32ULL, 64ULL, 128ULL, 256ULL})
            if (v <= shape) values.push_back(v);
        if (shape <= 256) values.push_back(shape);
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

std::vector<Geometry> Geometries(const Workload& w, const MipTileDomain& domain) {
    const auto cm = TileValues(domain.single_core_m, w.m);
    const auto cn = TileValues(domain.single_core_n, w.n);
    const auto bm = TileValues(domain.base_m, w.m);
    const auto bn = TileValues(domain.base_n, w.n);
    std::vector<Geometry> result;
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
    const double m = static_cast<double>(g.base_m), n = static_cast<double>(g.base_n);
    return std::min({w.k / 16,
        Bound(mem.l0a_bytes, 64.0 * m),
        Bound(mem.l0b_bytes, 64.0 * n),
        Bound(mem.l1_bytes, 64.0 * (m + n))});
}

std::uint64_t StepUpper(const Workload& w, const Geometry& g,
                        const CoreMemory& mem) {
    return std::min(w.k / 128,
                    Bound(mem.l1_bytes, 4.0 * 128.0 *
                         (static_cast<double>(g.base_m) + static_cast<double>(g.base_n))));
}

bool FixedTileFits(const Geometry& g, std::uint64_t base_k,
                   const CoreMemory& mem) {
    const double m = static_cast<double>(g.base_m), n = static_cast<double>(g.base_n);
    return 4.0 * m * base_k <= mem.l0a_bytes &&
           4.0 * n * base_k <= mem.l0b_bytes &&
           8.0 * m * n <= mem.l0c_bytes;
}

MipPlan Finish(const Workload& w, const Hardware& h, const CoreMemory& mem,
               PlanDecision decision, const std::string& id,
               double solver_objective) {
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
    if (kRpcBytes + conversion_bytes > h.workspace_limit_bytes)
        return std::nullopt;

    Model model;
    const int stage = model.Column(1.0, 0.0, kInf, false);
    const double q = static_cast<double>(w.k) / 16.0;
    const double compute_alpha = Flops(w) / r.cube.rate_per_us;
    struct Group { Geometry geometry; int y, b, packets, cores; };
    std::vector<Group> groups;
    std::vector<std::pair<int, double>> select_one;
    for (const auto& g : geometries) {
        if (8.0 * g.base_m * g.base_n > mem.l0c_bytes) continue;
        const auto bmax = BaseKUnitsUpper(w, g, mem);
        const auto cmax = std::min(h.aic_count, g.output_tiles);
        if (!bmax || !cmax) continue;
        const int y = model.Column(0.0, 0.0, 1.0, true);
        const int b = model.Column(0.0, 0.0, static_cast<double>(bmax), true);
        const int packets = model.Column(0.0, 0.0, q, true);
        const int cores = model.Column(0.0, 0.0, static_cast<double>(cmax), true);
        groups.push_back({g, y, b, packets, cores});
        select_one.emplace_back(y, 1.0);
        model.Row(0.0, kInf, {{b, 1.0}, {y, -1.0}});
        model.Row(-kInf, 0.0, {{b, 1.0}, {y, -static_cast<double>(bmax)}});
        model.Row(0.0, kInf, {{packets, 1.0}, {y, -1.0}});
        model.Row(-kInf, 0.0, {{packets, 1.0}, {y, -q}});
        model.Row(0.0, kInf, {{cores, 1.0}, {y, -1.0}});
        model.Row(-kInf, 0.0, {{cores, 1.0}, {y, -static_cast<double>(cmax)}});
        for (std::uint64_t j = 1; j <= bmax; ++j) {
            const double v = static_cast<double>(j);
            model.Row(0.0, kInf,
                      {{packets, 1.0}, {b, q / (v * v)},
                       {y, -2.0 * q / v}});
        }
        for (std::uint64_t j = 1; j <= cmax; ++j) {
            const double v = static_cast<double>(j);
            model.Row(0.0, kInf,
                      {{stage, 1.0}, {cores, compute_alpha / (v * v)},
                       {y, -2.0 * compute_alpha / v}});
        }
        const double hbm_constant = g.operand_bytes / r.read.rate_per_us +
            2.0 * OutputElements(w) / r.write.rate_per_us +
            static_cast<double>(g.output_tiles) * r.write.fixed_latency_us;
        const double read_per_packet = 2.0 *
            static_cast<double>(g.spatial_packets) * r.read.fixed_latency_us;
        model.Row(0.0, kInf,
                  {{stage, 1.0}, {packets, -read_per_packet},
                   {y, -hbm_constant}});
    }
    if (groups.empty()) return std::nullopt;
    model.Row(1.0, 1.0, select_one);
    const auto solution = model.Solve();
    if (!solution) return std::nullopt;
    const auto selected = std::find_if(groups.begin(), groups.end(),
        [&](const Group& g) { return Integer(solution->at(g.y), "geometry") == 1; });
    if (selected == groups.end())
        throw std::runtime_error("MILP returned no BASE geometry");
    PlanDecision d;
    d.family = KernelFamily::Base;
    d.used_cores = Integer(solution->at(selected->cores), "usedCoreNum");
    d.single_core_m = selected->geometry.core_m;
    d.single_core_n = selected->geometry.core_n;
    d.base_m = selected->geometry.base_m;
    d.base_n = selected->geometry.base_n;
    d.single_core_k = w.k;
    d.base_k = 16 * Integer(solution->at(selected->b), "baseK units");
    const double objective = solution->at(stage) + CommonUs(w, r);
    return Finish(w, h, mem, d, "MIP_BASE", objective);
}

std::optional<MipPlan> SolveSplit(const Workload& w, const Hardware& h,
                                  const CoreMemory& mem, const Rates& r,
                                  const std::vector<Geometry>& geometries) {
    constexpr std::uint64_t kBaseK = 128;
    const double mn = OutputElements(w);
    const double conversion_bytes = w.k > 65535 ? 2.0 * w.m * w.k : 0.0;
    const double fixed_workspace = kRpcBytes + conversion_bytes;
    if (fixed_workspace > h.workspace_limit_bytes) return std::nullopt;

    Model model;
    const double vector_coefficient = mn / r.vector.rate_per_us;
    const int stage = model.Column(1.0, 0.0, kInf, false);
    const double q = static_cast<double>(w.k) / kBaseK;
    const double scale = std::max(1.0, Flops(w) / r.cube.rate_per_us / 10000.0);
    const double compute_alpha = Flops(w) / r.cube.rate_per_us / scale;
    struct Group { Geometry geometry; int y, step, packets, cores; };
    std::vector<Group> groups;
    std::vector<std::pair<int, double>> select_one;
    const double base_packets = static_cast<double>(CeilDiv(w.k, kBaseK));
    for (const auto& g : geometries) {
        if (!FixedTileFits(g, kBaseK, mem)) continue;
        const auto smax = StepUpper(w, g, mem);
        const auto workspace_cores = Bound(
            static_cast<double>(h.workspace_limit_bytes) - fixed_workspace,
            8.0 * static_cast<double>(g.core_m) * g.core_n);
        const auto cmax = std::min<std::uint64_t>({h.aic_count, workspace_cores,
                                                   CeilDiv(w.k, kBaseK)});
        if (!smax || cmax < 2) continue;
        const int y = model.Column(static_cast<double>(g.output_tiles) *
                                     (r.sync.fixed_latency_us + r.vector.fixed_latency_us) /
                                     scale,
                                   0.0, 1.0, true);
        const int step = model.Column(0.0, 0.0, static_cast<double>(smax), true);
        const int packets = model.Column(0.0, 0.0, q, true);
        const int cores = model.Column(vector_coefficient / scale, 0.0,
                                       static_cast<double>(cmax), true);
        groups.push_back({g, y, step, packets, cores});
        select_one.emplace_back(y, 1.0);
        model.Row(0.0, kInf, {{step, 1.0}, {y, -1.0}});
        model.Row(-kInf, 0.0, {{step, 1.0}, {y, -static_cast<double>(smax)}});
        model.Row(0.0, kInf, {{packets, 1.0}, {y, -1.0}});
        model.Row(-kInf, 0.0, {{packets, 1.0}, {y, -q}});
        model.Row(0.0, kInf, {{cores, 1.0}, {y, -2.0}});
        model.Row(-kInf, 0.0, {{cores, 1.0}, {y, -static_cast<double>(cmax)}});
        for (std::uint64_t j = 1; j <= smax; ++j) {
            const double v = static_cast<double>(j);
            model.Row(0.0, kInf,
                      {{packets, 1.0}, {step, q / (v * v)},
                       {y, -2.0 * q / v}});
        }
        model.Row(0.0, kInf, {{packets, 1.0}, {cores, -1.0}});
        std::vector<std::pair<int, double>> choose_core = {{y, -1.0}};
        std::vector<std::pair<int, double>> core_equality = {{cores, 1.0}};
        std::vector<std::pair<int, double>> step_limit = {{step, 1.0}};
        for (std::uint64_t j = 2; j <= cmax; ++j) {
            const int z = model.Column(0.0, 0.0, 1.0, true);
            choose_core.emplace_back(z, 1.0);
            core_equality.emplace_back(z, -static_cast<double>(j));
            const auto max_step = std::min(smax, (w.k / kBaseK - 1) / (j - 1));
            step_limit.emplace_back(z, -static_cast<double>(max_step));
            const double v = static_cast<double>(j);
            model.Row(0.0, kInf,
                      {{stage, 1.0}, {cores, compute_alpha / (v * v)},
                       {y, -2.0 * compute_alpha / v}});
        }
        model.Row(0.0, 0.0, choose_core);
        model.Row(0.0, 0.0, core_equality);
        model.Row(-kInf, 0.0, step_limit);
        const double outputs = static_cast<double>(g.output_tiles);
        const double hbm_constant = (g.operand_bytes / r.read.rate_per_us +
            2.0 * mn / r.write.rate_per_us +
            (2.0 * static_cast<double>(g.spatial_packets) * base_packets +
             outputs) * r.read.fixed_latency_us +
            outputs * r.write.fixed_latency_us) / scale;
        const double hbm_packet = (4.0 * mn / r.read.rate_per_us +
            4.0 * mn / r.write.rate_per_us +
            outputs * (r.read.fixed_latency_us + r.write.fixed_latency_us)) /
            scale;
        model.Row(0.0, kInf,
                  {{stage, 1.0}, {packets, -hbm_packet},
                   {cores, outputs * r.read.fixed_latency_us / scale},
                   {y, -hbm_constant}});
    }
    if (groups.empty()) return std::nullopt;
    model.Row(1.0, 1.0, select_one);
    const auto solution = model.Solve();
    if (!solution) return std::nullopt;
    const auto selected = std::find_if(groups.begin(), groups.end(),
        [&](const Group& g) { return Integer(solution->at(g.y), "geometry") == 1; });
    if (selected == groups.end())
        throw std::runtime_error("MILP returned no Split-K geometry");
    PlanDecision d;
    d.family = KernelFamily::DeterministicSplitK;
    d.used_cores = Integer(solution->at(selected->cores), "usedCoreNum");
    d.single_core_m = selected->geometry.core_m;
    d.single_core_n = selected->geometry.core_n;
    d.base_m = selected->geometry.base_m;
    d.base_n = selected->geometry.base_n;
    d.base_k = kBaseK;
    d.step_k = Integer(solution->at(selected->step), "stepK");
    d.single_core_k = d.base_k * d.step_k;
    const double objective = solution->at(stage) * scale +
        vector_coefficient * static_cast<double>(d.used_cores) +
        CommonUs(w, r) + static_cast<double>(selected->geometry.output_tiles) *
        (r.sync.fixed_latency_us + r.vector.fixed_latency_us) -
        vector_coefficient;
    return Finish(w, h, mem, d, "MIP_DETERMINISTIC_SPLIT_K", objective);
}

}  // namespace

double ReducedObjectiveUs(const Workload& w, const Hardware& h,
                          const PlanDecision& d, const CoreMemory& mem) {
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
        (c - 1.0) * mn / r.vector.rate_per_us;
}

MipResult SolveReducedMip(const Workload& w, const Hardware& h,
                          const CoreMemory& mem, const MipTileDomain& domain) {
    if (!w.m || !w.n || !w.k || w.m > 1000000 || w.n > 1000000 ||
        w.k > 1000000000 || !h.aic_count || h.aic_count > 20)
        throw std::invalid_argument("unsupported reduced-MILP dimensions or AIC count");
    const auto rates = ReadRates(h);
    const auto geometries = Geometries(w, domain);
    MipResult result;
    result.base = SolveBase(w, h, mem, rates, geometries);
    result.deterministic_split_k = SolveSplit(w, h, mem, rates, geometries);
    if (result.base && result.deterministic_split_k)
        result.winner = result.base->surrogate_us <=
                        result.deterministic_split_k->surrogate_us
                            ? result.base : result.deterministic_split_k;
    else if (result.base) result.winner = result.base;
    else if (result.deterministic_split_k) result.winner = result.deterministic_split_k;
    return result;
}

}  // namespace mmopt
