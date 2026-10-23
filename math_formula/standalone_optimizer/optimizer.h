#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mmopt {

enum class Layout { ND, NZ };

struct Workload {
    std::uint64_t m = 0;
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    std::string dtype;
    Layout a_layout = Layout::ND;
    Layout b_layout = Layout::ND;
    Layout c_layout = Layout::ND;
    bool trans_a = false;
    bool trans_b = false;
    bool has_bias = false;
};

// A reusable hardware primitive, independent of implementation family.
// An event using it costs fixed_latency_us + work_units / rate_per_us.
struct PrimitiveTiming {
    std::string unit;
    double fixed_latency_us = 0.0;
    double rate_per_us = 0.0;
};

struct Hardware {
    std::uint64_t aic_count = 0;
    std::uint64_t workspace_limit_bytes = 0;
    std::map<std::string, PrimitiveTiming> primitives;
};

// An event's named resource is exclusive. Events using the same resource are
// ordered as listed. Different resources may overlap when dependencies allow.
// Its amount and unit are evaluated through the shared hardware profile.
struct Event {
    std::string id;
    std::string resource;
    std::string primitive;
    double work_units = 0.0;
    std::string unit;
    std::vector<std::string> after;
};

// The scorer requires a producer-supplied validity marker. Legacy callers
// assert it; FinalizePlan sets it after a narrow standalone preflight. Neither
// route proves that CANN accepted or serialized the plan.
struct Candidate {
    std::string id;
    std::string family;
    std::string tiling_key;
    bool backend_verified = false;
    std::uint64_t workspace_bytes = 0;
    bool nd2nz_a = false;
    bool nd2nz_b = false;
    std::map<std::string, std::uint64_t> tile;
    std::vector<Event> events;
};

struct EventTiming {
    std::string id;
    double start_us = 0.0;
    double finish_us = 0.0;
};

struct Score {
    double runtime_us = 0.0;
    std::vector<EventTiming> timeline;
    std::vector<std::string> critical_path;
};

struct RankedCandidate {
    std::string id;
    std::string family;
    std::string tiling_key;
    Score score;
};

struct Rejection {
    std::string id;
    std::string reason;
};

struct SolveResult {
    std::vector<RankedCandidate> ranking;
    std::vector<Rejection> rejected;
    std::size_t supplied = 0;

    bool has_winner() const { return !ranking.empty(); }
    const RankedCandidate& winner() const { return ranking.front(); }
};

// Exact exhaustive selection over the supplied finite candidate set. The
// caller is responsible for supplying all candidates in the intended domain.
SolveResult Optimize(const Workload& workload, const Hardware& hardware,
                     const std::vector<Candidate>& candidates);

using TileAssignment = std::map<std::string, std::uint64_t>;

struct TileVariable {
    std::string name;
    std::vector<std::uint64_t> values;
};

// One implementation family. The builder must construct a complete model plan
// from every feasible complete assignment, or return nullopt if infeasible.
// The optional lower bound must be <= every completion's predicted runtime.
// Leave it empty to enumerate exactly without pruning.
struct Implementation {
    std::string id;
    std::vector<TileVariable> variables;
    std::function<std::optional<Candidate>(const TileAssignment&)> build;
    std::function<double(const TileAssignment&)> lower_bound_us;
};

struct ImplementationResult {
    std::string id;
    std::optional<RankedCandidate> best;
    std::optional<Candidate> best_plan;
    TileAssignment best_tiling;
    std::size_t scored = 0;
    std::size_t infeasible = 0;
    std::size_t rejected = 0;
    std::size_t pruned_subtrees = 0;
};

struct HierarchicalResult {
    std::vector<ImplementationResult> implementations;
    std::optional<RankedCandidate> best;
    std::optional<Candidate> best_plan;
    TileAssignment best_tiling;
};

// Enumerates implementation families. Within each, solves the finite tiling
// domain by exhaustive search or safe branch-and-bound. There is no time cap:
// with complete builders and valid lower bounds, the returned domain optimum
// is exact. This does not imply a hardware optimum outside the domain.
HierarchicalResult OptimizeImplementations(
    const Workload& workload, const Hardware& hardware,
    const std::vector<Implementation>& implementations);

// Standalone text manifest reader. Throws std::runtime_error on malformed
// input. The manifest's `verified` token is an assertion by its producer,
// not a proof of kernel legality.
struct Manifest {
    Workload workload;
    Hardware hardware;
    std::vector<Candidate> candidates;
};
Manifest ReadManifest(const std::string& path);

}  // namespace mmopt
