#include "event_builder.h"
#include "split_vector_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mmopt {
namespace {

constexpr std::size_t kMaxEvents = 200000;
constexpr std::uint64_t kCubeAlignment = 16;
constexpr std::uint64_t kInputBytes = 2;
constexpr std::uint64_t kOutputBytes = 2;
constexpr std::uint64_t kPartialBytes = 4;

std::uint64_t Required(const Candidate& plan, const std::string& name) {
    const auto found = plan.tile.find(name);
    if (found == plan.tile.end() || found->second == 0)
        throw std::invalid_argument("missing or zero tiling field " + name);
    return found->second;
}

std::uint64_t CeilDiv(std::uint64_t value, std::uint64_t divisor) {
    return value / divisor + (value % divisor != 0);
}

std::uint64_t Align16(std::uint64_t value) {
    const auto blocks = CeilDiv(value, kCubeAlignment);
    if (blocks > std::numeric_limits<std::uint64_t>::max() / kCubeAlignment)
        throw std::overflow_error("aligned tile dimension overflow");
    return blocks * kCubeAlignment;
}

double Work(long double value) {
    if (value < 0.0L || value > static_cast<long double>(std::numeric_limits<double>::max()))
        throw std::overflow_error("event work overflow");
    const double result = static_cast<double>(value);
    if (!std::isfinite(result)) throw std::overflow_error("event work overflow");
    return result;
}

std::string Add(Candidate& plan, const std::string& prefix, const std::string& resource,
                const std::string& primitive, double amount, const std::string& unit,
                std::vector<std::string> after = {}) {
    if (plan.events.size() >= kMaxEvents)
        throw std::length_error("event graph exceeds 200000 events; add recurrence/compression");
    const std::string id = prefix + std::to_string(plan.events.size());
    plan.events.push_back({id, resource, primitive, amount, unit, std::move(after)});
    return id;
}

struct Tile {
    std::uint64_t cores;
    std::uint64_t core_m;
    std::uint64_t core_n;
    std::uint64_t core_k;
    std::uint64_t base_m;
    std::uint64_t base_n;
    std::uint64_t base_k;
};

Tile ReadTile(const Candidate& plan) {
    return {Required(plan, "usedCoreNum"), Required(plan, "singleCoreM"),
            Required(plan, "singleCoreN"), Required(plan, "singleCoreK"),
            Required(plan, "baseM"), Required(plan, "baseN"), Required(plan, "baseK")};
}

// One matmul invocation over an output tile and one K interval. Read packets
// are counted for every base-M/base-N/base-K combination: no implicit cache
// reuse is assumed. The Cube work uses 16-element padded instruction extents.
std::string AddMatmulWork(Candidate& plan, const Tile& tile, std::uint64_t core,
                          std::uint64_t m_extent, std::uint64_t n_extent,
                          std::uint64_t k_extent, const std::string& first_after) {
    std::string last_cube;
    for (std::uint64_t m = 0; m < m_extent;) {
        const auto m_use = std::min(tile.base_m, m_extent - m);
        for (std::uint64_t n = 0; n < n_extent;) {
            const auto n_use = std::min(tile.base_n, n_extent - n);
            for (std::uint64_t k = 0; k < k_extent;) {
                const auto k_use = std::min(tile.base_k, k_extent - k);
                std::vector<std::string> gate;
                if (!first_after.empty()) gate.push_back(first_after);
                const auto a = Add(plan, "a_read_", "hbm", "gm_read",
                                   Work(static_cast<long double>(m_use) * k_use * kInputBytes),
                                   "bytes", gate);
                const auto b = Add(plan, "b_read_", "hbm", "gm_read",
                                   Work(static_cast<long double>(k_use) * n_use * kInputBytes),
                                   "bytes", gate);
                last_cube = Add(plan, "cube_", "cube_" + std::to_string(core), "cube",
                                Work(2.0L * Align16(m_use) * Align16(n_use) * Align16(k_use)),
                                "flop", {a, b});
                k += k_use;
            }
            n += n_use;
        }
        m += m_use;
    }
    return last_cube;
}

void BuildBase(const Workload& workload, const Tile& tile, Candidate& plan,
               const std::string& launch) {
    if (tile.core_k < workload.k)
        throw std::invalid_argument("BASE singleCoreK must cover all K");
    std::uint64_t output_tile = 0;
    for (std::uint64_t m = 0; m < workload.m;) {
        const auto m_use = std::min(tile.core_m, workload.m - m);
        for (std::uint64_t n = 0; n < workload.n;) {
            const auto n_use = std::min(tile.core_n, workload.n - n);
            const auto core = output_tile % tile.cores;
            const auto last_cube = AddMatmulWork(plan, tile, core, m_use, n_use,
                                                  workload.k, launch);
            Add(plan, "output_", "hbm", "gm_write",
                Work(static_cast<long double>(m_use) * n_use * kOutputBytes),
                "bytes", {last_cube});
            ++output_tile;
            n += n_use;
        }
        m += m_use;
    }
}

void BuildDeterministic(const Workload& workload, const Tile& tile, Candidate& plan,
                        const std::string& launch) {
    const auto slices = CeilDiv(workload.k, tile.core_k);
    if (slices < tile.cores)
        throw std::invalid_argument("deterministic Split-K needs at least one K slice per core");
    const long double min_workspace = static_cast<long double>(tile.cores) *
                                      workload.m * workload.n * kPartialBytes;
    if (min_workspace > static_cast<long double>(plan.workspace_bytes))
        throw std::invalid_argument("workspace smaller than logical FP32 partials");

    for (std::uint64_t m = 0; m < workload.m;) {
        const auto m_use = std::min(tile.core_m, workload.m - m);
        for (std::uint64_t n = 0; n < workload.n;) {
            const auto n_use = std::min(tile.core_n, workload.n - n);
            std::vector<std::string> partials(tile.cores, launch);
            std::vector<std::uint64_t> core_begin(tile.cores), core_slices(tile.cores);
            std::uint64_t first_slice = 0;
            for (std::uint64_t core = 0; core < tile.cores; ++core) {
                core_begin[core] = first_slice;
                core_slices[core] = slices / tile.cores + (core < slices % tile.cores);
                first_slice += core_slices[core];
            }
            const auto max_rounds = CeilDiv(slices, tile.cores);
            // Interleave the cores' contiguous K slices for shared-service
            // ordering while preserving each core's source-level ownership.
            for (std::uint64_t local = 0; local < max_rounds; ++local) {
                for (std::uint64_t core = 0; core < tile.cores; ++core) {
                    if (local >= core_slices[core]) continue;
                    const auto slice = core_begin[core] + local;
                    const auto k_start = slice * tile.core_k;
                    const auto k_use = std::min(tile.core_k, workload.k - k_start);
                    std::string previous = partials[core];
                    if (local != 0) {
                        previous = Add(plan, "partial_read_", "hbm", "gm_read",
                                       Work(static_cast<long double>(m_use) * n_use * kPartialBytes),
                                       "bytes", {previous});
                    }
                    const auto last_cube = AddMatmulWork(plan, tile, core, m_use, n_use,
                                                          k_use, previous);
                    previous = Add(plan, "partial_write_", "hbm", "gm_write",
                                   Work(static_cast<long double>(m_use) * n_use * kPartialBytes),
                                   "bytes", {last_cube});
                    partials[core] = previous;
                }
            }
            const auto sync = Add(plan, "sync_", "sync", "sync", 0.0, "count", partials);
            const auto read = Add(plan, "reduce_read_", "hbm", "gm_read",
                                  Work(static_cast<long double>(tile.cores) * m_use * n_use *
                                       kPartialBytes), "bytes", {sync});
            // The mixed kernel dispatches two AIVs per occupied AIC. Model
            // concurrent balanced shards of this output tile rather than
            // serializing all (cores-1)*elements additions on one AIV.
            const auto elements = m_use * n_use;
            const auto shards = SplitVectorShardCount(tile.cores, elements);
            if (plan.events.size() >= kMaxEvents ||
                shards > kMaxEvents - plan.events.size() - 1)
                throw std::length_error("event graph exceeds 200000 events; add recurrence/compression");
            std::vector<std::string> reductions;
            reductions.reserve(shards);
            for (std::uint64_t lane = 0; lane < shards; ++lane) {
                const auto lane_elements = elements / shards + (lane < elements % shards);
                reductions.push_back(Add(
                    plan, "reduce_", "aiv_" + std::to_string(lane), "vector_add",
                    Work(static_cast<long double>(tile.cores - 1) * lane_elements),
                    "ops", {read}));
            }
            Add(plan, "output_", "hbm", "gm_write",
                Work(static_cast<long double>(elements) * kOutputBytes),
                "bytes", reductions);
            n += n_use;
        }
        m += m_use;
    }
}

}  // namespace

Candidate BuildMatmulEvents(const Workload& workload, Candidate plan) {
    if (workload.m == 0 || workload.n == 0 || workload.k == 0 ||
        (workload.dtype != "fp16" && workload.dtype != "bf16"))
        throw std::invalid_argument("event builder supports positive FP16/BF16 workloads only");
    if (workload.a_layout != Layout::ND || workload.b_layout != Layout::ND ||
        workload.c_layout != Layout::ND || workload.trans_a || workload.trans_b ||
        workload.has_bias)
        throw std::invalid_argument("event builder supports ND/ND/ND, NN, no bias only");
    if (plan.family != "BASE" && plan.family != "DETERMINISTIC_SPLIT_K")
        throw std::invalid_argument("unsupported implementation family for event builder");
    static const std::set<std::string> supported_tiles = {
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK"};
    for (const auto& [name, value] : plan.tile) {
        (void)value;
        if (supported_tiles.find(name) == supported_tiles.end())
            throw std::invalid_argument("unmodeled tiling field " + name);
    }
    const auto tile = ReadTile(plan);
    if (plan.family == "DETERMINISTIC_SPLIT_K" && tile.core_k % tile.base_k != 0)
        throw std::invalid_argument("Split-K singleCoreK must be a multiple of baseK");
    if (plan.nd2nz_b)
        throw std::invalid_argument("B ND2NZ conversion events are not modeled");
    if (plan.nd2nz_a) {
        const long double a_bytes = static_cast<long double>(workload.m) * workload.k * kInputBytes;
        if (a_bytes > static_cast<long double>(plan.workspace_bytes))
            throw std::invalid_argument("workspace smaller than converted A matrix");
    }
    plan.events.clear();
    const auto launch = Add(plan, "launch_", "launch", "launch", 0.0, "count");
    std::string matmul_gate = launch;
    if (plan.nd2nz_a) {
        // The mixed kernel converts A once into workspace on AIVs and
        // synchronizes before AIC computation. The three aggregate events
        // represent source read, vector format work, and workspace write.
        const auto elements = Work(static_cast<long double>(workload.m) * workload.k);
        const auto bytes = Work(static_cast<long double>(workload.m) * workload.k * kInputBytes);
        const auto read = Add(plan, "convert_a_read_", "hbm", "gm_read",
                              bytes, "bytes", {launch});
        const auto convert = Add(plan, "convert_a_", "aiv", "nd2nz",
                                 elements, "elements", {read});
        const auto write = Add(plan, "convert_a_write_", "hbm", "gm_write",
                               bytes, "bytes", {convert});
        matmul_gate = Add(plan, "convert_sync_", "sync", "sync", 0.0,
                          "count", {write});
    }
    if (plan.family == "BASE") BuildBase(workload, tile, plan, matmul_gate);
    else BuildDeterministic(workload, tile, plan, matmul_gate);
    return plan;
}

}  // namespace mmopt
