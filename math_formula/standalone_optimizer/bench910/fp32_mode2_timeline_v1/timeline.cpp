// Diagnostic FP32 mode-2 event schedule. No full-kernel time is fitted here.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kMte2 = 0, kL0A = 1, kL0B = 2, kCube = 3, kFix = 4;
constexpr int kStages = 5;
constexpr double kL0ABytesPerUs = 460800.0;
constexpr double kL0BBytesPerUs = 230400.0;
constexpr double kCubeFlopsPerUs = 3600000.0;
constexpr double kPlainStoreLatencyUs = 0.2432;
constexpr double kPlainStoreBytesPerUs = 34190.0;
constexpr double kAtomicStoreBytesPerUs = 91000.0;
constexpr double kLaunchUs = 2.68;

std::uint64_t CeilDiv(std::uint64_t a, std::uint64_t b) {
    return a / b + (a % b != 0);
}

struct Node {
    double finish = 0.0;
    double duration = 0.0;
    int stage = 0;
    int predecessor = -1;
};

struct PacketResult {
    double runtime_us = 0.0;
    std::array<double, kStages> busy_us{};
    std::array<double, kStages> critical_us{};
    std::size_t event_count = 0;
};

// Each stage is an exclusive engine. A node starts after its predecessors and
// the preceding node on its engine. The recorded predecessor traces one
// longest path through this explicitly stated dependency graph.
class Schedule {
public:
    int Add(int stage, double duration, const std::vector<int>& after = {}) {
        if (duration < 0 || !std::isfinite(duration))
            throw std::invalid_argument("invalid event duration");
        double start = available_[stage];
        int predecessor = last_[stage];
        for (int index : after) {
            if (index < 0) continue;
            if (nodes_.at(index).finish > start) {
                start = nodes_[index].finish;
                predecessor = index;
            }
        }
        const int index = static_cast<int>(nodes_.size());
        nodes_.push_back({start + duration, duration, stage, predecessor});
        available_[stage] = start + duration;
        last_[stage] = index;
        busy_[stage] += duration;
        return index;
    }

    PacketResult Finish() const {
        PacketResult result;
        result.busy_us = busy_;
        result.event_count = nodes_.size();
        int last = -1;
        for (int stage = 0; stage < kStages; ++stage) {
            if (available_[stage] > result.runtime_us) {
                result.runtime_us = available_[stage];
                last = last_[stage];
            }
        }
        while (last >= 0) {
            const auto& node = nodes_[last];
            result.critical_us[node.stage] += node.duration;
            last = node.predecessor;
        }
        return result;
    }

private:
    std::vector<Node> nodes_;
    std::array<double, kStages> available_{};
    std::array<double, kStages> busy_{};
    std::array<int, kStages> last_{-1, -1, -1, -1, -1};
};

struct Input {
    std::string id;
    std::uint64_t pitch = 0, m = 0, n = 0, k = 0;
    std::uint64_t packet_k = 0, base_m = 0, base_n = 0, base_k = 0;
    double probe_us = 0.0;
    // "mn" = M-block outer/N-block inner; "nm" reverses that order.
    std::string order;
};

PacketResult OnePacket(const Input& in, std::uint64_t packet_k,
                       double mte2_us, bool atomic) {
    Schedule schedule;
    const auto mb = CeilDiv(in.m, in.base_m);
    const auto nb = CeilDiv(in.n, in.base_n);
    const auto qk = CeilDiv(packet_k, in.base_k);

    // Source-shaped MTE2 stream: one A panel then one B base-column copy
    // for each N block. The independent probe measures the whole stream;
    // distribute its duration by requested bytes without fitting kernel time.
    const double a_bytes = 4.0 * in.m * packet_k;
    const double b_bytes = 4.0 * in.n * packet_k;
    const double total_bytes = a_bytes + b_bytes;
    const int a_ready = schedule.Add(kMte2, mte2_us * a_bytes / total_bytes);
    std::vector<int> b_ready(nb);
    for (std::uint64_t ni = 0; ni < nb; ++ni) {
        const auto n_extent = std::min(in.base_n, in.n - ni * in.base_n);
        b_ready[ni] = schedule.Add(kMte2,
            mte2_us * (4.0 * n_extent * packet_k) / total_bytes);
    }

    std::vector<int> cube_events;
    std::vector<int> output_events;
    cube_events.reserve(mb * nb * qk);
    output_events.reserve(mb * nb);
    auto output_tile = [&](std::uint64_t mi, std::uint64_t ni) {
        const auto m_extent = std::min(in.base_m, in.m - mi * in.base_m);
        const auto n_extent = std::min(in.base_n, in.n - ni * in.base_n);
        const std::size_t tile_index = output_events.size();
        int previous_cube = -1;
        for (std::uint64_t ki = 0; ki < qk; ++ki) {
            const auto k_extent = std::min(in.base_k, packet_k - ki * in.base_k);
            std::vector<int> a_deps{a_ready};
            std::vector<int> b_deps{b_ready[ni]};
            // Two L0A/L0B buffers: a new load cannot overwrite the buffer
            // until the Cube use two pieces earlier has completed.
            if (cube_events.size() >= 2) {
                a_deps.push_back(cube_events[cube_events.size() - 2]);
                b_deps.push_back(cube_events[cube_events.size() - 2]);
            }
            const int a_load = schedule.Add(kL0A,
                4.0 * m_extent * k_extent / kL0ABytesPerUs, a_deps);
            const int b_load = schedule.Add(kL0B,
                4.0 * n_extent * k_extent / kL0BBytesPerUs, b_deps);
            std::vector<int> cube_deps{a_load, b_load, previous_cube};
            // Two L0C buffers: the first Cube use of output tile t must
            // wait for Fixpipe to release the output buffer from t-2.
            if (ki == 0 && tile_index >= 2)
                cube_deps.push_back(output_events[tile_index - 2]);
            previous_cube = schedule.Add(kCube,
                2.0 * m_extent * n_extent * k_extent / kCubeFlopsPerUs,
                cube_deps);
            cube_events.push_back(previous_cube);
        }
        const double output_bytes = 4.0 * m_extent * n_extent;
        const double output_us = atomic ?
            output_bytes / kAtomicStoreBytesPerUs :
            kPlainStoreLatencyUs + output_bytes / kPlainStoreBytesPerUs;
        output_events.push_back(schedule.Add(kFix, output_us, {previous_cube}));
    };

    if (in.order == "mn") {
        for (std::uint64_t mi = 0; mi < mb; ++mi)
            for (std::uint64_t ni = 0; ni < nb; ++ni)
                output_tile(mi, ni);
    } else if (in.order == "nm") {
        for (std::uint64_t ni = 0; ni < nb; ++ni)
            for (std::uint64_t mi = 0; mi < mb; ++mi)
                output_tile(mi, ni);
    } else {
        throw std::invalid_argument("order must be mn or nm");
    }
    return schedule.Finish();
}

struct Result {
    double total_us = kLaunchUs, first_us = 0.0, steady_us = 0.0;
    std::array<double, kStages> busy_us{};
    std::array<double, kStages> critical_us{};
    std::size_t event_count = 0;
};

Result Evaluate(const Input& in) {
    if (!in.m || !in.n || !in.k || !in.packet_k || !in.base_m ||
        !in.base_n || !in.base_k || in.probe_us <= 0 ||
        !std::isfinite(in.probe_us))
        throw std::invalid_argument("invalid shape, tile, or probe duration");
    const auto packets = CeilDiv(in.k, in.packet_k);
    const double mte2_full_us = in.probe_us / packets;
    Result result;
    for (std::uint64_t p = 0; p < packets; ++p) {
        const auto current_k = std::min(in.packet_k, in.k - p * in.packet_k);
        const auto packet = OnePacket(in, current_k,
            mte2_full_us * current_k / in.packet_k, p > 0);
        result.total_us += packet.runtime_us;
        if (p == 0) result.first_us = packet.runtime_us;
        else result.steady_us += packet.runtime_us;
        result.event_count += packet.event_count;
        for (int stage = 0; stage < kStages; ++stage) {
            result.busy_us[stage] += packet.busy_us[stage];
            result.critical_us[stage] += packet.critical_us[stage];
        }
    }
    if (packets > 1) result.steady_us /= packets - 1;
    return result;
}

}  // namespace

int main() {
    std::cout << "id,pitch,order,predicted_us,first_packet_us,steady_packet_us,"
                 "mte2_critical_us,l0a_critical_us,l0b_critical_us,"
                 "cube_critical_us,fix_critical_us,mte2_busy_us,l0a_busy_us,"
                 "l0b_busy_us,cube_busy_us,fix_busy_us,event_count\n";
    std::cout << std::fixed << std::setprecision(6);
    Input in;
    while (std::cin >> in.id >> in.pitch >> in.m >> in.n >> in.k >>
           in.packet_k >> in.base_m >> in.base_n >> in.base_k >>
           in.probe_us >> in.order) {
        const auto result = Evaluate(in);
        std::cout << in.id << ',' << in.pitch << ',' << in.order << ','
                  << result.total_us << ',' << result.first_us << ','
                  << result.steady_us;
        for (double value : result.critical_us) std::cout << ',' << value;
        for (double value : result.busy_us) std::cout << ',' << value;
        std::cout << ',' << result.event_count << '\n';
    }
}
