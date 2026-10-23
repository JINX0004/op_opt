#include "optimizer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mmopt_cli <candidate-manifest>\n";
        return 2;
    }
    try {
        const auto manifest = mmopt::ReadManifest(argv[1]);
        const auto result = mmopt::Optimize(manifest.workload, manifest.hardware,
                                           manifest.candidates);
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "supplied=" << result.supplied << " valid=" << result.ranking.size()
                  << " rejected=" << result.rejected.size() << '\n';
        for (const auto& rejected : result.rejected)
            std::cout << "reject id=" << rejected.id << " reason=" << rejected.reason << '\n';
        for (const auto& candidate : result.ranking)
            std::cout << "rank id=" << candidate.id << " family=" << candidate.family
                      << " key=" << candidate.tiling_key
                      << " runtime_us=" << candidate.score.runtime_us << '\n';
        if (!result.has_winner()) {
            std::cerr << "no valid candidate\n";
            return 3;
        }
        const auto& winner = result.winner();
        std::cout << "winner=" << winner.id << " predicted_us="
                  << winner.score.runtime_us
                  << " critical_path_events=" << winner.score.critical_path.size()
                  << " critical_path_prefix=";
        const auto critical_shown = std::min<std::size_t>(12, winner.score.critical_path.size());
        for (std::size_t i = 0; i < critical_shown; ++i) {
            if (i) std::cout << ',';
            std::cout << winner.score.critical_path[i];
        }
        if (critical_shown < winner.score.critical_path.size()) std::cout << ",...";
        std::cout << '\n';
        const auto shown = std::min<std::size_t>(20, winner.score.timeline.size());
        for (std::size_t i = 0; i < shown; ++i) {
            const auto& event = winner.score.timeline[i];
            std::cout << "event id=" << event.id << " start_us=" << event.start_us
                      << " finish_us=" << event.finish_us << '\n';
        }
        if (shown < winner.score.timeline.size())
            std::cout << "... " << (winner.score.timeline.size() - shown)
                      << " additional events omitted\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
