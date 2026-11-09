#include "hardware_profile.h"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int Mode(mmopt::KernelFamily family) {
    switch (family) {
        case mmopt::KernelFamily::Base: return 0;
        case mmopt::KernelFamily::SingleCoreSplitKMode2: return 2;
        case mmopt::KernelFamily::DeterministicSplitK: return 3;
        case mmopt::KernelFamily::SingleCoreSplitKMode5: return 5;
        case mmopt::KernelFamily::SingleCoreSplitKMode6: return 6;
    }
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    const bool source_base_mode = argc == 2 &&
        std::string(argv[1]) == "--source-base";
    if (argc != 1 && !source_base_mode) {
        std::cerr << "usage: mmopt_selector [--source-base]\n";
        return 2;
    }
    const char* row_profile = std::getenv("MMV3_ONLINE_ROW_READ_PROFILE");
    const std::string read_profile = row_profile != nullptr ? row_profile : "";
    const auto default_hardware = mmopt::Calibrated910B3();
    const auto narrow_hardware = mmopt::NarrowB910B3();
    const auto full_hardware = mmopt::RowAware910B3();
    const auto bf16_hardware = mmopt::ExperimentalBf16910B3();
    const auto fp32_hardware = mmopt::ExperimentalFp32910B3();
    const auto domain = mmopt::SourceCompatibleDomain();
    std::cout << "m,n,k,dtype,mode,objective_us,used_cores,singleCoreM,singleCoreN,singleCoreK,baseM,baseN,baseK,stepM,stepN,stepK";
    if (source_base_mode)
        std::cout << ",optimized_base_us,source_base_us";
    std::cout << '\n';
    std::cout << std::fixed << std::setprecision(5);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream input(line);
        std::uint64_t m = 0, n = 0, k = 0;
        std::string dtype;
        if (!(input >> m >> n >> k >> dtype) ||
            (dtype != "fp16" && dtype != "bf16" && dtype != "fp32")) {
            std::cerr << "invalid input row: " << line << '\n';
            return 2;
        }
        try {
            const auto& hardware = dtype == "fp32" ? fp32_hardware :
                dtype == "bf16" ? bf16_hardware :
                read_profile == "full" ? full_hardware :
                read_profile == "1" ? narrow_hardware : default_hardware;
            auto result = mmopt::SolveReducedMip({m, n, k, dtype}, hardware, {}, domain);
            const double optimized_base_us = result.base ? result.base->surrogate_us : -1.0;
            double source_base_us = -1.0;
            if (source_base_mode) {
                mmopt::PlanDecision source;
                source.family = mmopt::KernelFamily::Base;
                source.single_core_k = k;
                if (!(input >> source.used_cores >> source.single_core_m >>
                      source.single_core_n >> source.base_m >> source.base_n >>
                      source.base_k))
                    throw std::invalid_argument("source BASE needs cores coreM coreN baseM baseN baseK");
                mmopt::CoreMemory source_memory;
                source_memory.base_l0c_buffers = 1;
                source_base_us = mmopt::ReducedObjectiveUs(
                    {m, n, k, dtype}, hardware, source, source_memory, true);
                if (result.base) {
                    result.base->decision = source;
                    result.base->surrogate_us = source_base_us;
                }
                result.winner.reset();
                for (const auto* plan : {&result.base, &result.deterministic_split_k,
                                         &result.single_core_split_k_mode2,
                                         &result.single_core_split_k_mode5,
                                         &result.single_core_split_k_mode6})
                    if (*plan && (!result.winner ||
                                  (*plan)->surrogate_us < result.winner->surrogate_us))
                        result.winner = *plan;
            }
            if (!result.winner) {
                std::cerr << "no feasible MIP plan: " << line << '\n';
                return 3;
            }
            const auto& plan = *result.winner;
            const auto& d = plan.decision;
            std::cout << m << ',' << n << ',' << k << ',' << dtype << ','
                      << Mode(d.family) << ',' << plan.surrogate_us << ','
                      << d.used_cores << ',' << d.single_core_m << ','
                      << d.single_core_n << ',' << d.single_core_k << ','
                      << d.base_m << ',' << d.base_n << ',' << d.base_k << ','
                      << d.step_m << ',' << d.step_n << ',' << d.step_k;
            if (source_base_mode)
                std::cout << ',' << optimized_base_us << ',' << source_base_us;
            std::cout << '\n';
        } catch (const std::exception& error) {
            std::cerr << "MIP failed for " << line << ": " << error.what() << '\n';
            return 4;
        }
    }
}
