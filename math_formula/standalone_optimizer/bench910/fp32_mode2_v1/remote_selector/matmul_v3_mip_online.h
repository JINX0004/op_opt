#pragma once

#include "model/hardware_profile.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

namespace optiling::matmul_v3 {

struct OnlineMipChoice {
    int mode = -1;
    double objective_us = 0.0;
    mmopt::PlanDecision tile;
    bool has_tile = false;
};

inline int OnlineMipMode(mmopt::KernelFamily family) {
    switch (family) {
        case mmopt::KernelFamily::Base: return 0;
        case mmopt::KernelFamily::SingleCoreSplitKMode2: return 2;
        case mmopt::KernelFamily::DeterministicSplitK: return 3;
        case mmopt::KernelFamily::SingleCoreSplitKMode5: return 5;
        case mmopt::KernelFamily::SingleCoreSplitKMode6: return 6;
    }
    return -1;
}

inline OnlineMipChoice LookupOnlineMipMode(std::uint64_t m, std::uint64_t n,
                                          std::uint64_t k,
                                          const mmopt::TypeSignature& types,
                                          const mmopt::PlanDecision* source_base = nullptr) {
    // The BF16 profile is still provisional. Keep the source decision as the
    // production path until BF16 source-backed calibration is validated.
    if (types.a == "bf16" && std::getenv("MMV3_ONLINE_BF16_EXPERIMENT") == nullptr)
        return {};
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                           std::string, std::string, std::string, std::string,
                           std::uint64_t, std::uint64_t, std::uint64_t,
                           std::uint64_t, std::uint64_t, std::uint64_t>;
    // Read once per process, like the cached decisions below. This keeps
    // primitive-profile experiments isolated from the established selector.
    static const std::string read_profile = [] {
        const char* enabled = std::getenv("MMV3_ONLINE_ROW_READ_PROFILE");
        return enabled != nullptr ? std::string(enabled) : std::string();
    }();
    static const auto default_hardware = mmopt::Calibrated910B3();
    static const auto narrow_hardware = mmopt::NarrowB910B3();
    static const auto full_hardware = mmopt::RowAware910B3();
    static const auto bf16_hardware = mmopt::ExperimentalBf16910B3();
    static const auto fp32_hardware = mmopt::ExperimentalFp32BaseMte2910B3();
    const auto& hardware = types.a == "fp32" ? fp32_hardware :
        types.a == "bf16" ? bf16_hardware :
        read_profile == "full" ? full_hardware :
        read_profile == "1" ? narrow_hardware : default_hardware;
    static const auto domain = mmopt::SourceCompatibleDomain();
    static std::mutex cache_mutex;
    static std::map<Key, OnlineMipChoice> cache;
    const bool use_source_base = types.a == "bf16" && source_base;
    const Key key{m, n, k, types.a, types.b, types.c, types.accumulator,
                  use_source_base ? source_base->used_cores : 0,
                  use_source_base ? source_base->single_core_m : 0,
                  use_source_base ? source_base->single_core_n : 0,
                  use_source_base ? source_base->base_m : 0,
                  use_source_base ? source_base->base_n : 0,
                  use_source_base ? source_base->base_k : 0};
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(key);
        if (found != cache.end()) return found->second;
    }

    OnlineMipChoice choice;
    try {
        mmopt::Workload workload{};
        workload.m = m;
        workload.n = n;
        workload.k = k;
        workload.dtype = types.a;
        workload.types = types;
        auto result = mmopt::SolveReducedMip(workload, hardware, {}, domain);
        if (use_source_base) {
            // The source has already prepared plain BASE's spatial and K
            // tile before asking for a family. Compare the source-emittable
            // BASE tile, not the solver's hypothetical best baseK/geometry.
            mmopt::CoreMemory source_memory;
            source_memory.base_l0c_buffers = 1;  // Plain BASE sets dbL0c=1.
            const auto score = mmopt::ReducedObjectiveUs(
                workload, hardware, *source_base, source_memory);
            if (result.base) {
                result.base->decision = *source_base;
                result.base->surrogate_us = score;
            }
            result.winner.reset();
            for (const auto* plan : {&result.base, &result.deterministic_split_k,
                                     &result.single_core_split_k_mode2,
                                     &result.single_core_split_k_mode5,
                                     &result.single_core_split_k_mode6}) {
                if (*plan && (!result.winner ||
                              (*plan)->surrogate_us < result.winner->surrogate_us))
                    result.winner = *plan;
            }
        }
        if (result.winner) {
            choice.mode = OnlineMipMode(result.winner->decision.family);
            choice.objective_us = result.winner->surrogate_us;
            choice.tile = result.winner->decision;
            choice.has_tile = true;
        }
    } catch (const std::exception&) {
        // A model/domain failure leaves the source tiler in control.
    }
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(key, choice);
    }
    return choice;
}

inline OnlineMipChoice LookupOnlineMipMode(std::uint64_t m, std::uint64_t n,
                                          std::uint64_t k, const char* dtype,
                                          const mmopt::PlanDecision* source_base = nullptr) {
    return LookupOnlineMipMode(m, n, k, mmopt::SameTypeSignature(dtype),
                               source_base);
}

}  // namespace optiling::matmul_v3
