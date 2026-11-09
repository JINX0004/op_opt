#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mmopt {

// mat_mul_v3's deterministic Split-K kernel launches two AIVs for each
// occupied AIC. This standalone model balances each output tile's reduction
// elements across those AIVs; source-specific row/NZ partitioning remains
// outside its current geometry description.
constexpr std::uint64_t kSplitAivsPerAic = 2;

inline std::uint64_t SplitVectorShardCount(std::uint64_t cores,
                                           std::uint64_t elements) {
    if (!cores || !elements)
        throw std::invalid_argument("Split-K vector partition requires positive cores and elements");
    if (cores > std::numeric_limits<std::uint64_t>::max() / kSplitAivsPerAic)
        throw std::overflow_error("Split-K AIV count overflow");
    return std::min(kSplitAivsPerAic * cores, elements);
}

inline std::uint64_t SplitVectorMaxElements(std::uint64_t elements,
                                            std::uint64_t cores) {
    const auto shards = SplitVectorShardCount(cores, elements);
    return elements / shards + (elements % shards != 0);
}

}  // namespace mmopt
