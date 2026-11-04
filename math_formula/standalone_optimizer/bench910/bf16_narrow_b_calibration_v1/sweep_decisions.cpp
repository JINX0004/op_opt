#include "model/hardware_profile.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int Mode(mmopt::KernelFamily family) {
    switch (family) {
        case mmopt::KernelFamily::Base: return 0;
        case mmopt::KernelFamily::DeterministicSplitK: return 3;
        case mmopt::KernelFamily::SingleCoreSplitKMode2: return 2;
        case mmopt::KernelFamily::SingleCoreSplitKMode5: return 5;
        case mmopt::KernelFamily::SingleCoreSplitKMode6: return 6;
    }
    throw std::runtime_error("unknown family");
}

}  // namespace

int main() {
    const auto generic = mmopt::ExperimentalBf16910B3();
    auto narrow = generic;
    const auto fp16 = mmopt::NarrowB910B3();
    for (const auto* name : {"base_mte2_b_narrow_rows_one",
                             "base_mte2_b_narrow_rows_full"})
        narrow.primitives[name] = fp16.primitives.at(name);
    std::cout << "m,n,k,generic_mode,generic_us,narrow_mode,narrow_us\n";
    std::uint64_t m, n, k;
    while (std::cin >> m >> n >> k) {
        try {
            const mmopt::Workload workload{m, n, k, "bf16"};
            const auto a = mmopt::SolveReducedMip(
                workload, generic, {}, mmopt::SourceCompatibleDomain());
            const auto b = mmopt::SolveReducedMip(
                workload, narrow, {}, mmopt::SourceCompatibleDomain());
            if (!a.winner || !b.winner)
                throw std::runtime_error("missing candidate");
            std::cout << m << ',' << n << ',' << k << ','
                      << Mode(a.winner->decision.family) << ','
                      << a.winner->surrogate_us << ','
                      << Mode(b.winner->decision.family) << ','
                      << b.winner->surrogate_us << '\n';
        } catch (const std::exception&) {
            // Both host selectors fall back to the source decision when the
            // shared preflight rejects a shape.
            std::cout << m << ',' << n << ',' << k << ",-1,,-1,\n";
        }
    }
}
