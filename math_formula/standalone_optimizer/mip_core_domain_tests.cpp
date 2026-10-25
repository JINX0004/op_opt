#include "mip_solver.h"

#include <cassert>
#include <iostream>

int main() {
    const mmopt::Hardware hardware{
        20, 32ULL * 1024 * 1024,
        {{"launch", {"count", 0.2, 1.0}},
         {"gm_read", {"bytes", 0.005, 1600000.0}},
         {"gm_write", {"bytes", 1.0, 1600000.0}},
         {"cube", {"flop", 0.0, 1000000000.0}},
         {"vector_add", {"ops", 0.0, 1000000.0}},
         {"nd2nz", {"elements", 0.0, 1000000.0}},
         {"sync", {"count", 0.1, 1.0}}}};
    const mmopt::Workload workload{2048, 16, 128, "fp16"};

    // The full M axis is in the default single-core domain even though
    // baseM remains a small L0 tile. The high fixed output-write cost makes
    // this larger single-core tile preferable in the reduced model.
    const auto expanded = mmopt::SolveReducedMip(workload, hardware);
    assert(expanded.base);
    assert(expanded.base->decision.single_core_m == workload.m);
    assert(expanded.base->decision.base_m < expanded.base->decision.single_core_m);

    mmopt::MipTileDomain old_core_grid;
    old_core_grid.single_core_m = {16, 32, 64, 128, 256};
    const auto restricted = mmopt::SolveReducedMip(workload, hardware, {},
                                                   old_core_grid);
    assert(restricted.base);
    assert(restricted.base->decision.single_core_m <= 256);
    assert(expanded.base->surrogate_us < restricted.base->surrogate_us);

    // The same freedom applies to N; its L0 base tile remains independent.
    const mmopt::Workload wide{16, 2048, 128, "fp16"};
    const auto expanded_n = mmopt::SolveReducedMip(wide, hardware);
    assert(expanded_n.base);
    assert(expanded_n.base->decision.single_core_n == wide.n);
    assert(expanded_n.base->decision.base_n < expanded_n.base->decision.single_core_n);

    mmopt::MipTileDomain old_n_grid;
    old_n_grid.single_core_n = {16, 32, 64, 128, 256};
    const auto restricted_n = mmopt::SolveReducedMip(wide, hardware, {},
                                                     old_n_grid);
    assert(restricted_n.base);
    assert(expanded_n.base->surrogate_us < restricted_n.base->surrogate_us);

    // An intermediate core extent matters when Split-K workspace excludes
    // the much larger shape/AIC partition tiles.
    auto tight_workspace = hardware;
    tight_workspace.workspace_limit_bytes = 20ULL * 1024 * 1024 + 131072;
    const mmopt::Workload tall{102400, 16, 256, "fp16"};
    const auto intermediate = mmopt::SolveReducedMip(tall, tight_workspace);
    assert(intermediate.deterministic_split_k);
    assert(intermediate.deterministic_split_k->decision.single_core_m == 512);
    assert(intermediate.deterministic_split_k->decision.base_m < 512);

    const auto small_grid = mmopt::SolveReducedMip(tall, tight_workspace, {},
                                                  old_core_grid);
    assert(small_grid.deterministic_split_k);
    assert(intermediate.deterministic_split_k->surrogate_us <
           small_grid.deterministic_split_k->surrogate_us);

    // mat_mul_v3 BASE assigns spatial output tiles to multiple Cube cores.
    // Long-K ND input needs a full A ND->NZ workspace before that kernel;
    // with enough workspace, this large shape can occupy all 20 AICs.
    auto large_hardware = hardware;
    large_hardware.workspace_limit_bytes = 64ULL * 1024 * 1024 * 1024;
    large_hardware.primitives["gm_read"].rate_per_us = 1.0e9;
    large_hardware.primitives["gm_write"].rate_per_us = 1.0e9;
    large_hardware.primitives["cube"].rate_per_us = 1.0e7;
    const mmopt::Workload large_spatial{64000, 48000, 256000, "fp16"};
    const mmopt::MipTileDomain spatial_grid{{3200}, {48000}, {128}, {128}};
    auto small_workspace = large_hardware;
    small_workspace.workspace_limit_bytes = 32ULL * 1024 * 1024;
    assert(!mmopt::SolveReducedMip(large_spatial, small_workspace, {},
                                   spatial_grid).base);
    const auto multicore = mmopt::SolveReducedMip(large_spatial, large_hardware,
                                                  {}, spatial_grid);
    assert(multicore.base);
    assert(multicore.base->decision.used_cores == 20);
    assert(multicore.base->candidate.nd2nz_a);
    assert(multicore.base->candidate.workspace_bytes > 32ULL * 1024 * 1024);

    std::cout << "mmopt_mip_core_domain_tests passed\n";
}
