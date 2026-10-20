#include <iostream>
#include <vector>
#include "generic_tiling.hpp"

int main() {
    // Define tile size choices for 3 dimensions (e.g., powers of 2 up to 256)
    std::vector<int> candidate_sizes = {8, 16, 32, 64, 128, 256};
    std::vector<std::vector<int>> search_space = {
        candidate_sizes, // Tx
        candidate_sizes, // Ty
        candidate_sizes  // Tz
    };

    // Example Analytical Cost Function (e.g., L1 cache miss estimate + memory traffic)
    // Goal: Minimize cost while keeping tile memory under L1 cache capacity (32 KB)
    auto cost_function = [](const std::vector<int>& tiles) -> double {
        int Tx = tiles[0];
        int Ty = tiles[1];
        int Tz = tiles[2];

        size_t working_set_bytes = (Tx * Ty + Ty * Tz + Tx * Tz) * sizeof(float);
        constexpr size_t L1_CAPACITY_BYTES = 32 * 1024; // 32KB

        // Hard penalty if tiles exceed L1 cache capacity
        if (working_set_bytes > L1_CAPACITY_BYTES) {
            return 1e9 + working_set_bytes; 
        }

        // Mock model: Penalize small tiles (loop overhead) and large tiles (poor locality)
        double loop_overhead = (1024.0 / Tx) * (1024.0 / Ty) * (1024.0 / Tz);
        double cache_misses = (Tx * Ty * Tz) * 0.05;

        return loop_overhead + cache_misses;
    };

    GAConfig config;
    config.population_size = 40;
    config.max_generations = 30;
    config.mutation_rate = 0.2;

    TilingGeneticAlgorithm ga(search_space, cost_function, config);
    Chromosome best_solution = ga.run();

    std::cout << "\nOptimal Tile Choices Found:\n";
    std::cout << "Tx=" << best_solution.tile_sizes[0]
              << ", Ty=" << best_solution.tile_sizes[1]
              << ", Tz=" << best_solution.tile_sizes[2]
              << " | Final Cost: " << best_solution.cost << "\n";

    return 0;
}