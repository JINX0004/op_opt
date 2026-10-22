#include <cassert>
#include "generic_tiling.hpp"

namespace {
GAConfig single_parent_config() {
    GAConfig config;
    config.population_size = 1;
    config.max_generations = 6;
    config.elitism_count = 0;
    return config;
}

void guaranteed_mutation_changes_each_mutable_gene() {
    auto config = single_parent_config();
    config.crossover_rate = 0.0;
    config.mutation_rate = 1.0;
    std::vector<std::vector<int>> evaluated;
    auto cost = [&](const std::vector<int>& genes) {
        evaluated.push_back(genes);
        assert(genes[0] == 1 || genes[0] == 2);
        assert(genes[1] == 128);
        assert(genes[2] == 3 || genes[2] == 4);
        return static_cast<double>(genes[0] + genes[2]);
    };
    TilingGeneticAlgorithm search({{1, 2}, {128}, {3, 4}}, cost, config);
    const auto best = search.run();
    assert(evaluated.size() == config.max_generations + 1);
    for (size_t i = 1; i < evaluated.size(); ++i) {
        // With one parent and two alternatives, mutation must toggle each gene.
        assert(evaluated[i][0] != evaluated[i - 1][0]);
        assert(evaluated[i][1] == evaluated[i - 1][1]);
        assert(evaluated[i][2] != evaluated[i - 1][2]);
    }
    assert(best.cost == best.tile_sizes[0] + best.tile_sizes[2]);
}

void unchanged_children_preserve_cached_cost() {
    for (double crossover_rate : {0.0, 1.0}) {
        auto config = single_parent_config();
        config.crossover_rate = crossover_rate;
        config.mutation_rate = 0.0;
        int calls = 0;
        auto cost = [&](const std::vector<int>& genes) {
            ++calls;
            return static_cast<double>(genes[0]);
        };
        TilingGeneticAlgorithm search({{1, 2, 3}}, cost, config);
        const auto best = search.run();
        // Crossover is either disabled or both selected parents are identical.
        assert(calls == 1);
        assert(best.cost == best.tile_sizes[0]);
    }
}

void mutation_without_alternatives_preserves_cached_cost() {
    auto config = single_parent_config();
    config.crossover_rate = 1.0;
    config.mutation_rate = 1.0;
    int calls = 0;
    auto cost = [&](const std::vector<int>& genes) {
        ++calls;
        assert((genes == std::vector<int>{128, 2}));
        return 9.0;
    };
    // Repeated copies of the current value are not mutation alternatives.
    TilingGeneticAlgorithm search({{128}, {2, 2}}, cost, config);
    const auto best = search.run();
    assert(calls == 1);
    assert(best.cost == 9.0);
}

void mutated_offspring_are_evaluated() {
    GAConfig config;
    config.population_size = 8;
    config.max_generations = 3;
    config.elitism_count = 0;
    config.crossover_rate = 1.0;
    config.mutation_rate = 1.0;
    int calls = 0;
    auto cost = [&](const std::vector<int>& genes) {
        ++calls;
        return static_cast<double>(genes[0] * genes[0]);
    };
    TilingGeneticAlgorithm search({{1, 2, 3}}, cost, config);
    const auto best = search.run();
    assert(calls == 32); // Initial population plus every mutated offspring.
    assert(best.cost == best.tile_sizes[0] * best.tile_sizes[0]);
}
}

int main() {
    guaranteed_mutation_changes_each_mutable_gene();
    unchanged_children_preserve_cached_cost();
    mutation_without_alternatives_preserves_cached_cost();
    mutated_offspring_are_evaluated();
}
