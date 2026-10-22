#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>

struct GAConfig {
    size_t population_size = 50;
    size_t max_generations = 100;
    double crossover_rate = 0.8;
    double mutation_rate = 0.15; // Probability per gene, not per chromosome
    size_t tournament_size = 3;
    size_t elitism_count = 2; // Keep top N best solutions directly
};

// Represents a tiling configuration across N dimensions
struct Chromosome {
    std::vector<int> tile_sizes;
    double cost = std::numeric_limits<double>::infinity(); // Lower cost is better
};

class TilingGeneticAlgorithm {
public:
    using CostFunction = std::function<double(const std::vector<int>&)>;

    TilingGeneticAlgorithm(
        std::vector<std::vector<int>> valid_choices_per_dim,
        CostFunction cost_fn,
        GAConfig config = GAConfig{}
    ) : valid_choices_(std::move(valid_choices_per_dim)),
        cost_fn_(std::move(cost_fn)),
        config_(config),
        rng_(std::random_device{}()) {}

    Chromosome run() {
        std::vector<Chromosome> population = initialize_population();
        evaluate_population(population);

        Chromosome best_ever = population[0];

        for (size_t gen = 0; gen < config_.max_generations; ++gen) {
            // Sort population by cost (ascending: lower cost = better performance)
            std::sort(population.begin(), population.end(), [](const Chromosome& a, const Chromosome& b) {
                return a.cost < b.cost;
            });

            if (population[0].cost < best_ever.cost) {
                best_ever = population[0];
            }

            std::cout << "Generation " << gen << " | Best Cost: " << best_ever.cost << " | Tile Sizes: [";
            for (size_t i = 0; i < best_ever.tile_sizes.size(); ++i) {
                std::cout << best_ever.tile_sizes[i] << (i + 1 < best_ever.tile_sizes.size() ? ", " : "");
            }
            std::cout << "]\n";

            // Create next generation
            std::vector<Chromosome> next_gen;
            next_gen.reserve(config_.population_size);

            // Elitism: carry over top configurations unchanged
            for (size_t i = 0; i < config_.elitism_count && i < population.size(); ++i) {
                next_gen.push_back(population[i]);
            }

            // Reproduce until population is full
            while (next_gen.size() < config_.population_size) {
                const auto& parent1 = select_parent(population);
                const auto& parent2 = select_parent(population);

                Chromosome child = crossover(parent1, parent2);
                mutate(child);
                next_gen.push_back(child);
            }

            population = std::move(next_gen);
            evaluate_population(population);
        }

        // Include offspring evaluated in the final generation.
        for (const auto& individual : population) {
            if (individual.cost < best_ever.cost) {
                best_ever = individual;
            }
        }
        return best_ever;
    }

private:
    std::vector<std::vector<int>> valid_choices_; // Allowed tile sizes for each dimension
    CostFunction cost_fn_;
    GAConfig config_;
    std::mt19937 rng_;

    std::vector<Chromosome> initialize_population() {
        std::vector<Chromosome> pop(config_.population_size);
        for (auto& indiv : pop) {
            indiv.tile_sizes.resize(valid_choices_.size());
            for (size_t d = 0; d < valid_choices_.size(); ++d) {
                std::uniform_int_distribution<size_t> dist(0, valid_choices_[d].size() - 1);
                indiv.tile_sizes[d] = valid_choices_[d][dist(rng_)];
            }
        }
        return pop;
    }

    void evaluate_population(std::vector<Chromosome>& population) {
        for (auto& indiv : population) {
            // Evaluates fitness only if cost hasn't been cached
            if (indiv.cost == std::numeric_limits<double>::infinity()) {
                indiv.cost = cost_fn_(indiv.tile_sizes);
            }
        }
    }

    const Chromosome& select_parent(const std::vector<Chromosome>& population) {
        std::uniform_int_distribution<size_t> dist(0, population.size() - 1);
        size_t best_idx = dist(rng_);

        for (size_t i = 1; i < config_.tournament_size; ++i) {
            size_t idx = dist(rng_);
            if (population[idx].cost < population[best_idx].cost) {
                best_idx = idx;
            }
        }
        return population[best_idx];
    }

    Chromosome crossover(const Chromosome& p1, const Chromosome& p2) {
        std::bernoulli_distribution should_cross(config_.crossover_rate);
        std::bernoulli_distribution inherit_second(0.5);
        Chromosome child = p1;

        if (should_cross(rng_)) {
            // Uniform crossover across vector dimensions
            for (size_t i = 0; i < p1.tile_sizes.size(); ++i) {
                if (inherit_second(rng_)) {
                    child.tile_sizes[i] = p2.tile_sizes[i];
                }
            }
        }
        // A clone can reuse its parent's evaluated fitness.
        if (child.tile_sizes != p1.tile_sizes) {
            child.cost = std::numeric_limits<double>::infinity();
        }
        return child;
    }

    void mutate(Chromosome& indiv) {
        std::bernoulli_distribution should_mutate(config_.mutation_rate);

        for (size_t d = 0; d < indiv.tile_sizes.size(); ++d) {
            if (!should_mutate(rng_)) {
                continue;
            }

            std::vector<int> alternatives;
            for (int value : valid_choices_[d]) {
                if (value != indiv.tile_sizes[d]) {
                    alternatives.push_back(value);
                }
            }
            if (alternatives.empty()) {
                continue;
            }

            std::uniform_int_distribution<size_t> choice_dist(0, alternatives.size() - 1);
            indiv.tile_sizes[d] = alternatives[choice_dist(rng_)];
            indiv.cost = std::numeric_limits<double>::infinity();
        }
    }
};
