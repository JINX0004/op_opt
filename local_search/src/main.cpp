#include "generic_tiling.hpp"
#include "matmul_cost_model.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef MATMUL_PYTHON_PATH
#define MATMUL_PYTHON_PATH "../cost_model"
#endif

int main(int argc, char **argv) {
  try {
    matmul::Config workload;
    std::string package_root = MATMUL_PYTHON_PATH;
    bool predict_only = false, jsonl = false, semantic = false;
    for (int i = 1; i < argc; ++i) {
      std::string option = argv[i];
      auto argument = [&]() -> std::string {
        if (++i >= argc)
          throw std::invalid_argument("missing value for " + option);
        return argv[i];
      };
      if (option == "--python-path")
        package_root = argument();
      else if (option == "--implementation")
        workload.implementation = argument();
      else if (option == "--dtype")
        workload.dtype = argument();
      else if (option == "--cores")
        workload.cores = std::stoi(argument());
      else if (option == "--shape")
        for (auto &dimension : workload.shape)
          dimension = std::stoi(argument());
      else if (option == "--tiling") {
        workload.tiling.base_m = std::stoi(argument());
        workload.tiling.base_n = std::stoi(argument());
        workload.tiling.base_k = std::stoi(argument());
        workload.tiling.step_k = std::stoi(argument());
      } else if (option == "--preset")
        workload.catlass_preset = argument();
      else if (option == "--predict")
        predict_only = true;
      else if (option == "--jsonl")
        jsonl = true;
      else if (option == "--semantic")
        semantic = true;
      else if (option == "--help") {
        std::cout
            << "Usage: local_search [--predict | --jsonl] [--semantic]\n"
               "  [--shape M N K] [--dtype fp16|bf16] [--cores 1..24]\n"
               "  [--implementation deterministic|base|scsk|catlass]\n"
               "  [--tiling base_m base_n base_k step_k] [--preset "
               "CAT_PRESET]\n"
               "  [--python-path PATH]\n"
               "Default: search deterministic FP16 [16,16,102400], 24 cores.\n"
               "--jsonl reads compact or expanded configurations from stdin.\n";
        return 0;
      } else
        throw std::invalid_argument("unknown option " + option);
    }
    matmul::CostModel model(package_root);
    if (jsonl) {
      std::string line;
      while (std::getline(std::cin, line)) {
        if (line.empty())
          continue;
        try {
          std::cout << model.predict_json(line, semantic).details_json
                    << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "Prediction error: " << e.what() << std::endl;
          return 1;
        }
      }
      return 0;
    }
    if (predict_only) {
      std::cout << model.predict(workload, semantic).details_json << '\n';
      return 0;
    }

    auto candidates = model.candidates(workload);
    std::vector<int> template_indices;
    for (std::size_t i = 0; i < candidates.size(); ++i)
      template_indices.push_back(static_cast<int>(i));
    if (candidates.empty())
      throw std::invalid_argument("no supported templates for workload/cores");
    // Validate the workload before starting the GA. Costs are runtime in us.
    model.predict(candidates.front(), semantic);
    auto cost_function = [&](const std::vector<int> &genes) -> double {
      return model.predict(candidates.at(genes.at(0)), semantic).predicted_us;
    };
    GAConfig config;
    config.population_size = 40;
    config.max_generations = 30;
    // One gene selects a complete template; crossover cannot combine decisions.
    config.crossover_rate = 0.0;
    config.mutation_rate = 0.2;
    TilingGeneticAlgorithm ga({template_indices}, cost_function, config);
    const Chromosome best = ga.run();
    const auto &selected = candidates.at(best.tile_sizes.at(0));
    std::cout << std::setprecision(12) << "\nBest reviewed tiling found: ";
    if (selected.implementation == "catlass")
      std::cout << selected.catlass_preset;
    else
      std::cout << "base_m=" << selected.tiling.base_m
                << ", base_n=" << selected.tiling.base_n
                << ", base_k=" << selected.tiling.base_k
                << ", step_k=" << selected.tiling.step_k;
    std::cout << " | Estimated runtime: " << best.cost << " us\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
