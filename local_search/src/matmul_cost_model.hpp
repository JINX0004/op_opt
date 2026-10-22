#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace matmul {
struct Tiling {
    int base_m = 128;
    int base_n = 128;
    int base_k = 128;
    int step_k = 2;
};
struct Config {
    std::string implementation = "ops_nn_deterministic_split_k";
    std::array<int,3> shape = {16,16,102400};
    std::string dtype = "fp16";
    int cores = 24;
    Tiling tiling;
    std::string catlass_preset = "default";
};
struct Prediction {
    double predicted_us;
    int active_compute_cores;
    std::string details_json;
};

// Embeds CPython and reuses the unchanged, fixed calibrated Python model.
// python_package_root is the directory containing matmul_cost_model/.
class CostModel {
public:
    explicit CostModel(const std::string& python_package_root);
    ~CostModel();
    CostModel(CostModel&&) noexcept;
    CostModel& operator=(CostModel&&) noexcept;
    CostModel(const CostModel&) = delete;
    CostModel& operator=(const CostModel&) = delete;
    Prediction predict(const Config& config, bool semantic = false);
    Prediction predict_json(const std::string& config_json, bool semantic = false);
    std::vector<Config> candidates(const Config& workload);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
