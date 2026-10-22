#include "matmul_cost_model.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <thread>

int main() {
    matmul::Config config;
    matmul::CostModel model(MATMUL_PYTHON_PATH);
    const auto expected=model.predict(config).predicted_us;
    assert(std::abs(expected-91.98508811412263)<1e-10);
    assert(model.predict(config).predicted_us==expected);
    auto invalid=config;
    invalid.dtype="fp32";
    bool rejected=false;
    try { model.predict(invalid); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    assert(model.predict(config).predicted_us==expected);
    {
        matmul::CostModel another(MATMUL_PYTHON_PATH);
        assert(another.predict(config).predicted_us==expected);
    }
    // A destroyed wrapper must not shut down another wrapper's interpreter.
    std::thread worker([&] { assert(model.predict(config).predicted_us==expected); });
    worker.join();
    assert(model.candidates(config).size()==3);
}
