# Search with the calibrated Python cost model

The C++ search embeds CPython through its C API. It imports the existing
`cost_model/matmul_cost_model` package and constructs `MatmulCostModel` once.
Each fitness evaluation calls its unchanged `predict` method; there is no
subprocess per candidate, model port, or replacement approximation. Repeated
configurations are cached in the wrapper. The Python package's input validation
and frozen calibration remain authoritative.

## Build and run

Requires a C++17 compiler and Python 3.10+ with development headers and an
embeddable shared library (`python3-config --embed --ldflags`). No pip packages
or additional C++ libraries are needed.

```sh
make -C local_search
# Search the three deterministic templates for FP16 [16,16,102400], 24 cores.
./local_search/build/bin/local_search
# Estimate the original configuration directly; output is the Python prediction.
./local_search/build/bin/local_search --predict
# Choose a different workload and implementation.
./local_search/build/bin/local_search --implementation base --shape 64 32 4096 --cores 8
# Run regression tests, including all 1198 frozen validation predictions.
make -C local_search test
```

Select another Python installation at build time, if needed:

```sh
make -C local_search PYTHON_CONFIG=/path/to/python3-config
```

The binary uses an absolute package path established at build time. To relocate
the Python package, pass `--python-path /path/to/cost_model`. This directory
must contain `matmul_cost_model/`, including its fixed model data and engine
sources. Python and its standard library must also remain installed. The build
adds the chosen Python library directories to the executable's runtime search
path.

## C++ API and fitness function

```cpp
#include "matmul_cost_model.hpp"

matmul::CostModel model("/path/to/op_opt/cost_model");
matmul::Config config;
config.implementation = "ops_nn_deterministic_split_k";
config.shape = {16, 16, 102400};
config.dtype = "fp16";
config.cores = 24;
config.tiling = {128, 128, 128, 2};

double runtime_us = model.predict(config).predicted_us;
// 91.98508811412263 us
```

`main.cpp` obtains legal candidates with `model.candidates(workload)`, which calls
the Python API's `generate_candidates`. A chromosome selects a reviewed tiling
by index. This prevents crossover from creating unsupported tile combinations.
The cost function is:

```cpp
auto cost_function = [&](const std::vector<int>& genes) -> double {
    return model.predict(candidates.at(genes.at(0))).predicted_us;
};
```

`predict_json` also accepts compact configurations or complete exported
`static_input` JSON. `--jsonl` reads one configuration per stdin line and emits
one complete Python prediction per line. Invalid inputs raise C++ exceptions;
the CLI reports the error and exits unsuccessfully. `--semantic` selects the
uncorrected semantic pipeline instead of the default calibrated estimate.

The wrapper holds the GIL around Python calls and serializes access to each
model. CPython lives for the process lifetime; destroying a wrapper releases its
Python references without finalizing an interpreter another wrapper may use.
Concurrent calls are safe but Python inference is serialized by the GIL.

The current single-template-index search disables crossover and uses a 0.2
mutation probability per gene. Mutation selects a different legal value when
one is available. The generic GA retains uniform crossover for searches with
multiple genes; changed offspring invalidate their cached costs, while unchanged
offspring retain them. The final evaluated generation is included in the result.
Its printed result is the best candidate found, not a certificate of globally
optimal measured performance.
