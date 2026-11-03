#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void mmv3_converted_a_probe_do(void*, uint8_t*, uint8_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError code, const char* name) {
    if (code != ACL_SUCCESS)
        throw std::runtime_error(std::string(name) + ": " + std::to_string(code));
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "usage: probe device M K baseAN baseAD mode warmup batches repetitions\n";
        return 2;
    }
    const int device = std::stoi(argv[1]);
    const uint32_t m = std::stoul(argv[2]);
    const uint32_t k = std::stoul(argv[3]);
    const uint32_t base_an = std::stoul(argv[4]);
    const uint32_t base_ad = std::stoul(argv[5]);
    const uint32_t mode = std::stoul(argv[6]);
    const int warmup = std::stoi(argv[7]);
    const int batches = std::stoi(argv[8]);
    const int repetitions = std::stoi(argv[9]);
    if (device < 0 || device > 7 || m != 2560 ||
        (k != 65536 && k != 73728) || !base_an || !base_ad ||
        mode > 1 || warmup < 0 || batches < 1 || repetitions < 1)
        return 2;
    const std::size_t bytes = 2ULL * m * k;
    const std::size_t output_bytes = bytes + 40 * 32;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        void* input = nullptr;
        void* output = nullptr;
        Check(aclrtMalloc(&input, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc input");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc output");
        Check(aclrtMemset(input, bytes, 1, bytes), "memset input");
        Check(aclrtMemset(output, output_bytes, 0, output_bytes), "memset output");
        Check(aclrtSynchronizeDevice(), "initialization sync");
        const auto launch = [&] {
            mmv3_converted_a_probe_do(stream, static_cast<uint8_t*>(input),
                static_cast<uint8_t*>(output), m, k, base_an, base_ad, mode);
        };
        for (int i = 0; i < warmup; ++i) launch();
        Check(aclrtSynchronizeStream(stream), "warmup sync");
        for (int batch = 0; batch < batches; ++batch) {
            aclrtEvent start = nullptr, end = nullptr;
            Check(aclrtCreateEvent(&start), "create start");
            Check(aclrtCreateEvent(&end), "create end");
            Check(aclrtRecordEvent(start, stream), "record start");
            for (int i = 0; i < repetitions; ++i) launch();
            Check(aclrtRecordEvent(end, stream), "record end");
            Check(aclrtSynchronizeEvent(end), "event sync");
            float elapsed_ms = 0;
            Check(aclrtEventElapsedTime(&elapsed_ms, start, end), "elapsed");
            std::cout << std::fixed << std::setprecision(6)
                      << "{\"device\":" << device << ",\"m\":" << m
                      << ",\"k\":" << k << ",\"baseAN\":" << base_an
                      << ",\"baseAD\":" << base_ad << ",\"mode\":" << mode
                      << ",\"batch\":" << batch
                      << ",\"repetitions\":" << repetitions
                      << ",\"event_us\":"
                      << 1000.0 * elapsed_ms / repetitions << "}\n";
            aclrtDestroyEvent(end);
            aclrtDestroyEvent(start);
        }
        // The generated vector-only stub can submit work on a runtime
        // stream other than the caller's event stream. Synchronize the
        // device before validating the output and handoff tokens.
        Check(aclrtSynchronizeDevice(), "post-timing device sync");
        if (mode == 1) {
            // The chosen M and K are both 16-aligned. A constant byte
            // pattern must occupy every output byte after the permutation.
            std::vector<uint8_t> observed(bytes);
            Check(aclrtMemcpy(observed.data(), bytes, output, bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST), "read converted A");
            const auto wrong = std::find_if(observed.begin(), observed.end(),
                [](uint8_t value) { return value != 1; });
            if (wrong != observed.end())
                throw std::runtime_error("converted A mismatch at byte " +
                    std::to_string(wrong - observed.begin()));
        }
        if (mode == 1) {
            std::vector<uint32_t> tokens(40 * 8);
            Check(aclrtMemcpy(tokens.data(), tokens.size() * 4,
                              static_cast<uint8_t*>(output) + bytes,
                              tokens.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST),
                  "read conversion tokens");
            for (std::size_t core = 0; core < 40; ++core)
                if (tokens[core * 8] != m + k + base_an + base_ad + mode + core)
                    throw std::runtime_error("conversion token mismatch at " +
                        std::to_string(core) + ": got " +
                        std::to_string(tokens[core * 8]));
        }
        Check(aclrtFree(output), "free output");
        Check(aclrtFree(input), "free input");
        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtResetDevice(device), "reset device");
        Check(aclFinalize(), "aclFinalize");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
