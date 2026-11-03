#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void mode6_output_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError status, const char* name) {
    if (status != ACL_SUCCESS)
        throw std::runtime_error(std::string(name) + ": " + std::to_string(status));
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: mode6_output_probe device k_packets n_tiles serial_barrier distinct_k atomic batches repetitions\n";
        return 2;
    }
    const int device = std::stoi(argv[1]);
    const uint32_t packets = std::stoul(argv[2]);
    const uint32_t n_tiles = std::stoul(argv[3]);
    const uint32_t serial = std::stoul(argv[4]);
    const uint32_t distinct = std::stoul(argv[5]);
    const uint32_t atomic = std::stoul(argv[6]);
    const int batches = std::stoi(argv[7]);
    const int repetitions = std::stoi(argv[8]);
    if (device < 0 || packets < 1 || packets > 64 ||
        (n_tiles != 8 && n_tiles != 12) || serial > 1 || distinct > 1 ||
        atomic > 1 || batches < 1 || repetitions < 1) return 2;

    constexpr uint32_t cores = 20;
    constexpr int warmup = 10;
    const uint64_t plane_elements = 2048ULL * (2 * n_tiles * 128);
    const uint64_t workspace_bytes = plane_elements * 4 * (distinct ? packets : 1);
    const uint64_t output_bytes = 32 * cores + workspace_bytes;
    const uint64_t input_bytes = 65536 * cores;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "set device");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        void *input = nullptr, *output = nullptr;
        Check(aclrtMalloc(&input, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc input");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc output");
        Check(aclrtMemset(output, output_bytes, 0, output_bytes), "zero output");
        std::vector<uint8_t> host(input_bytes);
        for (uint64_t i = 0; i < input_bytes; i += 2) {
            host[i] = 0;
            host[i + 1] = 0x3c;
        }
        Check(aclrtMemcpy(input, input_bytes, host.data(), input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        Check(aclrtSynchronizeDevice(), "init synchronize");
        auto launch = [&] {
            mode6_output_do(cores, stream, static_cast<uint8_t*>(input),
                static_cast<uint8_t*>(output), packets, n_tiles, serial,
                distinct, atomic);
        };
        for (int i = 0; i < warmup; ++i) launch();
        Check(aclrtSynchronizeStream(stream), "warmup synchronize");
        for (int b = 0; b < batches; ++b) {
            aclrtEvent start = nullptr, end = nullptr;
            Check(aclrtCreateEvent(&start), "create start");
            Check(aclrtCreateEvent(&end), "create end");
            Check(aclrtRecordEvent(start, stream), "record start");
            for (int i = 0; i < repetitions; ++i) launch();
            Check(aclrtRecordEvent(end, stream), "record end");
            Check(aclrtSynchronizeEvent(end), "synchronize end");
            float ms = 0.0f;
            Check(aclrtEventElapsedTime(&ms, start, end), "elapsed");
            std::cout << std::fixed << std::setprecision(6)
                      << "{\"device\":" << device << ",\"packets\":" << packets
                      << ",\"n_tiles\":" << n_tiles << ",\"serial\":" << serial
                      << ",\"distinct\":" << distinct << ",\"atomic\":" << atomic
                      << ",\"batch\":" << b << ",\"repetitions\":" << repetitions
                      << ",\"us\":" << ms * 1000.0 / repetitions << "}\n";
            aclrtDestroyEvent(end);
            aclrtDestroyEvent(start);
        }
        uint32_t token = 0;
        Check(aclrtMemcpy(&token, 4, output, 4, ACL_MEMCPY_DEVICE_TO_HOST), "token D2H");
        float value = 0;
        Check(aclrtMemcpy(&value, 4, static_cast<uint8_t*>(output) + 32 * cores,
                          4, ACL_MEMCPY_DEVICE_TO_HOST), "value D2H");
        const float expected = atomic ?
            128.0f * (distinct ? 1 : packets) * (warmup + batches * repetitions) : 128.0f;
        std::cerr << "observed token=" << token << " value=" << value
                  << " expected value=" << expected << '\n';
        if (token != packets + n_tiles)
            throw std::runtime_error("token mismatch: " + std::to_string(token) +
                                     " expected " + std::to_string(packets + n_tiles));
        if (std::abs(value - expected) > 0.01f * expected)
            throw std::runtime_error("FP32 output mismatch: " + std::to_string(value) +
                                     " expected " + std::to_string(expected));
        aclrtFree(output);
        aclrtFree(input);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device);
        aclFinalize();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
