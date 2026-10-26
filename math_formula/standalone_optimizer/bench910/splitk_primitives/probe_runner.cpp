#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void splitk_probe_do(void*, uint8_t*, uint8_t*, uint8_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError status, const char* call) {
    if (status != ACL_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + std::to_string(status));
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "usage: splitk_probe mode m k elements_per_aiv partials baseN baseD warmup repetitions\n";
        return 2;
    }
    const auto mode = static_cast<uint32_t>(std::stoul(argv[1]));
    const auto m = static_cast<uint32_t>(std::stoul(argv[2]));
    const auto k = static_cast<uint32_t>(std::stoul(argv[3]));
    const auto elements = static_cast<uint32_t>(std::stoul(argv[4]));
    const auto partials = static_cast<uint32_t>(std::stoul(argv[5]));
    const auto base_n = static_cast<uint32_t>(std::stoul(argv[6]));
    const auto base_d = static_cast<uint32_t>(std::stoul(argv[7]));
    const auto warmup = std::stoi(argv[8]);
    const auto repetitions = std::stoi(argv[9]);
    if (mode > 3 || !m || !k || !elements || elements > 8192 ||
        elements % 256 || !partials ||
        (mode == 2 && partials > 20) ||
        (mode == 3 && partials > 128) ||
        !base_n || !base_d || warmup < 0 || repetitions < 1) return 2;
    const uint64_t input_bytes = mode == 1 ? 2ULL * m * k :
        mode == 2 ? 4ULL * 40 * elements * partials :
        mode == 3 ? 4ULL * 40 * elements * 2 : 32;
    const uint64_t output_bytes = mode == 1 ? 2ULL * m * k :
        mode == 2 ? 2ULL * 40 * elements :
        mode == 3 ? 4ULL * 40 * elements : 32;
    if (input_bytes > 512ULL * 1024 * 1024 ||
        output_bytes > 512ULL * 1024 * 1024) return 2;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        void *input = nullptr, *output = nullptr, *tokens = nullptr;
        Check(aclrtMalloc(&input, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc input");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc output");
        Check(aclrtMalloc(&tokens, 40 * 32, ACL_MEM_MALLOC_HUGE_FIRST), "malloc tokens");
        std::vector<uint8_t> host(input_bytes, 0);
        if (mode == 1) {
            for (uint64_t i = 0; i < input_bytes; i += 2) {
                host[i] = 0; host[i + 1] = 0x3c; // FP16 1
            }
        } else if (mode == 2 || mode == 3) {
            const float one = 1.0f;
            for (uint64_t i = 0; i < input_bytes; i += 4)
                std::memcpy(host.data() + i, &one, 4);
        }
        Check(aclrtMemcpy(input, input_bytes, host.data(), input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
        Check(aclrtMemset(output, output_bytes, 0, output_bytes), "memset output");
        Check(aclrtMemset(tokens, 40 * 32, 0, 40 * 32), "memset tokens");
        Check(aclrtSynchronizeDevice(), "initialize");
        auto launch = [&] {
            splitk_probe_do(stream, static_cast<uint8_t*>(input),
                static_cast<uint8_t*>(output), static_cast<uint8_t*>(tokens),
                mode, m, k, elements, partials, base_n, base_d);
        };
        for (int i = 0; i < warmup; ++i) launch();
        Check(aclrtSynchronizeStream(stream), "warmup");
        aclrtEvent start = nullptr, end = nullptr;
        Check(aclrtCreateEvent(&start), "event start");
        Check(aclrtCreateEvent(&end), "event end");
        Check(aclrtRecordEvent(start, stream), "record start");
        for (int i = 0; i < repetitions; ++i) launch();
        Check(aclrtRecordEvent(end, stream), "record end");
        Check(aclrtSynchronizeEvent(end), "synchronize end");
        float millis = 0;
        Check(aclrtEventElapsedTime(&millis, start, end), "elapsed");
        std::vector<uint32_t> observed(40 * 8);
        Check(aclrtMemcpy(observed.data(), 40 * 32, tokens, 40 * 32,
                          ACL_MEMCPY_DEVICE_TO_HOST), "D2H tokens");
        for (uint32_t core = 0; core < 40; ++core)
            if (observed[core * 8] != mode + core + 1)
                throw std::runtime_error("AIV token mismatch on core " +
                                         std::to_string(core));
        if (mode == 1 || mode == 2) {
            uint16_t first = 0, last = 0;
            Check(aclrtMemcpy(&first, 2, output, 2,
                              ACL_MEMCPY_DEVICE_TO_HOST), "first output");
            Check(aclrtMemcpy(&last, 2,
                              static_cast<uint8_t*>(output) + output_bytes - 2,
                              2, ACL_MEMCPY_DEVICE_TO_HOST), "last output");
            uint16_t expected = 0x3c00;
            if (mode == 2) {
                switch (partials) {
                    case 1: expected = 0x3c00; break;
                    case 2: expected = 0x4000; break;
                    case 4: expected = 0x4400; break;
                    case 8: expected = 0x4800; break;
                    case 16: expected = 0x4c00; break;
                    case 20: expected = 0x4d00; break;
                    default: throw std::runtime_error("unsupported checked partial count");
                }
            }
            if (first != expected || last != expected)
                throw std::runtime_error("output mismatch: " +
                    std::to_string(first) + "," + std::to_string(last) +
                    " expected " + std::to_string(expected));
        } else if (mode == 3) {
            float first = 0, last = 0;
            Check(aclrtMemcpy(&first, 4, output, 4,
                              ACL_MEMCPY_DEVICE_TO_HOST), "first vector output");
            Check(aclrtMemcpy(&last, 4,
                              static_cast<uint8_t*>(output) + output_bytes - 4,
                              4, ACL_MEMCPY_DEVICE_TO_HOST), "last vector output");
            if (first != partials + 1.0f || last != partials + 1.0f)
                throw std::runtime_error("resident vector Add output mismatch");
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"mode\":" << mode << ",\"m\":" << m << ",\"k\":" << k
                  << ",\"elements_per_aiv\":" << elements
                  << ",\"partials\":" << partials
                  << ",\"base_n\":" << base_n << ",\"base_d\":" << base_d
                  << ",\"device_elapsed_us\":" << millis * 1000.0 / repetitions
                  << "}\n";
        aclrtDestroyEvent(end);
        aclrtDestroyEvent(start);
        aclrtFree(tokens);
        aclrtFree(output);
        aclrtFree(input);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
