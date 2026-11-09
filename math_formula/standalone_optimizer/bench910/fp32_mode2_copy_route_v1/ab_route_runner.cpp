#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void ab_route_probe_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError error, const char* call) {
    if (error != ACL_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr << "usage: ab_route_probe inline|plain pitch logicalK M N "
                     "repeats cores device warmup repetitions\n";
        return 2;
    }
    const std::string route_name = argv[1];
    const uint32_t route = route_name == "inline" ? 0 : 1;
    const uint32_t pitch = std::stoul(argv[2]);
    const uint32_t logical_k = std::stoul(argv[3]);
    const uint32_t m = std::stoul(argv[4]);
    const uint32_t n = std::stoul(argv[5]);
    const uint32_t repeats = std::stoul(argv[6]);
    const uint32_t cores = std::stoul(argv[7]);
    const uint32_t device = std::stoul(argv[8]);
    const uint32_t warmup = std::stoul(argv[9]);
    const uint32_t repetitions = std::stoul(argv[10]);
    if ((route_name != "inline" && route_name != "plain") ||
        logical_k == 0 || logical_k % 256 || logical_k > pitch ||
        pitch < 256 || pitch % 128 || (pitch - 256) / 8 > 65535 ||
        m != 2560 || (n != 768 && n != 1024) || repeats > 2 ||
        cores < 1 || cores > 20 || device > 7 || warmup > 100 ||
        repetitions < 1 || repetitions > 100)
        return 2;
    const uint64_t a_bytes = static_cast<uint64_t>(m) * pitch * 4;
    const uint64_t b_bytes = static_cast<uint64_t>(logical_k) * n * 4;
    const uint64_t output_bytes = static_cast<uint64_t>(cores) * 32;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        void *a = nullptr, *b = nullptr, *output = nullptr;
        Check(aclrtMalloc(&a, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(A)");
        Check(aclrtMalloc(&b, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(B)");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(output)");
        Check(aclrtMemset(a, a_bytes, 1, a_bytes), "aclrtMemset(A)");
        Check(aclrtMemset(b, b_bytes, 1, b_bytes), "aclrtMemset(B)");
        Check(aclrtMemset(output, output_bytes, 0, output_bytes), "aclrtMemset(output)");
        Check(aclrtSynchronizeDevice(), "initialize synchronize");
        auto launch = [&] {
            ab_route_probe_do(cores, stream, static_cast<uint8_t*>(a),
                static_cast<uint8_t*>(b), static_cast<uint8_t*>(output),
                route, pitch, logical_k, m, n, repeats);
        };
        for (uint32_t i = 0; i < warmup; ++i) launch();
        Check(aclrtSynchronizeStream(stream), "warmup synchronize");
        aclrtEvent start = nullptr, end = nullptr;
        Check(aclrtCreateEvent(&start), "create start");
        Check(aclrtCreateEvent(&end), "create end");
        Check(aclrtRecordEvent(start, stream), "record start");
        for (uint32_t i = 0; i < repetitions; ++i) launch();
        Check(aclrtRecordEvent(end, stream), "record end");
        Check(aclrtSynchronizeEvent(end), "synchronize end");
        float millis = 0;
        Check(aclrtEventElapsedTime(&millis, start, end), "elapsed");
        Check(aclrtSynchronizeStream(stream), "final stream synchronize");
        Check(aclrtSynchronizeDevice(), "final device synchronize");
        std::vector<uint32_t> observed(output_bytes / 4);
        Check(aclrtMemcpy(observed.data(), output_bytes, output, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        for (uint32_t core = 0; core < cores; ++core) {
            const uint32_t expected = route + pitch + logical_k + m + n + repeats + core;
            if (observed[core * 8] != expected)
                throw std::runtime_error("output token mismatch at AIC " +
                                         std::to_string(core) + ": observed " +
                                         std::to_string(observed[core * 8]) +
                                         ", expected " + std::to_string(expected));
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"route\":\"" << route_name
                  << "\",\"pitch\":" << pitch
                  << ",\"logical_k\":" << logical_k
                  << ",\"m\":" << m
                  << ",\"n\":" << n
                  << ",\"repeats\":" << repeats
                  << ",\"cores\":" << cores
                  << ",\"device\":" << device
                  << ",\"device_elapsed_us\":"
                  << (millis * 1000.0 / repetitions) << "}\n";
        aclrtDestroyEvent(end);
        aclrtDestroyEvent(start);
        aclrtFree(output);
        aclrtFree(b);
        aclrtFree(a);
        aclrtDestroyStream(stream);
        aclrtResetDevice(device);
        aclFinalize();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
