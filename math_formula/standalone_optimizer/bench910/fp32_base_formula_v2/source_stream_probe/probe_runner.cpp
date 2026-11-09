#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void base_stream_probe_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t);

static void Check(aclError error, const char* call) {
    if (error != ACL_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}

int main(int argc, char** argv) {
    if (argc != 12) {
        std::cerr << "usage: base_stream_probe A|B|AB|AB_REGION baseM baseN k_span M N K "
                     "repeats cores warmup repetitions\n";
        return 2;
    }
    const std::string operand_name = argv[1];
    const uint32_t operand = operand_name == "A" ? 0 : operand_name == "B" ? 1 :
                             operand_name == "AB" ? 2 : 3;
    const uint32_t base_m = std::stoul(argv[2]);
    const uint32_t base_n = std::stoul(argv[3]);
    const uint32_t k_span = std::stoul(argv[4]);
    const uint32_t m = std::stoul(argv[5]);
    const uint32_t n = std::stoul(argv[6]);
    const uint32_t k = std::stoul(argv[7]);
    const uint32_t repeats = std::stoul(argv[8]);
    const uint32_t cores = std::stoul(argv[9]);
    const uint32_t warmup = std::stoul(argv[10]);
    const uint32_t repetitions = std::stoul(argv[11]);
    if ((operand_name != "A" && operand_name != "B" && operand_name != "AB" &&
         operand_name != "AB_REGION") ||
        base_m < 16 || base_m > 256 || base_m % 16 ||
        base_n < 16 || base_n > 256 || base_n % 16 ||
        k_span < 128 || k_span > 4096 || k_span % 128 ||
        k < k_span || k % k_span ||
        (operand == 0 && uint64_t(base_m) * k_span * 4 > 131072) ||
        (operand == 1 && uint64_t(base_n) * k_span * 4 > 131072) ||
        (operand >= 2 && (uint64_t(base_m) * k_span * 8 > 131072 ||
                          uint64_t(base_n) * k_span * 4 > 131072)) ||
        (operand == 3 && ((m / base_m) % 4 || (n / base_n) > 5)) ||
        m < base_m || n < base_n ||
        m % base_m || n % base_n || repeats > 4 ||
        cores < 1 || cores > 20 || warmup > 1000 ||
        repetitions < 1 || repetitions > 1000) return 2;
    const uint64_t a_bytes = uint64_t(m) * k * 4;
    const uint64_t b_bytes = uint64_t(k) * n * 4;
    const uint64_t input_bytes = operand == 0 ? a_bytes :
                                 operand == 1 ? b_bytes : a_bytes + b_bytes;
    const uint64_t output_bytes = uint64_t(cores) * 32;
    if (input_bytes > (1ULL << 30)) return 2;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(4), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        void *input = nullptr, *output = nullptr;
        Check(aclrtMalloc(&input, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(input)");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(output)");
        Check(aclrtMemset(output, output_bytes, 0, output_bytes),
              "aclrtMemset(output)");
        std::vector<uint8_t> host(input_bytes, 1);
        Check(aclrtMemcpy(input, input_bytes, host.data(), input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
        Check(aclrtSynchronizeDevice(), "initialize synchronize");
        auto launch = [&] {
            base_stream_probe_do(cores, stream, static_cast<uint8_t*>(input),
                static_cast<uint8_t*>(output), operand, base_m, base_n,
                k_span, m, n, k, repeats);
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
        Check(aclrtSynchronizeDevice(), "final synchronize");
        std::vector<uint32_t> observed(output_bytes / 4);
        Check(aclrtMemcpy(observed.data(), output_bytes, output, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        for (uint32_t core = 0; core < cores; ++core) {
            const uint32_t expected = operand + base_m + base_n + k_span + repeats + core;
            if (observed[core * 8] != expected) {
                std::cerr << "observed token slots:";
                for (uint32_t j = 0; j < cores; ++j)
                    std::cerr << ' ' << j << ':' << observed[j * 8];
                std::cerr << '\n';
                throw std::runtime_error("output token mismatch at AIC " +
                                         std::to_string(core) + ": " +
                                         std::to_string(observed[core * 8]) +
                                         " != " + std::to_string(expected));
            }
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"operand\":\"" << operand_name
                  << "\",\"base_m\":" << base_m
                  << ",\"base_n\":" << base_n
                  << ",\"k_span\":" << k_span
                  << ",\"m\":" << m << ",\"n\":" << n
                  << ",\"k\":" << k << ",\"repeats\":" << repeats
                  << ",\"cores\":" << cores
                  << ",\"device_elapsed_us\":"
                  << (millis * 1000.0 / repetitions) << "}\n";
        aclrtDestroyEvent(end);
        aclrtDestroyEvent(start);
        aclrtFree(output);
        aclrtFree(input);
        aclrtDestroyStream(stream);
        aclrtResetDevice(4);
        aclFinalize();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
