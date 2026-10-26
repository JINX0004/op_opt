#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void directional_read_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError error, const char* call) {
    if (error != ACL_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "usage: directional_read_probe A|B tile_axis M N K "
                     "copies cores warmup repetitions\n";
        return 2;
    }
    const std::string operand_name = argv[1];
    const uint32_t operand = operand_name == "A" ? 0 : 1;
    const uint32_t tile_axis = std::stoul(argv[2]);
    const uint32_t m = std::stoul(argv[3]);
    const uint32_t n = std::stoul(argv[4]);
    const uint32_t k = std::stoul(argv[5]);
    const uint32_t copies = std::stoul(argv[6]);
    const uint32_t cores = std::stoul(argv[7]);
    const uint32_t warmup = std::stoul(argv[8]);
    const uint32_t repetitions = std::stoul(argv[9]);
    if ((operand_name != "A" && operand_name != "B") ||
        (tile_axis != 128 && tile_axis != 256) ||
        m < tile_axis || n < tile_axis || k < 64 || k % 64 ||
        m % tile_axis || n % tile_axis || copies < 1 || copies > 256 ||
        cores < 1 || cores > 20 || warmup > 1000 ||
        repetitions < 1 || repetitions > 1000) return 2;
    const uint64_t input_bytes = operand == 0 ?
        uint64_t(m) * k * 2 : uint64_t(k) * n * 2;
    const uint64_t output_bytes = uint64_t(cores) * 32;
    if (input_bytes > (1ULL << 30)) return 2;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
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
            directional_read_do(cores, stream, static_cast<uint8_t*>(input),
                static_cast<uint8_t*>(output), operand, tile_axis,
                m, n, k, copies);
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
        std::vector<uint32_t> observed(output_bytes / 4);
        Check(aclrtMemcpy(observed.data(), output_bytes, output, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        for (uint32_t core = 0; core < cores; ++core) {
            const uint32_t expected = operand + tile_axis + copies + core;
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
                  << "\",\"tile_axis\":" << tile_axis
                  << ",\"m\":" << m << ",\"n\":" << n
                  << ",\"k\":" << k << ",\"copies\":" << copies
                  << ",\"cores\":" << cores
                  << ",\"device_elapsed_us\":"
                  << (millis * 1000.0 / repetitions) << "}\n";
        aclrtDestroyEvent(end);
        aclrtDestroyEvent(start);
        aclrtFree(output);
        aclrtFree(input);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
