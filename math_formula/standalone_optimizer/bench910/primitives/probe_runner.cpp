#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void solvebase_probe_do(uint32_t, void*, uint8_t*, uint8_t*,
                                    uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError error, const char* call) {
    if (error != ACL_SUCCESS) throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: solvebase_probe mode cores packets packet_bytes warmup batches repetitions\n";
        return 2;
    }
    const uint32_t mode = std::stoul(argv[1]);
    const uint32_t cores = std::stoul(argv[2]);
    const uint32_t packets = std::stoul(argv[3]);
    const uint32_t packet_bytes = std::stoul(argv[4]);
    const int warmup = std::stoi(argv[5]);
    const int batches = std::stoi(argv[6]);
    const int repetitions = std::stoi(argv[7]);
    if (mode > 4 || cores < 1 || cores > 20 || packets > 100000 ||
        packet_bytes < 32 || packet_bytes > (mode == 4 ? 262144U : 131072U) ||
        packet_bytes % 32 ||
        warmup < 0 || batches < 1 || repetitions < 1) return 2;
    if (mode == 2 && (packets == 0 || packets > 256 ||
                      (packets & (packets - 1)) || packet_bytes != 32768)) return 2;
    if (mode == 3 && (packets == 0 || packets > 256 ||
                      packet_bytes < 4096 || packet_bytes > 32768 ||
                      packet_bytes % 4096)) return 2;
    if (mode == 4 && (packets == 0 || packets > 64 ||
                      packet_bytes < 4096 || packet_bytes > 262144 ||
                      packet_bytes % 4096)) return 2;
    const uint64_t stride = (mode == 2 || mode == 3) ? 65536 :
        std::max<uint64_t>(32, uint64_t(packets) * packet_bytes);
    const uint64_t bytes = std::max<uint64_t>(32, stride * cores);
    const uint64_t output_bytes = 32ULL * cores +
        (mode == 2 ? 32768ULL * cores :
         mode == 3 ? uint64_t(cores) * packets * packet_bytes : 0);
    if (bytes > (1ULL << 30)) return 2;
    try {
        const bool verbose = std::getenv("SOLVEBASE_PROBE_VERBOSE") != nullptr;
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        void *input = nullptr, *output = nullptr;
        Check(aclrtMalloc(&input, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(output)");
        if (verbose) std::cerr << "allocated " << bytes << " input bytes\n";
        Check(aclrtMemset(output, output_bytes, 0, output_bytes), "aclrtMemset(output)");
        std::vector<uint8_t> host(bytes, 1);
        if (mode == 2 || mode == 3) {
            for (uint64_t i = 0; i < bytes; i += 2) {
                host[i] = 0; host[i + 1] = 0x3c; // FP16 value 1.0
            }
        }
        Check(aclrtMemcpy(input, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
        if (verbose) std::cerr << "initialized input\n";
        // aclrtMemset may run on the default stream. Complete initialization
        // before launching work on our separate timing stream.
        Check(aclrtSynchronizeDevice(), "initialize synchronize");
        if (verbose) std::cerr << "initialization synchronized\n";
        auto launch = [&] {
            solvebase_probe_do(cores, stream, static_cast<uint8_t*>(input),
                static_cast<uint8_t*>(output), mode, packets, packet_bytes,
                static_cast<uint32_t>(stride));
        };
        for (int i = 0; i < warmup; ++i) launch();
        Check(aclrtSynchronizeStream(stream), "warmup synchronize");
        if (verbose) std::cerr << "warmup synchronized\n";
        for (int b = 0; b < batches; ++b) {
            aclrtEvent start = nullptr, end = nullptr;
            Check(aclrtCreateEvent(&start), "create start");
            Check(aclrtCreateEvent(&end), "create end");
            Check(aclrtRecordEvent(start, stream), "record start");
            for (int i = 0; i < repetitions; ++i) launch();
            Check(aclrtRecordEvent(end, stream), "record end");
            Check(aclrtSynchronizeEvent(end), "synchronize end");
            float millis = 0;
            Check(aclrtEventElapsedTime(&millis, start, end), "elapsed");
            std::cout << std::fixed << std::setprecision(6)
                      << "{\"mode\":" << mode << ",\"cores\":" << cores
                      << ",\"packets\":" << packets
                      << ",\"packet_bytes\":" << packet_bytes
                      << ",\"batch\":" << b
                      << ",\"repetitions\":" << repetitions
                      << ",\"device_elapsed_us\":" << (millis * 1000.0 / repetitions)
                      << "}\n";
            aclrtDestroyEvent(end);
            aclrtDestroyEvent(start);
        }
        std::vector<uint32_t> observed(output_bytes / 4);
        Check(aclrtMemcpy(observed.data(), observed.size() * 4, output,
                          observed.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        for (uint32_t i = 0; i < cores; ++i)
            if (observed[i * 8] != packets + packet_bytes + i) {
                std::cerr << "nonzero token slots:";
                for (uint32_t j = 0; j < cores * 8; ++j)
                    if (observed[j]) std::cerr << ' ' << j << ':' << observed[j];
                std::cerr << '\n';
                throw std::runtime_error("output token mismatch at core " +
                    std::to_string(i) + ": " + std::to_string(observed[i * 8]) +
                    " != " + std::to_string(packets + packet_bytes + i));
            }
        if (mode == 2 || mode == 3) {
            uint16_t value = 0;
            auto* matrix = static_cast<uint8_t*>(output) + 32 * cores;
            Check(aclrtMemcpy(&value, 2, matrix, 2, ACL_MEMCPY_DEVICE_TO_HOST), "matrix D2H");
            // The first output element is 128 * Cube repeat count. 128 and
            // these bounded multiples are exactly representable in FP16.
            uint16_t expected = 0x5800;
            if (mode == 2)
                for (uint32_t p = packets; p > 1; p >>= 1) expected += 0x400;
            if (value != expected)
                throw std::runtime_error("Cube/Fixpipe output mismatch: " +
                    std::to_string(value) + " != " + std::to_string(expected));
        }
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
