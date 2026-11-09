#include <acl/acl.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void mode2_stage_ladder_do(uint32_t, void*, uint8_t*, uint8_t*,
    uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t);

static void Check(aclError error, const char* call) {
    if (error != ACL_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}

int main(int argc, char** argv) {
    if (argc != 13) {
        std::cerr << "usage: mode2_stage_ladder pitch logicalK M N coreM coreN "
                     "innerM packetK mode cores device repetitions\n";
        return 2;
    }
    const uint32_t pitch = std::stoul(argv[1]);
    const uint32_t logical_k = std::stoul(argv[2]);
    const uint32_t m = std::stoul(argv[3]);
    const uint32_t n = std::stoul(argv[4]);
    const uint32_t core_m = std::stoul(argv[5]);
    const uint32_t core_n = std::stoul(argv[6]);
    const uint32_t inner_m = std::stoul(argv[7]);
    const uint32_t packet_k = std::stoul(argv[8]);
    const uint32_t mode = std::stoul(argv[9]);
    const uint32_t cores = std::stoul(argv[10]);
    const uint32_t device = std::stoul(argv[11]);
    const uint32_t repetitions = std::stoul(argv[12]);
    if (pitch < logical_k || pitch % 256 || !logical_k ||
        !m || !n || !core_m || !core_n || !inner_m || !packet_k ||
        inner_m > 384 || packet_k > 256 || packet_k % 64 ||
        logical_k % packet_k || n % 128 || core_n % 128 ||
        mode > 3 || cores < 1 || cores > 20 || core_m > 256 ||
        device > 7 || repetitions < 1 || repetitions > 20 ||
        ((m + core_m - 1) / core_m) * ((n + core_n - 1) / core_n) < cores)
        return 2;
    const uint64_t a_bytes = static_cast<uint64_t>(m) * pitch * 4;
    const uint64_t b_bytes = static_cast<uint64_t>(logical_k) * n * 4;
    const uint64_t scratch_bytes = static_cast<uint64_t>(cores) *
        (core_n / 128) * 2 * 128 * 128 * 4;
    const uint64_t output_bytes = static_cast<uint64_t>(cores) * 32;
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        void *a = nullptr, *b = nullptr, *scratch = nullptr, *output = nullptr;
        Check(aclrtMalloc(&a, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(A)");
        Check(aclrtMalloc(&b, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(B)");
        Check(aclrtMalloc(&scratch, scratch_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(scratch)");
        Check(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(output)");
        Check(aclrtMemset(a, a_bytes, 0x3f, a_bytes), "aclrtMemset(A)");
        Check(aclrtMemset(b, b_bytes, 0x3f, b_bytes), "aclrtMemset(B)");
        Check(aclrtMemset(scratch, scratch_bytes, 0, scratch_bytes),
              "aclrtMemset(scratch)");
        Check(aclrtMemset(output, output_bytes, 0, output_bytes), "aclrtMemset(output)");
        Check(aclrtSynchronizeDevice(), "initialize synchronize");
        auto launch = [&] {
            mode2_stage_ladder_do(cores, stream, static_cast<uint8_t*>(a),
                static_cast<uint8_t*>(b), static_cast<uint8_t*>(scratch),
                static_cast<uint8_t*>(output),
                pitch, logical_k, m, n, core_m, core_n, inner_m,
                packet_k, mode);
        };
        for (uint32_t i = 0; i < 3; ++i) launch();
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
            const uint32_t expected = pitch + logical_k + m + n + core_m +
                core_n + inner_m + packet_k + mode + core;
            if (observed[core * 8] != expected) {
                std::string detail;
                for (uint32_t slot = 0; slot < cores; ++slot)
                    detail += " " + std::to_string(slot) + ":" +
                              std::to_string(observed[slot * 8]);
                throw std::runtime_error("output token mismatch at AIC " +
                                         std::to_string(core) + ": observed " +
                                         std::to_string(observed[core * 8]) +
                                         ", expected " + std::to_string(expected) +
                                         "; slots" + detail);
            }
        }
        float scratch_sample = 0;
        if (mode == 3) {
            Check(aclrtMemcpy(&scratch_sample, sizeof(float), scratch,
                              sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                  "scratch sample D2H");
            if (!std::isfinite(scratch_sample) || scratch_sample <= 0)
                throw std::runtime_error("Fixpipe scratch sample invalid");
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"pitch\":" << pitch << ",\"logical_k\":" << logical_k
                  << ",\"m\":" << m << ",\"n\":" << n
                  << ",\"core_m\":" << core_m << ",\"core_n\":" << core_n
                  << ",\"inner_m\":" << inner_m << ",\"packet_k\":" << packet_k
                  << ",\"mode\":" << mode
                  << ",\"cores\":" << cores << ",\"device\":" << device
                  << ",\"scratch_sample\":" << scratch_sample
                  << ",\"device_elapsed_us\":" << (millis * 1000.0 / repetitions)
                  << "}\n";
        aclrtDestroyEvent(end);
        aclrtDestroyEvent(start);
        aclrtFree(output);
        aclrtFree(scratch);
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
