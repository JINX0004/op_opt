// Device-event timing harness for installed CANN matmul paths.
// The direct MatMulV3 path is the intended SolveBase validation route.
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mm.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

struct Options {
    int device = 0;
    std::int64_t m = 256, n = 256, k = 1024;
    int warmup = 20, batches = 5, repetitions = 20;
    std::string api = "v3";
};

std::int64_t ParseInteger(const char* value, const char* name) {
    errno = 0;
    char* end = nullptr;
    const auto result = std::strtoll(value, &end, 10);
    if (errno || end == value || *end != '\0')
        throw std::invalid_argument(std::string("invalid ") + name);
    return result;
}

Options Parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("option requires a value");
        const std::string name(argv[i]);
        if (name == "--api") {
            o.api = argv[i + 1];
            continue;
        }
        const auto x = ParseInteger(argv[i + 1], argv[i]);
        if ((name == "--device" || name == "--warmup" ||
             name == "--batches" || name == "--repetitions") &&
            (x < std::numeric_limits<int>::min() ||
             x > std::numeric_limits<int>::max()))
            throw std::invalid_argument("option out of int range: " + name);
        if (name == "--device") o.device = static_cast<int>(x);
        else if (name == "--m") o.m = x;
        else if (name == "--n") o.n = x;
        else if (name == "--k") o.k = x;
        else if (name == "--warmup") o.warmup = static_cast<int>(x);
        else if (name == "--batches") o.batches = static_cast<int>(x);
        else if (name == "--repetitions") o.repetitions = static_cast<int>(x);
        else throw std::invalid_argument("unknown option " + name);
    }
    if (o.device < 0 || o.m <= 0 || o.n <= 0 || o.k <= 0 ||
        o.warmup < 0 || o.batches <= 0 || o.repetitions <= 0 ||
        o.m > 1000000 || o.n > 1000000 || o.k > 1000000 ||
        (o.api != "v3" && o.api != "v3cached" &&
         o.api != "aclnn" && o.api != "mm"))
        throw std::invalid_argument("device, dimensions, or repetition count out of range");
    return o;
}

void Check(int code, const char* call) {
    if (code != 0) {
        const char* detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(call) + " failed: " + std::to_string(code) +
                                 (detail ? std::string("; ") + detail : ""));
    }
}

std::size_t Bytes(std::int64_t a, std::int64_t b) {
    if (static_cast<std::uint64_t>(a) >
        std::numeric_limits<std::size_t>::max() / 2 / static_cast<std::uint64_t>(b))
        throw std::overflow_error("tensor allocation too large");
    return static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * 2;
}

struct DeviceAllocation {
    void* ptr = nullptr;
    explicit DeviceAllocation(std::size_t bytes) {
        if (bytes) Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~DeviceAllocation() { if (ptr) aclrtFree(ptr); }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};

struct Tensor {
    aclTensor* ptr = nullptr;
    Tensor(void* data, std::int64_t rows, std::int64_t cols) {
        const std::int64_t shape[] = {rows, cols};
        const std::int64_t strides[] = {cols, 1};
        ptr = aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND,
                              shape, 2, data);
        if (!ptr) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { if (ptr) aclDestroyTensor(ptr); }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
};

struct OpTensor {
    aclTensorDesc* desc = nullptr;
    aclDataBuffer* buffer = nullptr;
    OpTensor(void* data, std::int64_t rows, std::int64_t cols) {
        const std::int64_t shape[] = {rows, cols};
        desc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
        if (!desc) throw std::runtime_error("aclCreateTensorDesc failed");
        buffer = aclCreateDataBuffer(data, Bytes(rows, cols));
        if (!buffer) throw std::runtime_error("aclCreateDataBuffer failed");
    }
    ~OpTensor() {
        if (buffer) aclDestroyDataBuffer(buffer);
        if (desc) aclDestroyTensorDesc(desc);
    }
    OpTensor(const OpTensor&) = delete;
    OpTensor& operator=(const OpTensor&) = delete;
};

struct OpAttributes {
    aclopAttr* ptr = nullptr;
    OpAttributes() {
        ptr = aclopCreateAttr();
        if (!ptr) throw std::runtime_error("aclopCreateAttr failed");
        Check(aclopSetAttrBool(ptr, "transpose_x1", 0), "transpose_x1");
        Check(aclopSetAttrBool(ptr, "transpose_x2", 0), "transpose_x2");
        Check(aclopSetAttrInt(ptr, "offset_x", 0), "offset_x");
        Check(aclopSetAttrInt(ptr, "opImplMode", 1), "opImplMode");
    }
    ~OpAttributes() { if (ptr) aclopDestroyAttr(ptr); }
    OpAttributes(const OpAttributes&) = delete;
    OpAttributes& operator=(const OpAttributes&) = delete;
};

void EnqueueV3(OpTensor& a, OpTensor& b, OpTensor& c,
               OpAttributes& attributes, aclrtStream stream, bool cached) {
    aclTensorDesc* input_desc[] = {a.desc, b.desc};
    aclDataBuffer* input_buffers[] = {a.buffer, b.buffer};
    aclTensorDesc* output_desc[] = {c.desc};
    aclDataBuffer* output_buffers[] = {c.buffer};
    if (cached)
        Check(aclopExecuteV2("MatMulV3", 2, input_desc, input_buffers, 1,
                             output_desc, output_buffers, attributes.ptr, stream),
              "aclopExecuteV2(MatMulV3) after compile");
    else
        Check(aclopCompileAndExecuteV2("MatMulV3", 2, input_desc, input_buffers, 1,
                                       output_desc, output_buffers, attributes.ptr,
                                       ACL_ENGINE_AICORE, ACL_COMPILE_SYS, nullptr,
                                       stream),
              "aclopCompileAndExecuteV2(MatMulV3)");
}

struct Event {
    aclrtEvent handle = nullptr;
    Event() { Check(aclrtCreateEvent(&handle), "aclrtCreateEvent"); }
    ~Event() { if (handle) aclrtDestroyEvent(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Context {
    aclrtStream stream = nullptr;
    bool initialized = false;
    bool device_set = false;
    explicit Context(int device) : device_(device) {
        Check(aclInit(nullptr), "aclInit");
        initialized = true;
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        device_set = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Context() {
        if (stream) aclrtDestroyStream(stream);
        if (device_set) aclrtResetDevice(device_);
        if (initialized) aclFinalize();
    }
private:
    int device_;
};

std::uint64_t WorkspaceSize(const Tensor& a, const Tensor& b, Tensor& c,
                            aclOpExecutor** executor, bool mm) {
    std::uint64_t size = 0;
    if (mm)
        Check(aclnnMmGetWorkspaceSize(a.ptr, b.ptr, c.ptr, 0, &size, executor),
              "aclnnMmGetWorkspaceSize");
    else
        Check(aclnnMatmulGetWorkspaceSize(a.ptr, b.ptr, c.ptr, 0, &size, executor),
              "aclnnMatmulGetWorkspaceSize");
    if (!*executor) throw std::runtime_error("aclnnMatmul returned null executor");
    return size;
}

void Enqueue(const Tensor& a, const Tensor& b, Tensor& c,
             DeviceAllocation& workspace, std::uint64_t workspace_size,
             aclrtStream stream, bool mm) {
    aclOpExecutor* executor = nullptr;
    const auto size = WorkspaceSize(a, b, c, &executor, mm);
    if (size != workspace_size)
        throw std::runtime_error("workspace size changed between identical calls");
    if (mm) Check(aclnnMm(workspace.ptr, size, executor, stream), "aclnnMm");
    else Check(aclnnMatmul(workspace.ptr, size, executor, stream), "aclnnMatmul");
}

double PositiveFp16(std::uint16_t bits) {
    if (bits & 0x8000) throw std::runtime_error("negative output in positive-input test");
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    if (exponent == 31) throw std::runtime_error("non-finite output");
    if (exponent == 0) return std::ldexp(static_cast<double>(fraction), -24);
    return std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
}

// Four exactly representable FP16 values create a nonuniform output while
// retaining a cheap, deterministic reference for every output element.
constexpr std::uint16_t kInputBits[] = {0x3000, 0x2c00, 0x2e00, 0x2800};
constexpr double kInputValues[] = {0.125, 0.0625, 0.09375, 0.03125};

double Expected(std::int64_t m, std::int64_t n, std::int64_t k) {
    double sum = 0;
    for (std::int64_t residue = 0; residue < 4; ++residue) {
        const auto count = (k + 3 - residue) / 4;
        sum += static_cast<double>(count) * kInputValues[(m + residue) % 4] *
               kInputValues[(n + residue) % 4];
    }
    return sum;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto o = Parse(argc, argv);
        Context context(o.device);
        const auto a_bytes = Bytes(o.m, o.k), b_bytes = Bytes(o.k, o.n);
        const auto c_bytes = Bytes(o.m, o.n);
        DeviceAllocation a_memory(a_bytes), b_memory(b_bytes), c_memory(c_bytes);
        std::vector<std::uint16_t> a_values(a_bytes / 2);
        std::vector<std::uint16_t> b_values(b_bytes / 2);
        for (std::int64_t row = 0; row < o.m; ++row)
            for (std::int64_t inner = 0; inner < o.k; ++inner)
                a_values[row * o.k + inner] = kInputBits[(row + inner) % 4];
        for (std::int64_t inner = 0; inner < o.k; ++inner)
            for (std::int64_t col = 0; col < o.n; ++col)
                b_values[inner * o.n + col] = kInputBits[(inner + col) % 4];
        Check(aclrtMemcpy(a_memory.ptr, a_bytes, a_values.data(), a_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
        Check(aclrtMemcpy(b_memory.ptr, b_bytes, b_values.data(), b_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
        Tensor a(a_memory.ptr, o.m, o.k), b(b_memory.ptr, o.k, o.n);
        Tensor c(c_memory.ptr, o.m, o.n);
        OpTensor va(a_memory.ptr, o.m, o.k), vb(b_memory.ptr, o.k, o.n);
        OpTensor vc(c_memory.ptr, o.m, o.n);
        OpAttributes attributes;
        std::uint64_t workspace_size = 0;
        aclOpExecutor* first_executor = nullptr;
        if (o.api != "v3" && o.api != "v3cached")
            workspace_size = WorkspaceSize(a, b, c, &first_executor, o.api == "mm");
        DeviceAllocation workspace(workspace_size);
        if (o.api == "v3" || o.api == "v3cached")
            EnqueueV3(va, vb, vc, attributes, context.stream, false);
        else if (o.api == "mm")
            Check(aclnnMm(workspace.ptr, workspace_size, first_executor,
                          context.stream), "first aclnnMm");
        else Check(aclnnMatmul(workspace.ptr, workspace_size, first_executor,
                               context.stream), "first aclnnMatmul");
        for (int i = 0; i < o.warmup; ++i) {
            if (o.api == "v3" || o.api == "v3cached")
                EnqueueV3(va, vb, vc, attributes, context.stream,
                          o.api == "v3cached");
            else Enqueue(a, b, c, workspace, workspace_size, context.stream,
                         o.api == "mm");
        }
        Check(aclrtSynchronizeStream(context.stream), "warmup synchronize");
        std::vector<std::uint16_t> output(c_bytes / 2);
        Check(aclrtMemcpy(output.data(), c_bytes, c_memory.ptr, c_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "copy result");
        double max_abs_error = 0;
        double max_relative_error = 0;
        std::int64_t first_bad_row = -1, first_bad_col = -1;
        double first_bad_actual = 0, first_bad_expected = 0;
        std::uint64_t bad_elements = 0;
        for (std::int64_t row = 0; row < o.m; ++row) {
            for (std::int64_t col = 0; col < o.n; ++col) {
                const double expected = Expected(row, col, o.k);
                const double actual = PositiveFp16(output[row * o.n + col]);
                const double error = std::abs(actual - expected);
                max_abs_error = std::max(max_abs_error, error);
                max_relative_error = std::max(max_relative_error, error / expected);
                if (error / expected > 0.02) {
                    if (first_bad_row < 0) {
                        first_bad_row = row;
                        first_bad_col = col;
                        first_bad_actual = actual;
                        first_bad_expected = expected;
                    }
                    ++bad_elements;
                }
            }
        }
        if (max_relative_error > 0.02)
            throw std::runtime_error("matmul output failed patterned-input check; max relative error " +
                                     std::to_string(max_relative_error) +
                                     ", bad elements " + std::to_string(bad_elements) +
                                     ", first at (" + std::to_string(first_bad_row) +
                                     "," + std::to_string(first_bad_col) +
                                     "): got " + std::to_string(first_bad_actual) +
                                     ", expected " + std::to_string(first_bad_expected));

        // Prepare the executor before recording the start event. Each event
        // interval brackets one device call; msprof should be used to check
        // whether host submission leaves a gap inside that interval.
        Event start, stop;
        const auto epoch = std::chrono::system_clock::now().time_since_epoch();
        const auto run_id = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
        std::cout << std::fixed << std::setprecision(6);
        for (int batch = 0; batch < o.batches; ++batch) {
            std::vector<double> samples;
            samples.reserve(o.repetitions);
            for (int rep = 0; rep < o.repetitions; ++rep) {
                aclOpExecutor* executor = nullptr;
                if (o.api != "v3" && o.api != "v3cached") {
                    const auto size = WorkspaceSize(a, b, c, &executor, o.api == "mm");
                    if (size != workspace_size)
                        throw std::runtime_error("workspace size changed");
                }
                Check(aclrtRecordEvent(start.handle, context.stream), "record start");
                if (o.api == "v3" || o.api == "v3cached")
                    EnqueueV3(va, vb, vc, attributes, context.stream,
                              o.api == "v3cached");
                else if (o.api == "mm")
                    Check(aclnnMm(workspace.ptr, workspace_size, executor,
                                  context.stream), "timed aclnnMm");
                else Check(aclnnMatmul(workspace.ptr, workspace_size, executor,
                                       context.stream), "timed aclnnMatmul");
                Check(aclrtRecordEvent(stop.handle, context.stream), "record stop");
                Check(aclrtSynchronizeStream(context.stream), "timed synchronize");
                float ms = 0;
                Check(aclrtEventElapsedTime(&ms, start.handle, stop.handle),
                      "aclrtEventElapsedTime");
                samples.push_back(static_cast<double>(ms) * 1000.0);
            }
            std::sort(samples.begin(), samples.end());
            const double median = samples[samples.size() / 2];
            std::cout << "{\"run_id\":" << run_id
                      << ",\"pid\":" << getpid()
                      << ",\"kind\":\"" << (o.api == "v3" || o.api == "v3cached" ?
                                                "matmul_v3_event" :
                                                o.api == "mm" ? "aclnn_mm_event" :
                                                                 "aclnn_matmul_event") << "\""
                      << ",\"device\":" << o.device
                      << ",\"m\":" << o.m << ",\"n\":" << o.n
                      << ",\"k\":" << o.k
                      << ",\"dtype\":\"fp16\""
                      << ",\"input_pattern\":\"mod4_exact_fp16\""
                      << ",\"batch\":" << batch
                      << ",\"repetitions\":" << o.repetitions
                      << ",\"warmup\":" << o.warmup
                      << ",\"workspace_bytes\":";
            if (o.api == "v3" || o.api == "v3cached") std::cout << "null";
            else std::cout << workspace_size;
            std::cout
                      << ",\"max_abs_error\":" << max_abs_error
                      << ",\"max_relative_error\":" << max_relative_error
                      << ",\"min_us\":" << samples.front()
                      << ",\"median_us\":" << median
                      << ",\"max_us\":" << samples.back()
                      << ",\"tiling_verified\":false}"
                      << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "bench910: " << e.what() << '\n';
        return 1;
    }
}
