#include "kernel_operator.h"

using namespace AscendC;

// Each AIC reads its assigned 128-row panels through the same physical
// addresses, packet order, and byte counts for both transfer routes.
// This is an A-only MTE2 probe, not a Matmul engine replay.
extern "C" __global__ __aicore__ void copy_route_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t route, uint32_t pitch,
    uint32_t logical_k, uint32_t m, uint32_t repeats) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    GlobalTensor<float> a;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    a.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    GlobalTensor<uint32_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    TPipe pipe;
    TBuf<TPosition::A1> l1_buffer;
    pipe.InitBuffer(l1_buffer, 128 * 256 * sizeof(float));
    auto l1 = l1_buffer.Get<float>();

    Nd2NzParams nd2nz;
    nd2nz.ndNum = 1;
    nd2nz.nValue = 128;
    nd2nz.dValue = 256;
    nd2nz.srcNdMatrixStride = 0;
    nd2nz.srcDValue = pitch;
    nd2nz.dstNzC0Stride = 128;
    nd2nz.dstNzNStride = 1;
    nd2nz.dstNzMatrixStride = 0;
    // DataCopyParams uses 32-byte blocks. The source gap is the distance
    // from the end of one 256-FP32-element row to the start of the next.
    DataCopyParams plain(128, 32, (pitch - 256) / 8, 0);

    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (uint32_t tile = core; tile < m / 128; tile += cores) {
            const uint64_t row_base = static_cast<uint64_t>(tile) * 128 * pitch;
            for (uint32_t k = 0; k < logical_k; k += 256) {
                if (route == 0) {
                    DataCopy(l1, a[row_base + k], nd2nz);
                } else {
                    DataCopy(l1, a[row_base + k], plain);
                }
                PipeBarrier<PIPE_MTE2>();
            }
        }
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    result.SetValue(core * 8, route + pitch + logical_k + m + repeats + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void copy_route_probe_do(uint32_t cores, void* stream,
    uint8_t* input, uint8_t* output, uint32_t route, uint32_t pitch,
    uint32_t logical_k, uint32_t m, uint32_t repeats) {
    copy_route_kernel<<<cores, nullptr, stream>>>(
        input, output, route, pitch, logical_k, m, repeats);
}
