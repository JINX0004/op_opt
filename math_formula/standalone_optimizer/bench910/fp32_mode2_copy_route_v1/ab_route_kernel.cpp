#include "kernel_operator.h"

using namespace AscendC;

// Source-shaped MTE2 demand for the exact mode-2 128xN outer tile.
// One A panel and N/128 B panels are loaded per 256-K slice. The copies
// are serialized by MTE2 barriers, so this is an input-service probe,
// not a Matmul engine timeline or a claim about its hidden issue order.
extern "C" __global__ __aicore__ void ab_route_kernel(
    GM_ADDR a_input, GM_ADDR b_input, GM_ADDR output, uint32_t a_route,
    uint32_t pitch, uint32_t logical_k, uint32_t m, uint32_t n,
    uint32_t repeats) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    const uint32_t tiles = m / 128;
    const uint32_t round = (tiles + cores - 1) / cores;
    uint32_t pre = tiles % cores;
    if (pre == 0) pre = cores;
    const uint32_t first = core < pre ? core * round : core * (round - 1) + pre;
    const uint32_t count = core < pre ? round : round - 1;

    GlobalTensor<float> a;
    GlobalTensor<float> b;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b_input));
    a.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    b.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    GlobalTensor<uint32_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    TPipe pipe;
    TBuf<TPosition::A1> a_l1_buffer;
    TBuf<TPosition::B1> b_l1_buffer;
    pipe.InitBuffer(a_l1_buffer, 128 * 256 * sizeof(float));
    pipe.InitBuffer(b_l1_buffer, 256 * 128 * sizeof(float));
    auto a_l1 = a_l1_buffer.Get<float>();
    auto b_l1 = b_l1_buffer.Get<float>();

    Nd2NzParams copy_a;
    copy_a.ndNum = 1;
    copy_a.nValue = 128;
    copy_a.dValue = 256;
    copy_a.srcNdMatrixStride = 0;
    copy_a.srcDValue = pitch;
    copy_a.dstNzC0Stride = 128;
    copy_a.dstNzNStride = 1;
    copy_a.dstNzMatrixStride = 0;
    DataCopyParams plain_a(128, 32, (pitch - 256) / 8, 0);

    Nd2NzParams copy_b;
    copy_b.ndNum = 1;
    copy_b.nValue = 256;
    copy_b.dValue = 128;
    copy_b.srcNdMatrixStride = 0;
    copy_b.srcDValue = n;
    copy_b.dstNzC0Stride = 256;
    copy_b.dstNzNStride = 1;
    copy_b.dstNzMatrixStride = 0;

    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (uint32_t tile = first; tile < first + count; ++tile) {
            const uint64_t a_row = static_cast<uint64_t>(tile) * 128 * pitch;
            for (uint32_t k = 0; k < logical_k; k += 256) {
                if (a_route == 0) {
                    DataCopy(a_l1, a[a_row + k], copy_a);
                } else {
                    DataCopy(a_l1, a[a_row + k], plain_a);
                }
                PipeBarrier<PIPE_MTE2>();
                for (uint32_t col = 0; col < n; col += 128) {
                    DataCopy(b_l1, b[static_cast<uint64_t>(k) * n + col], copy_b);
                    PipeBarrier<PIPE_MTE2>();
                }
            }
        }
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    result.SetValue(core * 8, a_route + pitch + logical_k + m + n + repeats + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void ab_route_probe_do(uint32_t cores, void* stream,
    uint8_t* a_input, uint8_t* b_input, uint8_t* output,
    uint32_t a_route, uint32_t pitch, uint32_t logical_k,
    uint32_t m, uint32_t n, uint32_t repeats) {
    ab_route_kernel<<<cores, nullptr, stream>>>(
        a_input, b_input, output, a_route, pitch, logical_k, m, n, repeats);
}
