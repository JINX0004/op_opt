#include "kernel_operator.h"

using namespace AscendC;

// A controlled FP32 producer/consumer probe. Modes 1-3 wait for each
// GM->L1 copy, load a fixed 32-KiB NZ prefix into L0, then wait before
// reusing the L1 destination. This exposes a real L1 buffer dependency;
// it is not an implementation of MatmulImpl's hidden Cube/Fixpipe path.
// 0: A+B copy only, packet barrier (reference)
// 1: A copy and L0A consume
// 2: B copies and L0B consume
// 3: A+B copies with L0A/L0B consume
extern "C" __global__ __aicore__ void mode2_consumer_kernel(
    GM_ADDR a_input, GM_ADDR b_input, GM_ADDR output,
    uint32_t pitch, uint32_t logical_k, uint32_t m, uint32_t n,
    uint32_t core_m, uint32_t core_n, uint32_t inner_m,
    uint32_t packet_k, uint32_t a_cache_disable, uint32_t mode) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    const uint32_t m_tiles = (m + core_m - 1) / core_m;
    const uint32_t n_tiles = (n + core_n - 1) / core_n;
    const uint32_t tiles = m_tiles * n_tiles;
    const uint32_t round = (tiles + cores - 1) / cores;
    uint32_t pre = tiles % cores;
    if (pre == 0) pre = cores;
    const uint32_t first = core < pre ? core * round : core * (round - 1) + pre;
    const uint32_t count = core < pre ? round : round - 1;

    GlobalTensor<float> a;
    GlobalTensor<float> b;
    GlobalTensor<uint32_t> result;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b_input));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    if (a_cache_disable) a.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    else a.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    b.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);

    TPipe pipe;
    TBuf<TPosition::A1> a_buffer;
    TBuf<TPosition::B1> b_buffer;
    TBuf<TPosition::A2> a_l0_buffer;
    TBuf<TPosition::B2> b_l0_buffer;
    pipe.InitBuffer(a_buffer, inner_m * packet_k * sizeof(float));
    pipe.InitBuffer(b_buffer, packet_k * 128 * sizeof(float));
    pipe.InitBuffer(a_l0_buffer, 32768);
    pipe.InitBuffer(b_l0_buffer, 32768);
    auto a_l1 = a_buffer.Get<float>();
    auto b_l1 = b_buffer.Get<float>();
    auto a_l0 = a_l0_buffer.Get<float>();
    auto b_l0 = b_l0_buffer.Get<float>();

    Nd2NzParams copy_a;
    copy_a.ndNum = 1;
    copy_a.dValue = packet_k;
    copy_a.srcNdMatrixStride = 0;
    copy_a.srcDValue = pitch;
    copy_a.dstNzNStride = 1;
    copy_a.dstNzMatrixStride = 0;
    Nd2NzParams copy_b;
    copy_b.ndNum = 1;
    copy_b.nValue = packet_k;
    copy_b.dValue = 128;
    copy_b.srcNdMatrixStride = 0;
    copy_b.srcDValue = n;
    copy_b.dstNzC0Stride = packet_k;
    copy_b.dstNzNStride = 1;
    copy_b.dstNzMatrixStride = 0;
    LoadData2DParams load;
    load.startIndex = 0;
    load.repeatTimes = 64;
    load.srcStride = 1;
    load.dstGap = 0;
    load.ifTranspose = false;

    for (uint32_t tile = first; tile < first + count; ++tile) {
        const uint32_t m_index = tile / n_tiles;
        const uint32_t n_index = tile % n_tiles;
        const uint32_t m_begin = m_index * core_m;
        const uint32_t n_begin = n_index * core_n;
        const uint32_t tile_m = m - m_begin < core_m ? m - m_begin : core_m;
        const uint32_t tile_n = n - n_begin < core_n ? n - n_begin : core_n;
        for (uint32_t panel = 0; panel < tile_m; panel += inner_m) {
            copy_a.nValue = tile_m - panel < inner_m ? tile_m - panel : inner_m;
            copy_a.dstNzC0Stride = copy_a.nValue;
            for (uint32_t k = 0; k < logical_k; k += packet_k) {
                if (mode != 2) {
                    const uint64_t a_offset =
                        static_cast<uint64_t>(m_begin + panel) * pitch + k;
                    DataCopy(a_l1, a[a_offset], copy_a);
                    if (mode != 0) {
                        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                        LoadData(a_l0, a_l1, load);
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
                }
                if (mode != 1) {
                    for (uint32_t col = 0; col < tile_n; col += 128) {
                        const uint64_t b_offset =
                            static_cast<uint64_t>(k) * n + n_begin + col;
                        DataCopy(b_l1, b[b_offset], copy_b);
                        if (mode != 0) {
                            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                            LoadData(b_l0, b_l1, load);
                            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                        }
                    }
                }
                if (mode == 0) PipeBarrier<PIPE_MTE2>();
            }
        }
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    result.SetValue(core * 8, pitch + logical_k + m + n + core_m + core_n +
                              inner_m + packet_k + a_cache_disable + mode + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void mode2_consumer_probe_do(uint32_t cores, void* stream,
    uint8_t* a_input, uint8_t* b_input, uint8_t* output,
    uint32_t pitch, uint32_t logical_k, uint32_t m, uint32_t n,
    uint32_t core_m, uint32_t core_n, uint32_t inner_m,
    uint32_t packet_k, uint32_t a_cache_disable, uint32_t mode) {
    mode2_consumer_kernel<<<cores, nullptr, stream>>>(
        a_input, b_input, output, pitch, logical_k, m, n,
        core_m, core_n, inner_m, packet_k, a_cache_disable, mode);
}
