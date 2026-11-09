#include "kernel_operator.h"

using namespace AscendC;

// Source-shaped MKN input service: inner-M panel, K packet, N base pieces.
// Synchronization 0: each copy; 1: each K packet; 2: each M panel.
// Copy destinations are reused in MTE2 issue order and never consumed;
// this probes outstanding copy scheduling, not Matmul's operand buffers.
extern "C" __global__ __aicore__ void panel_ab_kernel(
    GM_ADDR a_input, GM_ADDR b_input, GM_ADDR output,
    uint32_t pitch, uint32_t logical_k, uint32_t m, uint32_t n,
    uint32_t core_m, uint32_t core_n, uint32_t inner_m, uint32_t packet_k,
    uint32_t a_cache_disable, uint32_t sync_mode) {
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
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b_input));
    if (a_cache_disable) a.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    else a.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    b.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    GlobalTensor<uint32_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    TPipe pipe;
    TBuf<TPosition::A1> a_buffer;
    TBuf<TPosition::B1> b_buffer;
    pipe.InitBuffer(a_buffer, inner_m * packet_k * sizeof(float));
    pipe.InitBuffer(b_buffer, packet_k * 128 * sizeof(float));
    auto a_l1 = a_buffer.Get<float>();
    auto b_l1 = b_buffer.Get<float>();

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

    for (uint32_t tile = first; tile < first + count; ++tile) {
        const uint32_t m_index = tile / n_tiles;
        const uint32_t n_index = tile % n_tiles;
        const uint32_t m_begin = m_index * core_m;
        const uint32_t n_begin = n_index * core_n;
        const uint32_t tile_m = m - m_begin < core_m ? m - m_begin : core_m;
        const uint32_t tile_n = n - n_begin < core_n ? n - n_begin : core_n;
        for (uint32_t panel = 0; panel < tile_m; panel += inner_m) {
            const uint32_t panel_m = tile_m - panel < inner_m ? tile_m - panel : inner_m;
            copy_a.nValue = panel_m;
            copy_a.dstNzC0Stride = panel_m;
            for (uint32_t k = 0; k < logical_k; k += packet_k) {
                const uint64_t a_offset =
                    static_cast<uint64_t>(m_begin + panel) * pitch + k;
                DataCopy(a_l1, a[a_offset], copy_a);
                if (sync_mode == 0) PipeBarrier<PIPE_MTE2>();
                for (uint32_t col = 0; col < tile_n; col += 128) {
                    const uint64_t b_offset =
                        static_cast<uint64_t>(k) * n + n_begin + col;
                    DataCopy(b_l1, b[b_offset], copy_b);
                    if (sync_mode == 0) PipeBarrier<PIPE_MTE2>();
                }
                if (sync_mode == 1) PipeBarrier<PIPE_MTE2>();
            }
            if (sync_mode == 2) PipeBarrier<PIPE_MTE2>();
        }
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    result.SetValue(core * 8, pitch + logical_k + m + n + core_m + core_n +
                              inner_m + packet_k + a_cache_disable + sync_mode + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void panel_ab_probe_do(uint32_t cores, void* stream,
    uint8_t* a_input, uint8_t* b_input, uint8_t* output,
    uint32_t pitch, uint32_t logical_k, uint32_t m, uint32_t n,
    uint32_t core_m, uint32_t core_n, uint32_t inner_m, uint32_t packet_k,
    uint32_t a_cache_disable, uint32_t sync_mode) {
    panel_ab_kernel<<<cores, nullptr, stream>>>(
        a_input, b_input, output, pitch, logical_k, m, n,
        core_m, core_n, inner_m, packet_k, a_cache_disable, sync_mode);
}
