#include "kernel_operator.h"
using namespace AscendC;

// Diagnostic MKN ladder. It preserves source-shaped GM->L1 A/B traffic and
// tile assignment, then adds L1->L0 (1), Cube (2), and Fixpipe (3).
// L0 replays one 128x64 prefix four times per K packet: hardware work is
// representative, but operand values and MatmulImpl buffering are not exact.
extern "C" __global__ __aicore__ void mode2_stage_ladder_kernel(
    GM_ADDR a_input, GM_ADDR b_input, GM_ADDR c_scratch, GM_ADDR token_output,
    uint32_t pitch, uint32_t logical_k, uint32_t m, uint32_t n,
    uint32_t core_m, uint32_t core_n, uint32_t inner_m,
    uint32_t packet_k, uint32_t mode) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint32_t m_tiles = (m + core_m - 1) / core_m;
    const uint32_t n_tiles = (n + core_n - 1) / core_n;
    const uint32_t tiles = m_tiles * n_tiles;
    const uint32_t round = (tiles + cores - 1) / cores;
    uint32_t pre = tiles % cores;
    if (pre == 0) pre = cores;
    const uint32_t first = core < pre ? core * round : core * (round - 1) + pre;
    const uint32_t count = core < pre ? round : round - 1;

    GlobalTensor<float> a, b, c;
    GlobalTensor<uint32_t> token;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b_input));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c_scratch));
    token.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(token_output));
    a.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    b.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);

    TPipe pipe;
    TBuf<TPosition::A1> a1_buf;
    TBuf<TPosition::B1> b1_buf;
    TBuf<TPosition::A2> a2_buf;
    TBuf<TPosition::B2> b2_buf;
    TBuf<TPosition::CO1> c_ping_buf, c_pong_buf;
    pipe.InitBuffer(a1_buf, inner_m * packet_k * sizeof(float));
    pipe.InitBuffer(b1_buf, packet_k * 128 * sizeof(float));
    pipe.InitBuffer(a2_buf, 32768);
    pipe.InitBuffer(b2_buf, 32768);
    pipe.InitBuffer(c_ping_buf, 65536);
    pipe.InitBuffer(c_pong_buf, 65536);
    auto a1 = a1_buf.Get<float>(), b1 = b1_buf.Get<float>();
    auto a2 = a2_buf.Get<float>(), b2 = b2_buf.Get<float>();
    auto c_ping = c_ping_buf.Get<float>(), c_pong = c_pong_buf.Get<float>();

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
    FixpipeParamsV220 fix;
    fix.nSize = 128;
    fix.srcStride = 128;
    fix.dstStride = 128;
    fix.ndNum = 1;
    const uint32_t scratch_core_stride = (core_n / 128) * 2 * 128 * 128;

    for (uint32_t tile = first; tile < first + count; ++tile) {
        const uint32_t m_index = tile / n_tiles, n_index = tile % n_tiles;
        const uint32_t m_begin = m_index * core_m, n_begin = n_index * core_n;
        const uint32_t tile_m = m - m_begin < core_m ? m - m_begin : core_m;
        const uint32_t tile_n = n - n_begin < core_n ? n - n_begin : core_n;
        for (uint32_t panel = 0; panel < tile_m; panel += inner_m) {
            copy_a.nValue = tile_m - panel < inner_m ? tile_m - panel : inner_m;
            copy_a.dstNzC0Stride = copy_a.nValue;
            const uint32_t m_pieces = (copy_a.nValue + 127) / 128;
            for (uint32_t k = 0; k < logical_k; k += packet_k) {
                const uint64_t a_offset =
                    static_cast<uint64_t>(m_begin + panel) * pitch + k;
                DataCopy(a1, a[a_offset], copy_a);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                for (uint32_t col = 0; col < tile_n; col += 128) {
                    const uint64_t b_offset =
                        static_cast<uint64_t>(k) * n + n_begin + col;
                    DataCopy(b1, b[b_offset], copy_b);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                    if (mode >= 1) {
                        for (uint32_t part = 0; part < packet_k / 64; ++part) {
                            LoadData(a2, a1, load);
                            LoadData(b2, b1, load);
                            if (mode >= 2) {
                                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
                                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                                Mmad(c_ping, a2, b2,
                                     MmadParams(128, 128, 64, 0, false, part == 0));
                                if (m_pieces == 2) {
                                    Mmad(c_pong, a2, b2,
                                         MmadParams(copy_a.nValue - 128, 128, 64,
                                                    0, false, part == 0));
                                }
                                // The same L0A/L0B addresses are overwritten
                                // for the next 64-wide fragment. Drain Cube
                                // before that write to avoid a read/write fault.
                                PipeBarrier<PIPE_ALL>();
                            } else {
                                PipeBarrier<PIPE_MTE1>();
                            }
                        }
                    }
                    if (mode == 3) {
                        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                        if (k != 0) SetAtomicAdd<float>();
                        const uint32_t base = core * scratch_core_stride +
                            (col / 128) * 2 * 128 * 128;
                        fix.mSize = copy_a.nValue < 128 ? copy_a.nValue : 128;
                        Fixpipe(c[base], c_ping, fix);
                        if (m_pieces == 2) {
                            fix.mSize = copy_a.nValue - 128;
                            Fixpipe(c[base + 128 * 128], c_pong, fix);
                        }
                        PipeBarrier<PIPE_FIX>();
                        if (k != 0) SetAtomicNone();
                    }
                    // All modes have the same B/L0C destination reuse wait.
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }
    const uint32_t completion = pitch + logical_k + m + n + core_m + core_n +
                                inner_m + packet_k + mode + core;
    token.SetValue(core * 8, completion);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(token[core * 8]);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void mode2_stage_ladder_do(uint32_t cores, void* stream,
    uint8_t* a_input, uint8_t* b_input, uint8_t* c_scratch,
    uint8_t* token_output, uint32_t pitch, uint32_t logical_k,
    uint32_t m, uint32_t n, uint32_t core_m, uint32_t core_n,
    uint32_t inner_m, uint32_t packet_k, uint32_t mode) {
    mode2_stage_ladder_kernel<<<cores, nullptr, stream>>>(
        a_input, b_input, c_scratch, token_output, pitch, logical_k,
        m, n, core_m, core_n, inner_m, packet_k, mode);
}
