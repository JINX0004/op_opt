#include "kernel_operator.h"

using namespace AscendC;

// Output-only stream at the source mode-6 tile dimensions. All AICs own
// disjoint workspaces. Two precomputed 128x128 L0C tiles alternate as
// Fixpipe sources; this isolates output service from Cube and operand copies.
extern "C" __global__ __aicore__ void mode6_output_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t k_packets, uint32_t n_tiles,
    uint32_t serial_barrier, uint32_t distinct_k, uint32_t atomic) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx();
    const uint32_t core_n = n_tiles * 128;
    const uint32_t global_n = 2 * core_n;
    const uint32_t global_m = 2048;
    const uint32_t m_index = core % 10;
    const uint32_t n_index = core / 10;
    const uint32_t core_m = m_index == 9 ? 176 : 208;
    const uint32_t plane_elements = global_m * global_n;

    GlobalTensor<float> workspace;
    GlobalTensor<uint32_t> token;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output + 32 * GetBlockNum()));
    token.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));

    TPipe pipe;
    TBuf<TPosition::A1> a_l1_buffer;
    TBuf<TPosition::B1> b_l1_buffer;
    TBuf<TPosition::A2> a_l0_buffer;
    TBuf<TPosition::B2> b_l0_buffer;
    TBuf<TPosition::CO1> c_l0_buffer_ping;
    TBuf<TPosition::CO1> c_l0_buffer_pong;
    pipe.InitBuffer(a_l1_buffer, 32768);
    pipe.InitBuffer(b_l1_buffer, 32768);
    pipe.InitBuffer(a_l0_buffer, 32768);
    pipe.InitBuffer(b_l0_buffer, 32768);
    pipe.InitBuffer(c_l0_buffer_ping, 65536);
    pipe.InitBuffer(c_l0_buffer_pong, 65536);
    auto a_l1 = a_l1_buffer.Get<half>();
    auto b_l1 = b_l1_buffer.Get<half>();
    auto a_l0 = a_l0_buffer.Get<half>();
    auto b_l0 = b_l0_buffer.Get<half>();
    auto c_l0_ping = c_l0_buffer_ping.Get<float>();
    auto c_l0_pong = c_l0_buffer_pong.Get<float>();
    GlobalTensor<half> a_gm;
    GlobalTensor<half> b_gm;
    a_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    b_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    const uint32_t src = core * 65536 / 2;
    DataCopy(a_l1, a_gm[src], 16384);
    DataCopy(b_l1, b_gm[src + 16384], 16384);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData2DParams ld;
    ld.startIndex = 0;
    ld.repeatTimes = 64;
    ld.srcStride = 1;
    ld.dstGap = 0;
    ld.ifTranspose = false;
    LoadData(a_l0, a_l1, ld);
    LoadData(b_l0, b_l1, ld);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    Mmad(c_l0_ping, a_l0, b_l0, MmadParams(128, 128, 128, 0, false, true));
    Mmad(c_l0_pong, a_l0, b_l0, MmadParams(128, 128, 128, 0, false, true));
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

    FixpipeParamsV220 fp;
    fp.nSize = 128;
    fp.srcStride = 128;
    fp.dstStride = global_n;
    fp.ndNum = 1;
    if (atomic) SetAtomicAdd<float>();
    for (uint32_t k = 0; k < k_packets; ++k) {
        const uint32_t k_slot = distinct_k ? k : 0;
        const uint32_t base = k_slot * plane_elements +
            m_index * 208 * global_n + n_index * core_n;
        for (uint32_t n = 0; n < n_tiles; ++n) {
            for (uint32_t m = 0; m < 2; ++m) {
                fp.mSize = m == 0 ? 128 : core_m - 128;
                auto c_l0 = ((k * n_tiles * 2 + n * 2 + m) & 1) ?
                    c_l0_pong : c_l0_ping;
                Fixpipe(workspace[base + m * 128 * global_n + n * 128], c_l0, fp);
                if (serial_barrier) PipeBarrier<PIPE_FIX>();
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    if (atomic) SetAtomicNone();
    token.SetValue(core * 8, core + k_packets + n_tiles);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void mode6_output_do(uint32_t cores, void* stream,
    uint8_t* input, uint8_t* output, uint32_t k_packets, uint32_t n_tiles,
    uint32_t serial_barrier, uint32_t distinct_k, uint32_t atomic) {
    mode6_output_kernel<<<cores, nullptr, stream>>>(input, output, k_packets,
        n_tiles, serial_barrier, distinct_k, atomic);
}
