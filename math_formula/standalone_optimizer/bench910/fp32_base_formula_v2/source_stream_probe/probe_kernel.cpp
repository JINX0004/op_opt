#include "kernel_operator.h"

using namespace AscendC;

// Replays one full GM->L1 operand stream per source-sized spatial output
// tile. Blocks get round-robin tiles; this is a controlled approximation of
// MatMulV3's L2-region assignment, not a full Matmul engine replay.
extern "C" __global__ __aicore__ void base_stream_probe_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t operand, uint32_t base_m,
    uint32_t base_n, uint32_t k_span, uint32_t m, uint32_t n, uint32_t k,
    uint32_t repeats) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx();
    const uint32_t cores = GetBlockNum();
    const uint32_t m_tiles = m / base_m;
    const uint32_t n_tiles = n / base_n;
    const uint32_t total_tiles = m_tiles * n_tiles;
    const uint32_t rows = operand == 0 ? base_m : k_span;
    const uint32_t cols = operand == 0 ? k_span : base_n;
    const uint32_t pitch = operand == 0 ? k : n;

    GlobalTensor<float> source;
    GlobalTensor<float> source_b;
    GlobalTensor<uint32_t> result;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    if (operand >= 2) {
        source_b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
            input + static_cast<uint64_t>(m) * k * sizeof(float)));
    }
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    TPipe pipe;
    TBuf<TPosition::A1> a_l1_buffer;
    TBuf<TPosition::B1> b_l1_buffer;
    if (operand == 0 || operand >= 2) pipe.InitBuffer(a_l1_buffer, 131072);
    if (operand == 1 || operand >= 2) pipe.InitBuffer(b_l1_buffer, 131072);
    Nd2NzParams nd2nz;
    nd2nz.ndNum = 1;
    nd2nz.nValue = rows;
    nd2nz.dValue = cols;
    nd2nz.srcNdMatrixStride = 0;
    nd2nz.srcDValue = pitch;
    nd2nz.dstNzC0Stride = rows;
    nd2nz.dstNzNStride = 1;
    nd2nz.dstNzMatrixStride = 0;
    Nd2NzParams params_a = nd2nz;
    params_a.nValue = base_m;
    params_a.dValue = k_span * 2;
    params_a.srcDValue = k;
    params_a.dstNzC0Stride = base_m;
    Nd2NzParams params_b = nd2nz;
    params_b.nValue = k_span;
    params_b.dValue = base_n;
    params_b.srcDValue = n;
    params_b.dstNzC0Stride = k_span;

    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const uint32_t assigned_slots = operand == 3 ? (m_tiles / 4) * cores : total_tiles;
        for (uint32_t slot = core; slot < assigned_slots; slot += cores) {
            // MatMulV3's fallback L2 region is four M tiles by up to five N
            // tiles. For these N=768/1024 cases it activates 12/16 AICs per
            // region, rather than distributing all spatial tiles over 20.
            if (operand == 3 && core >= 4 * n_tiles) continue;
            const uint32_t tile = operand == 3 ?
                (slot / cores) * 4 * n_tiles + core : slot;
            const uint32_t m_index = tile / n_tiles;
            const uint32_t n_index = tile % n_tiles;
            if (operand >= 2) {
                for (uint32_t k_offset = 0; k_offset < k; k_offset += k_span) {
                    if (k_offset % (k_span * 2) == 0) {
                        DataCopy(a_l1_buffer.Get<float>(),
                                 source[m_index * base_m * k + k_offset], params_a);
                        PipeBarrier<PIPE_MTE2>();
                    }
                    DataCopy(b_l1_buffer.Get<float>(),
                             source_b[k_offset * n + n_index * base_n], params_b);
                    PipeBarrier<PIPE_MTE2>();
                }
                continue;
            }
            for (uint32_t k_offset = 0; k_offset < k; k_offset += k_span) {
                const uint32_t row = operand == 0 ? m_index * base_m : k_offset;
                const uint32_t col = operand == 0 ? k_offset : n_index * base_n;
                if (operand == 0) {
                    DataCopy(a_l1_buffer.Get<float>(), source[row * pitch + col], nd2nz);
                } else {
                    DataCopy(b_l1_buffer.Get<float>(), source[row * pitch + col], nd2nz);
                }
                PipeBarrier<PIPE_MTE2>();
            }
        }
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    result.SetValue(core * 8, operand + base_m + base_n + k_span + repeats + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void base_stream_probe_do(uint32_t cores, void* stream,
    uint8_t* input, uint8_t* output, uint32_t operand, uint32_t base_m,
    uint32_t base_n, uint32_t k_span, uint32_t m, uint32_t n, uint32_t k,
    uint32_t repeats) {
    base_stream_probe_kernel<<<cores, nullptr, stream>>>(
        input, output, operand, base_m, base_n, k_span, m, n, k, repeats);
}
