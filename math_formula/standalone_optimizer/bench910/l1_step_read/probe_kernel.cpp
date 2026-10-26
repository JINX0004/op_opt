#include "kernel_operator.h"

using namespace AscendC;

// Probe one source-emitted L1 step, not merely one baseK=64 slice. The A
// packet is [baseM,stepKa*baseK], the B packet [stepKb*baseK,baseN].
// source pitch remains K for A and N for B. Reusing one L1 allocation and
// a barrier makes each AIC's MTE2 packet service serial.
extern "C" __global__ __aicore__ void l1_step_read_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t operand, uint32_t tile_axis,
    uint32_t k_span, uint32_t m, uint32_t n, uint32_t k, uint32_t copies) {
    if ASCEND_IS_AIV { return; }
    const uint32_t core = GetBlockIdx();
    const uint32_t rows = operand == 0 ? tile_axis : k_span;
    const uint32_t cols = operand == 0 ? k_span : tile_axis;
    const uint32_t pitch = operand == 0 ? k : n;
    const uint32_t row_base = operand == 0 ?
        (core % (m / tile_axis)) * tile_axis : 0;
    const uint32_t col_base = operand == 0 ? 0 :
        (core % (n / tile_axis)) * tile_axis;

    GlobalTensor<half> source;
    GlobalTensor<uint32_t> result;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    TPipe pipe;
    TBuf<TPosition::A1> a_l1_buffer;
    TBuf<TPosition::B1> b_l1_buffer;
    if (operand == 0) pipe.InitBuffer(a_l1_buffer, 262144);
    else pipe.InitBuffer(b_l1_buffer, 262144);
    Nd2NzParams nd2nz;
    nd2nz.ndNum = 1;
    nd2nz.nValue = rows;
    nd2nz.dValue = cols;
    nd2nz.srcNdMatrixStride = 0;
    nd2nz.srcDValue = pitch;
    nd2nz.dstNzC0Stride = rows;
    nd2nz.dstNzNStride = 1;
    nd2nz.dstNzMatrixStride = 0;
    for (uint32_t i = 0; i < copies; ++i) {
        const uint32_t k_offset = (i % (k / k_span)) * k_span;
        const uint32_t row = row_base + (operand == 0 ? 0 : k_offset);
        const uint32_t col = col_base + (operand == 0 ? k_offset : 0);
        if (operand == 0) {
            DataCopy(a_l1_buffer.Get<half>(), source[row * pitch + col], nd2nz);
        } else {
            DataCopy(b_l1_buffer.Get<half>(), source[row * pitch + col], nd2nz);
        }
        PipeBarrier<PIPE_MTE2>();
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    result.SetValue(core * 8, operand + tile_axis + k_span + copies + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void l1_step_read_do(uint32_t cores, void* stream,
    uint8_t* input, uint8_t* output, uint32_t operand, uint32_t tile_axis,
    uint32_t k_span, uint32_t m, uint32_t n, uint32_t k, uint32_t copies) {
    l1_step_read_kernel<<<cores, nullptr, stream>>>(
        input, output, operand, tile_axis, k_span, m, n, k, copies);
}
