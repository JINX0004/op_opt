#include "kernel_operator.h"

using namespace AscendC;

// This is intentionally an AIC kernel. Each core issues the requested number
// of equal-sized MTE2 GM->L1 transfers into a private L1 allocation. The
// packet count and packet size are runtime arguments so compilation is fixed
// across the sweep. Each core writes a token to make the launch observable.
extern "C" __global__ __aicore__ void solvebase_probe_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t mode, uint32_t packets,
    uint32_t packet_bytes, uint32_t source_stride_bytes) {
    if ASCEND_IS_AIV { return; }
    GlobalTensor<uint16_t> source;
    GlobalTensor<uint32_t> result;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(input));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
    const uint32_t core = GetBlockIdx();
    if (mode == 1 || mode == 4) {
        TPipe pipe;
        TBuf<TPosition::A1> l1_buffer;
        pipe.InitBuffer(l1_buffer, mode == 4 ? 262144 : 131072);
        auto l1 = l1_buffer.Get<uint16_t>();
        for (uint32_t i = 0; i < packets; ++i) {
            const uint32_t src = (core * source_stride_bytes + i * packet_bytes) / 2;
            if (mode == 1) {
                DataCopy(l1, source[src], packet_bytes / 2);
            } else {
                // ND operand tile to NZ L1, as used by the BASE matmul path.
                // dValue=128, nValue varies with the requested packet bytes.
                Nd2NzParams nd2nz;
                nd2nz.ndNum = 1;
                nd2nz.nValue = packet_bytes / 256;
                nd2nz.dValue = 128;
                nd2nz.srcNdMatrixStride = 0;
                nd2nz.srcDValue = 128;
                nd2nz.dstNzC0Stride = packet_bytes / 256;
                nd2nz.dstNzNStride = 1;
                nd2nz.dstNzMatrixStride = 0;
                DataCopy(l1, source[src], nd2nz);
            }
            PipeBarrier<PIPE_MTE2>();
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    } else if (mode == 2 || mode == 3) {
        // Fixed source tiles in NZ layout. Read and L1->L0 setup stay constant
        // while mode 2 changes only the number of Cube accumulations. Mode 3
        // changes only the Fixpipe packet count and output dimensions.
        TPipe pipe;
        TBuf<TPosition::A1> a_l1_buffer;
        TBuf<TPosition::B1> b_l1_buffer;
        TBuf<TPosition::A2> a_l0_buffer;
        TBuf<TPosition::B2> b_l0_buffer;
        TBuf<TPosition::CO1> c_l0_buffer;
        pipe.InitBuffer(a_l1_buffer, 32768);
        pipe.InitBuffer(b_l1_buffer, 32768);
        pipe.InitBuffer(a_l0_buffer, 32768);
        pipe.InitBuffer(b_l0_buffer, 32768);
        pipe.InitBuffer(c_l0_buffer, 65536);
        auto a_l1 = a_l1_buffer.Get<half>();
        auto b_l1 = b_l1_buffer.Get<half>();
        auto a_l0 = a_l0_buffer.Get<half>();
        auto b_l0 = b_l0_buffer.Get<half>();
        auto c_l0 = c_l0_buffer.Get<float>();
        GlobalTensor<half> a_gm;
        GlobalTensor<half> b_gm;
        GlobalTensor<half> c_gm;
        a_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        b_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        c_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
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
        const uint32_t cube_repeats = mode == 2 ? packets : 1;
        for (uint32_t i = 0; i < cube_repeats; ++i) {
            Mmad(c_l0, a_l0, b_l0,
                 MmadParams(128, 128, 128, 0, false, i == 0));
        }
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 fp;
        fp.nSize = 128;
        fp.mSize = mode == 2 ? 128 : packet_bytes / 256;
        fp.srcStride = 128;
        fp.dstStride = 128;
        fp.ndNum = 1;
        fp.quantPre = QuantMode_t::F322F16;
        const uint32_t out_stride = mode == 2 ? 32768 : packets * packet_bytes;
        for (uint32_t i = 0; i < (mode == 2 ? 1U : packets); ++i) {
            const uint32_t dst = (32 * GetBlockNum() + core * out_stride + i * packet_bytes) / 2;
            Fixpipe(c_gm[dst], c_l0, fp);
            PipeBarrier<PIPE_FIX>();
        }
    }
    // AIC scalar GM stores must not share a 32-byte sector across cores.
    result.SetValue(core * 8, packets + packet_bytes + core);
    PipeBarrier<PIPE_ALL>();
}

extern "C" void solvebase_probe_do(uint32_t cores, void* stream,
    uint8_t* input, uint8_t* output, uint32_t mode, uint32_t packets,
    uint32_t packet_bytes, uint32_t source_stride_bytes) {
    solvebase_probe_kernel<<<cores, nullptr, stream>>>(
        input, output, mode, packets, packet_bytes, source_stride_bytes);
}
