#include "/home/xianfei/dev/ops-nn-sribd-mip-selector/matmul/mat_mul_v3/op_kernel/mat_mul_nd2nz.h"

using namespace AscendC;

// Source MatrixAtoNZV2 vector path. The source's mixed kernel launches
// 40 AIVs alongside 20 AICs; this standalone probe launches 40 AIVs and
// omits the AIC handoff. Mode zero measures launch/token overhead.
extern "C" __global__ __aicore__ void mmv3_converted_a_probe_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t m, uint32_t k,
    uint32_t base_an, uint32_t base_ad, uint32_t mode) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> ub_buffer;
    pipe.InitBuffer(ub_buffer, TOTAL_UB_SIZE);
    TCubeTiling cfg{};
    cfg.M = m;
    cfg.Ka = k;
    cfg.usedCoreNum = 40 / GetTaskRation();
    if (mode == 1) {
        MatrixAtoNZV2<half>(output, input, cfg, false, ub_buffer, base_an, base_ad);
    }
    GlobalTensor<uint32_t> token;
    token.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output + 2ULL * m * k));
    token.SetValue(GetBlockIdx() * 8, m + k + base_an + base_ad + mode + GetBlockIdx());
    PipeBarrier<PIPE_ALL>();
}

extern "C" void mmv3_converted_a_probe_do(void* stream, uint8_t* input,
    uint8_t* output, uint32_t m, uint32_t k, uint32_t base_an,
    uint32_t base_ad, uint32_t mode) {
    mmv3_converted_a_probe_kernel<<<40, nullptr, stream>>>(
        input, output, m, k, base_an, base_ad, mode);
}
