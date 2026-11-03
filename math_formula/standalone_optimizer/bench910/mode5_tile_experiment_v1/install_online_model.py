"""Apply the measured mode-5 objective to the experimental 910 selector."""

from pathlib import Path

root = Path('/home/xianfei/dev/ops-nn-sribd-mip-selector/experimental/mip_selector/model')
solver_path = root / 'mip_solver.cpp'
profile_path = root / 'hardware_profile.h'
solver = solver_path.read_text()
profile = profile_path.read_text()


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one anchor, found {count}')
    return source.replace(old, new, 1)


solver = replace_once(solver,
    '''    std::optional<PrimitiveTiming> sc_fp32_store, sc_fp32_store_steady;
    std::optional<PrimitiveTiming> sc_atomic_extra, sc_cast;''',
    '''    std::optional<PrimitiveTiming> sc_fp32_store, sc_fp32_store_steady;
    // Source-in-context mode-5 total output and per-IterateAll setup.
    std::optional<PrimitiveTiming> sc_mode5_fixpipe, sc_mode5_iterate;
    std::optional<PrimitiveTiming> sc_atomic_extra, sc_cast;''',
    'Rates fields')
solver = replace_once(solver,
    '0.0, {}, {}, {}, {}, {}, {}, {}, {}};',
    '0.0, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};',
    'Rates initialization')
solver = replace_once(solver,
    '''    if (hardware.primitives.count("sc_fp32_store_steady"))
        r.sc_fp32_store_steady = Get(hardware, "sc_fp32_store_steady", "bytes");
    if (hardware.primitives.count("sc_cast"))''',
    '''    if (hardware.primitives.count("sc_fp32_store_steady"))
        r.sc_fp32_store_steady = Get(hardware, "sc_fp32_store_steady", "bytes");
    const bool mode5_fix = hardware.primitives.count("sc_mode5_fixpipe") != 0;
    const bool mode5_iterate = hardware.primitives.count("sc_mode5_iterate") != 0;
    if (mode5_fix != mode5_iterate)
        throw std::invalid_argument("sc_mode5_fixpipe and sc_mode5_iterate must be supplied together");
    if (mode5_fix) {
        r.sc_mode5_fixpipe = Get(hardware, "sc_mode5_fixpipe", "bytes");
        r.sc_mode5_iterate = Get(hardware, "sc_mode5_iterate", "count");
    }
    if (hardware.primitives.count("sc_cast"))''',
    'ReadRates')
solver = replace_once(solver,
    '''double ScMode6ReadUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                     std::uint64_t k) {''',
    '''double ScMode5SliceUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                      std::uint64_t k) {
    // Source-in-context Fixpipe demand includes repeated FP32 updates.
    // MTE2, MAC and Fixpipe overlap; each IterateAll exposes setup work.
    const double mb = static_cast<double>(CeilDiv(m, 128));
    const double nb = static_cast<double>(CeilDiv(n, 128));
    const double qk = static_cast<double>(CeilDiv(k, 128));
    const double cube = 2.0 * m * static_cast<double>(n) * k /
        r.cube.rate_per_us;
    const double input = 2.0 * static_cast<double>(k) * (m * nb + n * mb);
    const double read = input / r.read.rate_per_us +
        2.0 * mb * nb * qk * r.read.fixed_latency_us;
    const double fixpipe = r.sc_mode5_fixpipe->fixed_latency_us * mb * nb +
        4.0 * m * static_cast<double>(n) /
            r.sc_mode5_fixpipe->rate_per_us;
    return std::max({cube, read, fixpipe}) +
        r.sc_mode5_iterate->fixed_latency_us;
}

double ScMode6ReadUs(const Rates& r, std::uint64_t m, std::uint64_t n,
                     std::uint64_t k) {''',
    'ScMode5SliceUs')
solver = replace_once(solver,
    '''    const auto tail = w.k - (packets - 1) * d.single_core_k;
    if (d.family != KernelFamily::SingleCoreSplitKMode6) {''',
    '''    const auto tail = w.k - (packets - 1) * d.single_core_k;
    if (w.dtype == "fp16" &&
        d.family == KernelFamily::SingleCoreSplitKMode5 &&
        r.sc_mode5_fixpipe) {
        const auto full = ScMode5SliceUs(r, m, n, d.single_core_k);
        return (packets - 1) * full + ScMode5SliceUs(r, m, n, tail);
    }
    if (d.family != KernelFamily::SingleCoreSplitKMode6) {''',
    'ScKChainUs')
profile = replace_once(profile,
    '''             {"sc_fp32_store_steady", {"bytes", 0.0, 91000.0}},
             {"sc_atomic_extra"''',
    '''             {"sc_fp32_store_steady", {"bytes", 0.0, 91000.0}},
             // 49152x384x1536 source-in-context mode-5 measurements.
             {"sc_mode5_fixpipe", {"bytes", 0.0, 108000.0}},
             {"sc_mode5_iterate", {"count", 1.8, 1.0}},
             {"sc_atomic_extra"''',
    'hardware profile')
solver_path.write_text(solver)
profile_path.write_text(profile)
print('updated', solver_path, profile_path)
