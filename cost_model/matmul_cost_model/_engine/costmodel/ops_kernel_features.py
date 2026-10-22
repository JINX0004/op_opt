"""Time-free source features for attention_matmul's FP16 NN direct ops wrapper.

The launch/block allocation and explicit conversion/reduction loops are source
derived. Installed CANN 8.5 headers also establish MK33 A reuse, Load3DV2 APIs,
and actual call boundaries. Remaining L1/L0 issue timing and service costs are
*schedule hypotheses*, not identified PMU services. A latency model
may fit context-effective costs to them, but must validate on held-out kernels.
No measured duration, throughput, or correctness label is consumed here.

``matmul_api_calls`` counts SetSingleShape + Iterate/IterateAll invocation
groups, not individual SDK function calls. ``control_iterations`` and
``sync_events`` are explicit scheduling proxies, not scalar instruction counts.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from math import prod

STRATEGIES = {"base": "BASE", "scsk": "SINGLE_CORE_SPLIT_K",
              "deterministic": "DETERMINISTIC_SPLIT_K"}
POLICY = "manual_conservative_v1"
SDK_SOURCE_ROOT = "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/asc/impl/adv_api/detail/matmul"
CONVERSION_API_KEYS = (
    "mte2_calls", "mte2_bursts", "mte3_calls", "mte3_bursts",
    "vector_api_calls", "nonzero_vector_api_calls", "vector_copy_calls", "vector_copy_repeats", "vector_copy_zero_repeat_calls",
    "transdata_calls", "transdata_repeats", "duplicate_calls", "duplicate_elements",
    "duplicate_zero_work_calls", "scalar_address_assignments", "scalar_loop_iterations",
    "set_flag_calls", "wait_flag_calls", "pipe_barrier_calls",
)
FEATURE_KEYS = (
    "mte2_bytes", "mte1_bytes", "cube_calls", "cube_blocks", "fixpipe_bytes",
    "atomic_bytes", "vector_add_elements", "cast_elements", "mte3_bytes",
    "nd2nz_elements", "control_iterations", "matmul_api_calls", "sync_events",
) + CONVERSION_API_KEYS

SINGLE_CORE_REDUCER_HAZARD = dict(
    hazard_id="deterministic_single_core_missing_mte2_v",
    condition="singleCoreNum == block_dim == 1 and FIXPIPE_OPT == BASE",
    source="mat_mul_deterministic_splitk_kernel.h:180-195; mat_mul_v3_common.h:196-202",
    evidence="First CopyGmToUbufAlign has no MTE2_V fence; the only fence is inside j=1..singleCoreNum-1. "
             "For one partial the loop is skipped, and PIPE_V then Cast do not establish MTE2-to-Vector readiness.",
    causal_status="strong source candidate consistent with canary failure; not hardware causal proof",
)


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(a: int, b: int = 16) -> int:
    return _ceil(a, b) * b


def _positive(**values) -> None:
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise ValueError("positive integer geometry required: " + ", ".join(values))


def _strategy(value: str) -> str:
    result = STRATEGIES.get(value, value)
    if result not in STRATEGIES.values():
        raise ValueError("unsupported ops strategy")
    return result


def _features(**values) -> dict:
    return {key: values.get(key, 0) for key in FEATURE_KEYS}


def _sum_features(rows) -> dict:
    result = _features()
    for row in rows:
        for key in FEATURE_KEYS:
            result[key] += row.get(key, 0)
    return result


def _partition(size: int, cores: int, index: int) -> range:
    """Contiguous balanced partition, matching preCoreNum/realRound source."""
    q, rem = divmod(size, cores)
    start = index * q + min(index, rem)
    return range(start, start + q + int(index < rem))


def ops_shape_support(m: int, n: int, k: int, strategy: str) -> dict:
    """Report this adapter's reviewed domain, never hardware qualification.

    K<384 deterministic is deliberately excluded after the historical K128 QK
    numerical failure. This is a conservative wrapper gate, not an assertion
    that upstream MatMulV3 cannot calculate a short-K GEMM.
    """
    _positive(m=m, n=n, k=k)
    strategy = _strategy(strategy)
    reasons = []
    if m > 128:
        reasons.append("current wrapper rejects M>128; large-M tiling needs a new policy")
    if n > 131072 or k > 131072:
        reasons.append("current wrapper rejects N/K>131072")
    if k % 16:
        reasons.append("current wrapper requires K divisible by 16")
    if strategy == "DETERMINISTIC_SPLIT_K" and k < 384:
        reasons.append("short-K MK33 unreviewed; historical 16x4096x128 numerical failure")
    pending = []
    if n > 128:
        pending.append("expanded N domain needs an independent numerical canary")
    if m not in (1, 16) or n not in (24, 64, 128) or k not in (4096, 32768, 102400):
        pending.append("shape outside the original PV experiment; no empirical support inferred")
    return dict(status="unsupported" if reasons else "source_reviewed_requires_canary",
                feature_supported=not reasons, reasons=reasons, validation_notes=pending,
                hardware_validated=False, known_source_hazards=[dict(SINGLE_CORE_REDUCER_HAZARD)]
                if strategy == "DETERMINISTIC_SPLIT_K" and k <= 384 else [])


def manual_ops_configuration(m: int, n: int, k: int, strategy: str,
                             physical_aic: int = 24) -> dict:
    """Reproduce MakePlan metadata for a prediction frozen before measurement."""
    _positive(m=m, n=n, k=k, physical_aic=physical_aic)
    strategy = _strategy(strategy)
    support = ops_shape_support(m, n, k, strategy)
    if not support["feature_supported"]:
        raise ValueError("; ".join(support["reasons"]))
    det = strategy == "DETERMINISTIC_SPLIT_K"
    single = [384, n, 384] if det else [128, 128, k if strategy == "BASE" else min(k, 512)]
    blocks = min(physical_aic, _ceil(k, single[2]) if det else _ceil(n, 128))
    return dict(backend="ops_nn", algorithm=strategy, actual_strategy=strategy,
                tiling_policy=POLICY, selection_mode="forced_experimental_direct",
                m=m, n=n, k=k, dtype="float16", output_dtype="float16",
                layout_a="row_major", layout_b="row_major", trans_a=False, trans_b=False,
                physical_aic=physical_aic, block_dim=blocks, base_tile=[128, 128, 128],
                single_core=single, steps=[3, 1, 3, 3] if det else [1, 1, 4, 4],
                nd2nz_a=int(k > 65535), nd2nz_b=int(n % 16 != 0 or n > 65535),
                k_partitions=_ceil(k, single[2]))


def _configuration(record: Mapping) -> dict:
    # Select a whitelist: never copy the input mapping wholesale, since a
    # measurement can contain arbitrary timing/label fields (including NaN).
    required = ("backend", "actual_strategy", "tiling_policy", "m", "n", "k",
                "dtype", "output_dtype", "layout_a", "layout_b", "trans_a", "trans_b",
                "physical_aic", "block_dim", "base_tile", "single_core", "steps",
                "nd2nz_a", "nd2nz_b")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("missing configuration fields: " + ", ".join(missing))
    _positive(block_dim=record["block_dim"])
    if type(record["trans_a"]) is not bool or type(record["trans_b"]) is not bool:
        raise ValueError("transpose metadata must be boolean")
    for key, length in (("base_tile", 3), ("single_core", 3), ("steps", 4)):
        if not isinstance(record[key], list) or len(record[key]) != length or any(
                type(v) is not int or v <= 0 for v in record[key]):
            raise ValueError("positive integral tile metadata required: " + key)
    expected = manual_ops_configuration(record["m"], record["n"], record["k"],
                                        record["actual_strategy"], record["physical_aic"])
    for key in required:
        if record[key] != expected[key]:
            raise ValueError(f"configuration does not match {POLICY}: {key}")
    for key in ("algorithm", "selection_mode", "k_partitions"):
        if key in record and record[key] != expected[key]:
            raise ValueError(f"inconsistent executed metadata: {key}")
    return expected


def _l1_tiles(m: int, n: int, k: int, step: list[int]) -> list[dict]:
    """Enumerate an explicit INNER_PRODUCT hypothesis, K then N then M.

    Operand bytes initially include reload per L1 tile. _chunk applies only
    the MK33 A reuse established by the installed SDK. Both request bytes and
    useful bytes are kept.
    Cube shapes are the effective tail shape, plus separate 16-rounded blocks.
    """
    result = []
    for ko in range(0, k, step[2]):
        tk = min(step[2], k - ko)
        for no in range(0, n, step[1]):
            tn = min(step[1], n - no)
            for mo in range(0, m, step[0]):
                tm = min(step[0], m - mo)
                shapes = Counter()
                a_bytes = b_bytes = 0
                for mi in range(0, tm, 128):
                    mm = min(128, tm - mi)
                    for ni in range(0, tn, 128):
                        nn = min(128, tn - ni)
                        for ki in range(0, tk, 128):
                            kk = min(128, tk - ki)
                            shapes[mm, nn, kk] += 1
                            a_bytes += 2 * _align(mm) * _align(kk)
                            b_bytes += 2 * _align(kk) * _align(nn)
                result.append(dict(m_offset=mo, n_offset=no, k_offset=ko, m=tm, n=tn, k=tk,
                                   mte2_a_bytes=2*tm*tk, mte2_b_bytes=2*tk*tn,
                                   mte1_a_bytes=a_bytes, mte1_b_bytes=b_bytes,
                                   # V220 FP16 MATRIX NN uses the loadInstr
                                   # specializations, not load2d.h's loops.
                                   # SDK stage/split/load_to_l0{a,b}/*loadInstr.h
                                   # A:109-120, B:160-179; counts per instance.
                                   cube_tiles=[dict(m=a, n=b, k=c, count=count,
                                                    mte1_a_primitive="LoadData3DV2",
                                                    mte1_b_primitive="LoadData3DV2",
                                                    mte1_a_api_calls=1, mte1_b_api_calls=1)
                                               for (a, b, c), count in shapes.items()]))
    return result


def _chunk(cfg: dict, ci: int, seq: int, no: int, ko: int, n: int, k: int,
           atomic: bool) -> dict:
    m, strategy = cfg["m"], cfg["actual_strategy"]
    step = [128*cfg["steps"][0], 128*cfg["steps"][1], 128*cfg["steps"][2]]
    tiles = _l1_tiles(m, n, k, step)
    if strategy == "DETERMINISTIC_SPLIT_K":
        # SDK scheduler_n_buffer.h:52-63 retains A until all N tiles finish;
        # cube_in_buffer_n_buffer.h:87-101 returns the cached M-slice buffer.
        # The next IterateAll resets this cache, so never reuse across chunks.
        loaded_a = set()
        for tile in tiles:
            key = f"core:{ci}:chunk:{seq}:a:m:{tile['m_offset']}:k:{tile['k_offset']}"
            hit = key in loaded_a
            tile["mte2_a_cache_key"] = key
            tile["mte2_a_cache_hit"] = hit
            tile["mte2_a_reuse_scope"] = "same_mk33_iterate_all_across_n"
            tile["mte2_a_logical_bytes"] = tile["mte2_a_bytes"]
            if hit:
                tile["mte2_a_bytes"] = 0
            loaded_a.add(key)
    calls = sum(t["count"] for b in tiles for t in b["cube_tiles"])
    blocks = sum(t["count"] * prod(_ceil(t[d], 16) for d in ("m", "n", "k"))
                 for b in tiles for t in b["cube_tiles"])
    out_bytes = 2 if strategy == "BASE" else 4
    flushes = _ceil(m, 128) * _ceil(n, 128)
    output_id = f"partial:{ci}" if strategy == "DETERMINISTIC_SPLIT_K" else f"output:0:{no}"
    return dict(sequence=seq, m_offset=0, n_offset=no, k_offset=ko, m=m, n=n, k=k,
                output_id=output_id, atomic=atomic, output_bytes_per_element=out_bytes,
                # direct_kernels.hpp calls Init/Process, never End; upstream
                # SetParamAndExec and deterministic kIndex loops also omit End.
                matmul_entry="Iterate_GetTensorC" if strategy == "BASE" else "IterateAll",
                call_end=False, call_boundary="api_call_without_end",
                l1_step=step, l0_tile=[128, 128, 128], l1_tiles=tiles,
                fixpipe_flushes=flushes, logical_a_unique_bytes=2*m*k,
                logical_b_unique_bytes=2*n*k,
                features=_features(
                    mte2_bytes=sum(t["mte2_a_bytes"] + t["mte2_b_bytes"] for t in tiles),
                    mte1_bytes=sum(t["mte1_a_bytes"] + t["mte1_b_bytes"] for t in tiles),
                    cube_calls=calls, cube_blocks=blocks, fixpipe_bytes=m*n*out_bytes,
                    atomic_bytes=m*n*out_bytes if atomic else 0,
                    control_iterations=len(tiles), matmul_api_calls=1))


def _matmul(cfg: dict) -> dict:
    m, n, k, cores = (cfg[x] for x in ("m", "n", "k", "block_dim"))
    strategy, chunk_k = cfg["actual_strategy"], cfg["single_core"][2]
    result = []
    for ci in range(cores):
        chunks = []
        if strategy == "DETERMINISTIC_SPLIT_K":
            for ki in _partition(_ceil(k, chunk_k), cores, ci):
                chunks.append(_chunk(cfg, ci, len(chunks), 0, ki*chunk_k, n,
                                     min(chunk_k, k-ki*chunk_k), bool(chunks)))
        else:
            # BASE UpdateBasicIndex is cyclic for mCnt=1; SCSK InitBlockIndex
            # assigns a contiguous interval. This matters after the first wave.
            nt = _ceil(n, 128)
            indices = range(ci, nt, cores) if strategy == "BASE" else _partition(nt, cores, ci)
            for ni in indices:
                for ki in range(_ceil(k, chunk_k)):
                    chunks.append(_chunk(cfg, ci, len(chunks), ni*128, ki*chunk_k,
                                         min(128, n-ni*128), min(chunk_k, k-ki*chunk_k), ki > 0))
        if chunks:
            chunks[-1]["process_final_barrier"] = "PIPE_ALL"
        # The aligned SCSK path additionally signals/casts after all K chunks
        # of each output tile (mat_mul_sc_splitk_kernel.h:218-230).
        if strategy == "SINGLE_CORE_SPLIT_K" and n % 128 == 0 and not (
                cfg["nd2nz_a"] or cfg["nd2nz_b"]):
            for index, chunk in enumerate(chunks):
                if index == len(chunks)-1 or chunks[index+1]["output_id"] != chunk["output_id"]:
                    chunk["output_tile_barrier"] = "PIPE_ALL"
        feature = _sum_features(c["features"] for c in chunks)
        feature["sync_events"] = (0 if strategy == "BASE" else
                                  1 if strategy == "DETERMINISTIC_SPLIT_K" else
                                  len({c["output_id"] for c in chunks}) if n % 128 == 0
                                  and not (cfg["nd2nz_a"] or cfg["nd2nz_b"]) else 2)
        result.append(dict(core_type="aic", core_id=ci, chunks=chunks, features=feature))
    return dict(phase_id="matmul", per_core=result,
                sdk_source_root=SDK_SOURCE_ROOT,
                l0c_buffer_depth=1 if strategy == "BASE" else 2,
                l0c_buffer_qualification=("source: unit-flag single TBuf" if strategy == "BASE" else
                                         "source: non-unit-flag TQue dbL0C=2" if strategy == "SINGLE_CORE_SPLIT_K" else
                                         "hypothesis: static config unit-flag derivation not fully traced"),
                detail_qualification="MK33 A reuse, Load3DV2 API geometry and call boundaries source-reviewed; issue/service timing remains a hypothesis")


def _conversion_api_work(path: str, h: int, w: int, th: int, tw: int,
                         ph: int, pw: int, h_buffer: int | None = None,
                         *, last_progress: bool = False) -> dict:
    """Exact visible helper API arguments, not decoded instruction counts.

    Zero-work Duplicate/Copy calls are retained because the C++ invokes them;
    separate counters allow a cost model to exclude likely compiler no-ops.
    Duplicate(count) has no explicit repeat argument, so its repeat_times is
    None rather than an invented hardware instruction decomposition.
    """
    counts = {key: 0 for key in CONVERSION_API_KEYS}
    mte2, mte3, vector = [], [], []

    def vec(api, *, repeat_times=None, mask_elements=None, elements=None, **params):
        if elements is None:
            elements = (repeat_times*256 if api == "TransDataTo5HD" else
                        mask_elements*repeat_times if mask_elements is not None else 0)
        vector.append(dict(api=api, repeat_times=repeat_times, mask_elements=mask_elements,
                           elements=elements, **params))
        counts["vector_api_calls"] += 1
        counts["nonzero_vector_api_calls"] += int(elements > 0)
        if api == "Copy":
            counts["vector_copy_calls"] += 1
            counts["vector_copy_repeats"] += repeat_times
            counts["vector_copy_zero_repeat_calls"] += int(repeat_times == 0)
        elif api == "TransDataTo5HD":
            counts["transdata_calls"] += 1
            counts["transdata_repeats"] += repeat_times
        else:
            counts["duplicate_calls"] += 1
            counts["duplicate_elements"] += elements
            counts["duplicate_zero_work_calls"] += int(elements == 0)

    geometry = dict(original_rows=h, original_columns=w, copy_rows=th, copy_columns=tw,
                    output_padded_rows=ph, output_padded_columns=pw)
    if path == "copy_pad_nd2nz":
        for row in range(0, th, 4095):
            mte2.append(dict(api="DataCopyPad", bursts=min(4095, th-row),
                             burst_bytes=tw*2, source_stride_bytes=(w-tw)*2,
                             destination_stride_bytes=0))
        if ph > th:
            vec("Duplicate_count", elements=(ph-th)*pw)
            counts["pipe_barrier_calls"] = 1
        for row in range(0, ph, 255):
            for col in range(0, pw, 128):
                vec("Copy", repeat_times=min(255, ph-row), mask_elements=min(128, pw-col),
                    dst_block_stride=ph, src_block_stride=1, dst_repeat_stride=1,
                    src_repeat_stride=pw//16)
        n_blocks = pw//16
        large_stride = _align(h)-ph > 65535
        if large_stride:
            mte3 = [dict(api="DataCopy", bursts=1, burst_bytes=ph*32,
                         source_stride_bytes=0, destination_stride_bytes=0) for _ in range(n_blocks)]
        else:
            mte3 = [dict(api="DataCopy", bursts=n_blocks, burst_bytes=ph*32,
                         source_stride_bytes=0, destination_stride_bytes=(_align(h)-ph)*32)]
        counts["set_flag_calls"] = counts["wait_flag_calls"] = 3
        # MatrixtoNZ outer loop + explicit for bodies inside CopyPadNd2Nz.
        counts["scalar_loop_iterations"] = (1 + th//4095 + ph//255 +
            ph//255*(pw//128) + (pw//128 if ph % 255 else 0) + (n_blocks if large_stride else 0))
        geometry.update(mte3_stride_overflow_fallback=large_stride, c0_elements=16)
    else:
        if not h_buffer or h_buffer % 16:
            raise ValueError("VNCHW requires explicit aligned hBuffer")
        h_blocks = h_buffer//16
        width_blocks = _ceil(w, 16)
        mte2 = [dict(api="DataCopyPad", bursts=1, burst_bytes=th*w*2,
                     source_stride_bytes=0, destination_stride_bytes=0)]
        mte3 = [dict(api="DataCopy_count", bursts=1, burst_bytes=ph*32,
                     source_stride_bytes=0, destination_stride_bytes=0) for _ in range(width_blocks)]
        geometry.update(h_buffer=h_buffer, h_block_num=h_blocks, width_blocks=width_blocks,
                        c0_elements=16, input_output_alias=path == "vnchw_out_db_reuse")
        if w % 16 == 0:
            repeat = h_blocks*2
            for _ in range(w//16):
                for _ in range(repeat//255):
                    vec("Copy", repeat_times=255, mask_elements=128,
                        dst_block_stride=1, src_block_stride=w//16,
                        dst_repeat_stride=8, src_repeat_stride=8*w//16)
                # PadDAligned makes this API call even if loopTail is zero.
                vec("Copy", repeat_times=repeat % 255, mask_elements=128,
                    dst_block_stride=1, src_block_stride=w//16,
                    dst_repeat_stride=8, src_repeat_stride=8*w//16)
            counts["scalar_loop_iterations"] = 1 + w//16 + w//16*(repeat//255) + width_blocks
            counts["set_flag_calls"] = counts["wait_flag_calls"] = 4
        else:
            block_ele = (1, 16, 8, 16, 4, 16, 8, 16, 2, 16, 8, 16, 4, 16, 8, 16)[w % 16]
            gcd = (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[w % 16]
            copy_repeat = (h_blocks//block_ele)*w//gcd
            geometry["copy_in_repeat"] = copy_repeat
            vec("TransDataTo5HD", repeat_times=copy_repeat,
                dst_repeat_stride=0 if copy_repeat == 1 else 16,
                src_repeat_stride=0 if copy_repeat == 1 else 1)
            for _ in range(w//16):
                vec("TransDataTo5HD", repeat_times=h_blocks,
                    dst_repeat_stride=0 if h_blocks == 1 else 1,
                    src_repeat_stride=0 if h_blocks == 1 else w)
            duplicate_loops = 0
            if w > 16:
                if 8*w > 255:
                    duplicate_loops = h_blocks//8
                    for _ in range(duplicate_loops):
                        vec("Duplicate", repeat_times=1, mask_elements=128,
                            dst_block_stride=w, dst_repeat_stride=8*w)
                else:
                    vec("Duplicate", repeat_times=h_blocks//8, mask_elements=128,
                        dst_block_stride=w, dst_repeat_stride=8*w)
                vec("Duplicate", repeat_times=1, mask_elements=16*(h_blocks % 8),
                    dst_block_stride=w, dst_repeat_stride=0)
            vec("TransDataTo5HD", repeat_times=h_blocks, dst_repeat_stride=1, src_repeat_stride=w)
            counts["scalar_address_assignments"] = 32*(w//16+2)
            counts["scalar_loop_iterations"] = 1 + 16 + (w//16)*17 + duplicate_loops + 32 + width_blocks
            counts["set_flag_calls"] = counts["wait_flag_calls"] = 3
        if last_progress:
            # CopyOutMM has one Duplicate(count) per output column block even
            # when original h is 16-aligned and padSize is zero.
            for _ in range(width_blocks):
                vec("Duplicate_count", elements=(_align(h)-h)*16)
            counts["scalar_loop_iterations"] += width_blocks
    counts["mte2_calls"], counts["mte3_calls"] = len(mte2), len(mte3)
    counts["mte2_bursts"] = sum(v["bursts"] for v in mte2)
    counts["mte3_bursts"] = sum(v["bursts"] for v in mte3)
    return dict(geometry=geometry, mte2=mte2, mte3=mte3, vector=vector, counts=counts,
                qualification="visible C++ API calls/repeat arguments, not ISA instructions or transactions")


def _conversion_setup(path: str, h_buffer: int, width: int) -> dict:
    """Per-core helper setup/drain, including idle participants; charged once."""
    counts = {key: 0 for key in CONVERSION_API_KEYS}
    flags = 4 if path == "vnchw_in_out_db" else 2
    counts["set_flag_calls"] = counts["wait_flag_calls"] = flags
    counts["pipe_barrier_calls"] = 1
    if width < 16:
        counts["vector_api_calls"] = counts["nonzero_vector_api_calls"] = counts["duplicate_calls"] = 1
        counts["duplicate_elements"] = h_buffer*width
    return dict(path=path, counts=counts, zero_buffer_elements=h_buffer*width if width < 16 else 0,
                qualification="SetBufMM plus Process setup/drain, once per AIV and operand")


def _conversion(cfg: dict, ub_bytes: int) -> dict:
    cores = 2*cfg["block_dim"]
    rows = [dict(core_type="aiv", core_id=i, operations=[], conversion_groups=[], features=_features())
            for i in range(cores)]
    for operand, h, w, enabled in (("a", cfg["m"], cfg["k"], cfg["nd2nz_a"]),
                                   ("b", cfg["k"], cfg["n"], cfg["nd2nz_b"])):
        if not enabled:
            continue
        vnchw = h > 8192 and w > 1 and w != 16 and (
            w*2 <= 192 or (w*2 <= 384 and w % 2 == 0) or (w*2 <= 512 and w % 4 == 0))
        if vnchw:
            # KernelND2NZMM::SetBufMM/ProcessMM, half input, batch=1.
            aligned_w = _align(w)
            total_width = 3*w + 2*aligned_w if w % 16 == 0 else w + 2*aligned_w + (w if w < 16 else 0)
            blocks = (1, 16, 8, 16, 4, 16, 8, 16, 2, 16, 8, 16, 4, 16, 8, 16)[w % 16]
            ele = min(_ceil(h, blocks*16), (ub_bytes//2//total_width)//(blocks*16))
            if ele*blocks > 255:
                ele = 255//blocks
            height = ele*blocks*16
            if not height:
                raise ValueError("VNCHW UB setup outside reviewed domain")
            tasks = [(idx % cores, idx*height, 0, min(height, h-idx*height), w,
                      _align(min(height, h-idx*height)), aligned_w)
                     for idx in range(_ceil(h, height))]
            path = "vnchw_in_out_db" if w % 16 == 0 else "vnchw_out_db_reuse"
        else:
            bh, bw = min(64, _align(h)), min(512, _align(w))
            nh, nw = _ceil(h, bh), _ceil(w, bw)
            tasks = []
            for ci in range(cores):
                for idx in _partition(nh*nw, cores, ci):
                    hi, wi = divmod(idx, nw)
                    th, tw = min(bh, h-hi*bh), min(bw, w-wi*bw)
                    tasks.append((ci, hi*bh, wi*bw, th, tw, _align(th), _align(tw)))
            path = "copy_pad_nd2nz"
        for row in rows:
            setup = (_conversion_setup(path, height, w) if vnchw else
                     dict(path=path, counts={key: 0 for key in CONVERSION_API_KEYS}))
            row["conversion_groups"].append(dict(
                operand=operand, path=path, sequence=len(row["conversion_groups"]),
                buffer_depth=2 if vnchw else 1,
                buffer_roles=(dict(input_slots=2, output_slots=2, shared_mid_slots=1,
                                   input_output_alias=path == "vnchw_out_db_reuse") if vnchw else
                              dict(input_slots=1, transform_slots=1, input_output_alias=False)),
                previous_operand_must_finish=True, operations=[], setup=setup))
        for ci, ho, wo, th, tw, ph, pw in tasks:
            # VNCHW PadDMain uses full hBuffer even on the last short CopyIn;
            # CopyOut writes only the padded actual tail. Keep work and traffic
            # separate rather than making the tail artificially free.
            converted_elements = height*pw if vnchw else ph*pw
            f = _features(mte2_bytes=th*tw*2, mte3_bytes=ph*pw*2,
                          nd2nz_elements=converted_elements, control_iterations=1,
                          sync_events=3)  # source-level producer/consumer boundaries
            api_work = _conversion_api_work(path, h, w, th, tw, ph, pw,
                height if vnchw else None, last_progress=bool(vnchw and th < height))
            f.update(api_work["counts"])
            operation = dict(operand=operand, path=path,
                row_offset=ho, col_offset=wo, rows=th, cols=tw, padded_rows=ph,
                padded_cols=pw, conversion_elements=converted_elements, api_work=api_work, features=f)
            rows[ci]["operations"].append(operation)
            rows[ci]["conversion_groups"][-1]["operations"].append(operation)
    for row in rows:
        for group in row["conversion_groups"]:
            group["features"] = _sum_features([group["setup"]["counts"]] +
                                               [op["features"] for op in group["operations"]])
        row["features"] = _sum_features(group["features"] for group in row["conversion_groups"])
        if cfg["nd2nz_a"] or cfg["nd2nz_b"]:
            row["features"]["sync_events"] += 2  # SyncAll then C/V notification
    return dict(phase_id="nd2nz", per_core=rows,
                conversion_api_features_version=1,
                detail_qualification="API descriptors expose helper calls/repeats; scalar counts are source loops/assignments, not ISA",
                composition="ordered operand groups per AIV; copy-pad tiles serial, VNCHW groups have two output credits")


def _reduce(cfg: dict, matmul: dict, ub_bytes: int) -> dict:
    m, n, cores = cfg["m"], cfg["n"], cfg["block_dim"]
    strategy = cfg["actual_strategy"]
    rows = [dict(core_type="aiv", core_id=i, operations=[], features=_features()) for i in range(2*cores)]
    def add(ci, output_id, count, *, partials=1, cast_count=None, offset=0, depends_on=None):
        if count <= 0:
            return
        cast_count = count if cast_count is None else cast_count
        capacity = ub_bytes//16 if strategy == "DETERMINISTIC_SPLIT_K" else ub_bytes//8
        f = _features(mte2_bytes=4*cast_count*partials,
                      vector_add_elements=(partials-1)*count, cast_elements=cast_count,
                      mte3_bytes=2*count, control_iterations=_ceil(cast_count, capacity),
                      sync_events=_ceil(cast_count, capacity)*(2*partials+1))
        rows[ci]["operations"].append(dict(output_id=output_id, output_offset=offset,
            elements=count, partial_count=partials, mte2_bytes=f["mte2_bytes"],
            vector_add_elements=f["vector_add_elements"], cast_elements=cast_count,
            mte3_bytes=f["mte3_bytes"], depends_on=depends_on or [], features=f))
    boundary = "none"
    if strategy == "DETERMINISTIC_SPLIT_K":
        boundary = "all_partial_cores"
        # ReduceKInUb orderFlag=false uses nominal singleCoreM, NOT actual M!
        span = _ceil(384*n, 2*cores)
        for ci in range(2*cores):
            count = max(0, min(span, m*n-ci*span))
            add(ci, "output", count, partials=cores, offset=ci*span,
                depends_on=[f"partial:{j}" for j in range(cores)])
    elif strategy == "SINGLE_CORE_SPLIT_K":
        if n % 128 == 0 and not (cfg["nd2nz_a"] or cfg["nd2nz_b"]):
            boundary = "per_output_tile"
            for core in matmul["per_core"]:
                outputs = {c["output_id"]: c for c in core["chunks"]}
                for output_id, c in outputs.items():
                    top = _ceil(m, 2)
                    add(core["core_id"]*2, output_id, top*c["n"],
                        offset=c["n_offset"], depends_on=[output_id])
                    add(core["core_id"]*2+1, output_id, (m-top)*c["n"],
                        offset=top*n+c["n_offset"], depends_on=[output_id])
        elif n % 64:
            boundary = "all_matmul_cores"
            split = max(1, m//(2*cores))
            for ci in range(2*cores):
                height = max(0, min(split, m-ci*split))
                if m >= 2*cores and ci == 2*cores-1:
                    height = m-ci*split
                add(ci, "output", height*n, cast_count=height*_align(n, 64), offset=ci*split*n)
        else:
            boundary = "all_matmul_cores"
            span = m*n//(2*cores)
            for ci in range(2*cores):
                count = span if ci < 2*cores-1 else m*n-ci*span
                add(ci, "output", count, offset=ci*span)
    for row in rows:
        row["features"] = _sum_features(op["features"] for op in row["operations"])
        if strategy != "BASE":
            row["features"]["sync_events"] += 2  # includes idle reduction participants
    return dict(phase_id="reduce_cast", per_core=rows, start_dependency=boundary,
                single_core_num=cores if strategy == "DETERMINISTIC_SPLIT_K" else None,
                source_hazards=[dict(SINGLE_CORE_REDUCER_HAZARD)]
                if strategy == "DETERMINISTIC_SPLIT_K" and cores == 1 else [],
                detail_qualification="source payload exact; control/sync are boundary proxies")


def ops_kernel_features(measurement: Mapping, *, aiv_ub_bytes: int = 192*1024) -> dict:
    """Extract ordered per-core tasks from explicit configuration or measurement.

    Unknown/mismatched tiles fail closed. Existing measurements need no timing
    fields; adding/changing them cannot affect the result. aiv_ub_bytes is the
    A2/C220 compile-time UB domain, not an inferred measurement.
    """
    if aiv_ub_bytes != 192*1024:
        raise ValueError("only reviewed C220 192KiB UB domain supported")
    cfg = _configuration(measurement)
    estimated_tiles = _ceil(cfg["n"], 128)*_ceil(cfg["k"], 128)
    if estimated_tiles > 1_000_000:
        raise ValueError("expanded schedule exceeds one million L0 tiles; use a bounded case")
    matmul = _matmul(cfg)
    phases = [_conversion(cfg, aiv_ub_bytes), matmul, _reduce(cfg, matmul, aiv_ub_bytes)]
    for phase in phases:
        phase["features"] = _sum_features(c["features"] for c in phase["per_core"])
    has_convert = bool(cfg["nd2nz_a"] or cfg["nd2nz_b"])
    dependencies = [["nd2nz", "matmul"]] if has_convert else []
    if cfg["actual_strategy"] != "BASE":
        dependencies.append(["matmul", "reduce_cast"])
    support = ops_shape_support(cfg["m"], cfg["n"], cfg["k"], cfg["actual_strategy"])
    if cfg["actual_strategy"] == "DETERMINISTIC_SPLIT_K" and cfg["block_dim"] == 1:
        support["known_source_hazards"] = [dict(SINGLE_CORE_REDUCER_HAZARD)]
    return dict(schema_version=1, backend="ops_nn", strategy=cfg["actual_strategy"],
                m=cfg["m"], n=cfg["n"], k=cfg["k"], configuration=cfg,
                support=support,
                phases=phases, phase_dependencies=dependencies,
                totals=_sum_features(p["features"] for p in phases),
                nominal_flops=2*cfg["m"]*cfg["n"]*cfg["k"],
                qualification=dict(source_derived=True, counter_validated=False,
                                   latency_qualified=False, time_labels_consumed=False),
                caveats=[
                    "MTE2 payload is logical request volume, not cache misses or physical HBM bytes",
                    "MK33 A reuse within IterateAll and FP16 NN Load3DV2 APIs are SDK-derived; remaining L1/L0 issue timing is a hypothesis",
                    "All direct-wrapper chunks have call_end=false; no per-K-chunk End drain may be inferred",
                    "MK33 SDK asserts stepKa==ceil(chunkK/baseK); short final tails may violate this source assertion despite observed passes; domain unchanged",
                    "Atomic bytes are update payload, not a claim about read-modify-write transactions",
                    "Control/sync counts are source boundary proxies, not decoded scalar instructions",
                    "Detailed API counters currently cover explicit ND2NZ helpers only; zero in other phases means not enumerated",
                    "Zero-work source API calls are counted separately; compiler lowering may remove them",
                    "Deterministic singleCoreNum=1 skips the sole MTE2_V fence before Cast: strong source hazard candidate, not proven causality",
                    "per_output_tile casts may overlap later Matmul tiles; phase edge is not a global barrier",
                    "M=1 may use GEMV lowering; Cube instruction geometry needs independent validation",
                    "Shape support reports feature derivability, never a passed numerical canary",
                ])
