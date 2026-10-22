"""Ordered, source-derived work for the frozen CATLASS example09 wrapper.

This is a work/geometry extractor, not a latency fit. It never reads measured
time, throughput, accuracy, or profiler counters. ``intrinsic_calls`` count
AscendC API invocations in the reviewed source, not lowered machine instructions.
Byte counts are logical requests or explicitly named local padded payloads;
they are not HBM traffic. The ordered tasks retain the two ping-pong stages
across tasks so a separate resource/dependency simulator can model overlap.

Reviewed paths, relative to the workspace:
* experiments/attention_matmul/catlass/catlass_bench.cpp (Run and tile choices)
* catlass/include/catlass/gemm/kernel/splitk_matmul.hpp:25,197,285
* catlass/include/catlass/gemm/block/block_swizzle.hpp:197
* catlass/include/catlass/gemm/block/block_mmad_pingpong.hpp:158,203
* catlass/include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp:1058
* catlass/include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp:225
* catlass/include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp:390
* catlass/include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp:134
* catlass/include/catlass/gemm/tile/tile_mmad.hpp:43
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from .external_gemm_work import catlass_split_factor, ceildiv


_TILES = {
    "default": ([128, 256, 256], [128, 256, 64]),
    "deepk": ([16, 16, 4096], [16, 16, 1024]),
}
_CONTRACT = dict(backend="catlass", algorithm="example09_splitk",
                 dtype="float16", output_dtype="float16", layout_a="row_major",
                 layout_b="row_major", trans_a=False, trans_b=False)
_COMPUTE_LENGTH = 192 * 1024 // 4


def _round16(value: int) -> int:
    return ceildiv(value, 16) * 16


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or not 0 < value <= 0x7fffffff:
        raise ValueError(f"{name} must be a positive signed 32-bit integer")
    return value


def manual_catlass_configuration(m: int, n: int, k: int, tile: str,
                                 physical_aic: int = 24) -> dict:
    """Freeze source host-policy metadata without invoking or measuring a kernel.

    AtlasA2's mixed AIC/AIV launch has two vector subblocks per AIC. This is an
    explicit architecture assumption, not a measured count of busy vector cores.
    """
    for name, value in (("m", m), ("n", n), ("k", k), ("physical_aic", physical_aic)):
        _positive_int(name, value)
    if tile not in _TILES:
        raise ValueError("only reviewed default/deepk tile configurations are supported")
    l1, l0 = _TILES[tile]
    split = catlass_split_factor(m, n, k, l1, physical_aic)
    return dict(_CONTRACT, m=m, n=n, k=k, tile=tile, physical_aic=physical_aic,
                aiv_subblocks_per_aic=2, splitk_factor=split,
                l1_tile=list(l1), l0_tile=list(l0), workspace_bytes=4*m*n*split)


def _configuration(measurement: Mapping) -> dict:
    for name, expected in _CONTRACT.items():
        if measurement.get(name) != expected:
            raise ValueError(f"unreviewed CATLASS contract: {name}")
    if type(measurement["trans_a"]) is not bool or type(measurement["trans_b"]) is not bool:
        raise ValueError("transpose flags must be booleans")
    tile = next((name for name, (l1, l0) in _TILES.items()
                 if measurement.get("l1_tile") == l1 and measurement.get("l0_tile") == l0), None)
    if tile is None:
        raise ValueError("unreviewed CATLASS L1/L0 tile combination")
    if "tile" in measurement and measurement["tile"] != tile:
        raise ValueError("tile label conflicts with executed L1/L0 geometry")
    cfg = manual_catlass_configuration(*(measurement[name] for name in ("m", "n", "k")),
                                       tile, measurement["physical_aic"])
    for name in ("splitk_factor", "workspace_bytes", "aiv_subblocks_per_aic"):
        if name in measurement and (type(measurement[name]) is not int or measurement[name] != cfg[name]):
            raise ValueError(f"{name} conflicts with the frozen host/architecture policy")
    if "splitk_factor" not in measurement:
        raise ValueError("executed splitk_factor is required; use manual_catlass_configuration for prediction")
    return cfg


def _mte2(rows: int, columns: int, gm_stride: int, l1_rows: int,
          gm_offset: int) -> dict:
    """Half RowMajor -> zN; row-stride overflow causes one ND2NZ per row."""
    rowwise = gm_stride >= 65536
    calls = rows if rowwise else 1
    return dict(api="DataCopy_Nd2Nz", intrinsic_calls=calls, rows=rows, columns=columns,
                row_bytes=2*columns, source_row_stride_bytes=2*gm_stride,
                source_row_gap_bytes=2*(gm_stride-columns), gm_offset_bytes=2*gm_offset,
                rowwise_stride_fallback=rowwise, logical_read_bytes=2*rows*columns,
                local_padded_payload_bytes=2*_round16(rows)*_round16(columns),
                nd_num=1, n_value=1 if rowwise else rows, d_value=columns,
                src_d_value=0 if rowwise else gm_stride,
                dst_nz_c0_stride=l1_rows, dst_nz_n_stride=0 if rowwise else 1,
                call_source_step_bytes=2*gm_stride if rowwise else 0,
                call_destination_step_bytes=32 if rowwise else 0)


def _mte1(m: int, n: int, k: int, l1: list[int], local_k: int) -> tuple[dict, dict]:
    """LoadData2D repeat/stride units are 512-byte FP16 fractals."""
    kp = _round16(k)
    common = dict(api="LoadData2D", repeat_bytes=512, dst_gap_fractals=0)
    a = dict(common, intrinsic_calls=m//16, repeats_per_call=kp//16,
             transpose=False, local_read_bytes=2*m*kp,
             src_stride_fractals=l1[0]//16, call_source_step_bytes=512,
             call_destination_step_bytes=2*kp*16,
             l1_offset_bytes=2*(local_k//16)*l1[0]*16)
    b = dict(common, intrinsic_calls=kp//16, repeats_per_call=n//16,
             transpose=True, local_read_bytes=2*kp*n,
             src_stride_fractals=l1[2]//16, call_source_step_bytes=512,
             call_destination_step_bytes=2*n*16, l1_offset_bytes=2*local_k*16)
    return a, b


def _task(cfg: dict, task_id: int, m_index: int, n_index: int, slice_index: int,
          k_start: int, k_size: int, l1_sequence: int, l0_sequence: int) -> dict:
    m, n, k = (cfg[key] for key in ("m", "n", "k"))
    l1, l0 = cfg["l1_tile"], cfg["l0_tile"]
    mo, no = m_index*l1[0], n_index*l1[1]
    tm, tn = min(m-mo, l1[0]), min(n-no, l1[1])
    mr, nr = _round16(tm), _round16(tn)
    steps, totals = [], Counter(task_count=1)
    l0_index = 0
    for step_index, ko in enumerate(range(0, k_size, l1[2])):
        tk = min(k_size-ko, l1[2])
        global_k = k_start+ko
        a = _mte2(tm, tk, k, l1[0], mo*k+global_k)
        b = _mte2(tk, tn, n, l1[2], global_k*n+no)
        chunks = []
        for part_index, local_k in enumerate(range(0, tk, l0[2])):
            pk = min(tk-local_k, l0[2])
            la, lb = _mte1(mr, nr, pk, l1, local_k)
            blocks = mr//16 * (nr//16) * ceildiv(pk, 16)
            barrier = int(mr//16 * (nr//16) < 10)
            last = ko+local_k+pk == k_size
            cube = dict(intrinsic_calls=1, m=mr, n=nr, k=pk, blocks_16=blocks,
                        init_c=l0_index == 0, unit_flag=3 if last else 2,
                        pipe_m_barriers=barrier)
            chunks.append(dict(index=l0_index, part_index=part_index, m=mr, n=nr, k=pk,
                               offset_k=global_k+local_k,
                               l0_stage=(l0_sequence+l0_index)%2,
                               a_mte1=la, b_mte1=lb, cube=cube))
            totals.update(l1_to_l0_a_calls=la["intrinsic_calls"],
                          l1_to_l0_b_calls=lb["intrinsic_calls"],
                          l1_to_l0_a_bytes=la["local_read_bytes"],
                          l1_to_l0_b_bytes=lb["local_read_bytes"],
                          cube_calls=1, cube_blocks=blocks, pipe_m_barriers=barrier)
            l0_index += 1
        steps.append(dict(index=step_index, m=tm, n=tn, k=tk, offset_k=global_k,
                          l1_stage=(l1_sequence+step_index)%2, a_mte2=a, b_mte2=b,
                          l0_chunks=chunks, l0_call_count=len(chunks),
                          prefetch_next_before_l0=ko+tk < k_size))
        totals.update(l1_steps=1, gm_to_l1_a_calls=a["intrinsic_calls"],
                      gm_to_l1_b_calls=b["intrinsic_calls"],
                      logical_a_read_bytes=a["logical_read_bytes"],
                      logical_b_read_bytes=b["logical_read_bytes"])
    fix = dict(api="Fixpipe", intrinsic_calls=1, m=tm, n=tn, source_dtype="float32",
               destination_dtype="float32", quant_mode="NoQuant", unit_flag=3,
               source_stride_elements=mr, destination_stride_elements=n,
               gm_offset_bytes=4*(slice_index*m*n+mo*n+no),
               logical_write_bytes=4*tm*tn, local_padded_payload_bytes=4*mr*nr)
    event_counts = dict(MTE1_MTE2=2*len(steps), MTE2_MTE1=2*len(steps),
                        M_MTE1=2*l0_index, MTE1_M=l0_index)
    hard_events = sum(event_counts.values())
    control = dict(hard_event_sets=hard_events, hard_event_waits=hard_events,
                   event_sets_by_type=event_counts, event_waits_by_type=dict(event_counts),
                   l1_loop_iterations=len(steps), l0_loop_iterations=l0_index,
                   prefetch_branch_evaluations=len(steps), prefetch_taken=max(0, len(steps)-1),
                   scheduler_tasks=1, gm_row_fallback_loop_iterations=sum(
                       x[key]["rows"] if x[key]["rowwise_stride_fallback"] else 0
                       for x in steps for key in ("a_mte2", "b_mte2")),
                   l0a_load_loop_iterations=totals["l1_to_l0_a_calls"],
                   l0b_load_loop_iterations=totals["l1_to_l0_b_calls"])
    totals.update(output_flushes=1, fixpipe_write_bytes=4*tm*tn,
                  hard_event_sets=hard_events, hard_event_waits=hard_events)
    return dict(task_id=task_id, slice_index=slice_index, m=tm, n=tn, k=k_size,
                rounded_m=mr, rounded_n=nr, offset_m=mo, offset_n=no, offset_k=k_start,
                l1_step_count=len(steps), l0_call_count=l0_index, l1_steps=steps,
                fixpipe=fix, control=control, totals=dict(totals))


def _reduction(cfg: dict) -> tuple[list[dict], dict]:
    elements = cfg["m"]*cfg["n"]
    split = cfg["splitk_factor"]
    aivs = cfg["physical_aic"]*cfg["aiv_subblocks_per_aic"]
    # Preserve the source's floor(elementCount/aivNum), then round to 64.
    per_aiv = max(64, ceildiv(elements//aivs, 64)*64)
    per_aiv = min(per_aiv, _COMPUTE_LENGTH//split//64*64)
    loops = ceildiv(elements, per_aiv)
    separate_copies = elements >= 4294977295//4  # Literal in reviewed source.
    cores = []
    total = Counter()
    for core_id in range(aivs):
        tasks, ct = [], Counter()
        for task_id in range(core_id, loops, aivs):
            offset, length = task_id*per_aiv, min(per_aiv, elements-task_id*per_aiv)
            copies = split if separate_copies else 1
            mte2 = dict(api="DataCopyPad", intrinsic_calls=copies,
                        block_count=1 if separate_copies else split, block_bytes=4*length,
                        src_stride_bytes=0 if separate_copies else 4*(elements-length),
                        dst_stride_blocks=0 if separate_copies else (_round16(length)-length)*4//32,
                        gm_offset_bytes=4*offset, slice_source_stride_bytes=4*elements,
                        ub_slice_stride_bytes=4*_round16(length), logical_read_bytes=4*length*split)
            vector = dict(add_calls=split-1, add_elements=(split-1)*length,
                          add_vector_blocks_64=(split-1)*ceildiv(length, 64),
                          pipe_v_barriers=split-1, cast_calls=1, cast_elements=length,
                          cast_dtype_from="float32", cast_dtype_to="float16", cast_mode="CAST_NONE")
            mte3 = dict(api="DataCopyPad", intrinsic_calls=1, block_count=1,
                        block_bytes=2*length, gm_offset_bytes=2*offset, logical_write_bytes=2*length)
            tasks.append(dict(task_id=task_id, offset_elements=offset, elements=length,
                              ub_stage=0, mte2=mte2, vector=vector, mte3=mte3,
                              control=dict(hard_event_sets=3, hard_event_waits=3)))
            ct.update(task_count=1, mte2_calls=copies, reduction_read_bytes=4*length*split,
                      vector_add_calls=split-1, reduction_add_elements=(split-1)*length,
                      pipe_v_barriers=split-1, cast_calls=1, output_cast_elements=length,
                      mte3_calls=1, output_write_bytes=2*length)
        control = dict(hard_event_sets=1+3*len(tasks), hard_event_waits=1+3*len(tasks),
                       cross_core_waits=2, cross_core_sets=1, pipe_all_barriers=1,
                       reduction_loop_iterations=len(tasks), reduction_slice_loop_iterations=(split-1)*len(tasks))
        ct.update({name: control[name] for name in ("hard_event_sets", "hard_event_waits",
                                                   "cross_core_waits", "cross_core_sets", "pipe_all_barriers")})
        total.update(ct)
        cores.append(dict(core_id=core_id, tasks=tasks, control=control, totals=dict(ct)))
    reduction = dict(total, task_elements=per_aiv, task_count=loops, launched_aiv=aivs,
                     active_aiv=sum(bool(core["tasks"]) for core in cores),
                     split_k_slices=split, ub_stage_count=1, ub_reserved_bytes=192*1024,
                     ub_input_output_alias=True, separate_slice_copy_path=separate_copies)
    return cores, reduction


def catlass_kernel_features(measurement: Mapping) -> dict:
    """Extract ordered per-core AIC and AIV phases from metadata alone.

    Existing measurement JSON is accepted but only whitelisted launch/config
    fields are consumed. Unsupported kernels/layouts and inconsistent executed
    split metadata fail closed. All launched cores, including idle ones, are
    retained because they participate in initialization and synchronization.
    """
    cfg = _configuration(measurement)
    m, n, k, aics = (cfg[key] for key in ("m", "n", "k", "physical_aic"))
    l1, l0, split = cfg["l1_tile"], cfg["l0_tile"], cfg["splitk_factor"]
    gm, gn, gk = (ceildiv(dim, tile) for dim, tile in zip((m, n, k), l1))
    if split > gk or gm*gn*split > 0xffffffff:
        raise ValueError("split/task geometry exceeds the reviewed source index domain")
    cores = [dict(core_id=ci, tasks=[], totals=Counter(), _l1_sequence=0, _l0_sequence=0)
             for ci in range(aics)]
    for si in range(split):
        start = (si*(gk//split)+min(si, gk % split))*l1[2]
        ks = min((gk//split+int(si < gk % split))*l1[2], k-start)
        for inner in range(gm*gn):
            block, rem = divmod(inner, 3*gn)
            height = min(3, gm-3*block)
            mi, ni = 3*block+rem % height, rem//height
            if block % 2:
                ni = gn-1-ni
            task_id = si*gm*gn+inner
            core = cores[task_id % aics]
            task = _task(cfg, task_id, mi, ni, si, start, ks,
                         core["_l1_sequence"], core["_l0_sequence"])
            core["tasks"].append(task)
            core["totals"].update(task["totals"])
            core["_l1_sequence"] += task["l1_step_count"]
            core["_l0_sequence"] += task["l0_call_count"]
    total = Counter()
    for core in cores:
        del core["_l1_sequence"], core["_l0_sequence"]
        # BlockMmad is constructed/destructed once, even on idle AICs.
        core["control"] = dict(constructor_event_sets=9, destructor_event_waits=9,
                               cross_core_sets=1, pipe_all_barriers=1,
                               scheduler_loop_iterations=len(core["tasks"]))
        core["totals"].update(hard_event_sets=9, hard_event_waits=9,
                              cross_core_sets=1, pipe_all_barriers=1)
        total.update(core["totals"])
        core["totals"] = dict(core["totals"])
    aiv_cores, reduction = _reduction(cfg)
    return dict(schema_version=1, source_model="catlass_atlasa2_example09_full_work_v1",
                source_derived=True, counter_validated=False, configuration=cfg,
                nominal_flops=2*m*n*k, launched_aic=aics,
                active_aic=sum(bool(c["tasks"]) for c in cores), launched_aiv=2*aics,
                grid_m=gm, grid_n=gn, grid_k=gk, splitk_factor=split,
                task_count=gm*gn*split, aic_cores=cores, aiv_cores=aiv_cores,
                aic_totals=dict(total), reduction=reduction,
                buffers=dict(l1_stages=2, l0a_stages=2, l0b_stages=2, l0c_stages=1,
                             l1_reserved_bytes=2*(l1[0]+l1[1])*l1[2]*2,
                             l0a_stage_bytes=32768, l0b_stage_bytes=32768,
                             l0c_tile_capacity_bytes=l0[0]*l0[1]*4),
                sync=dict(aic_finish_flag_mode=2, aic_finish_flag_pipe="FIX",
                          aiv_waits_for_paired_aic=True, aiv_interblock_barrier_mode=0,
                          aiv_interblock_barrier_pipe="MTE3",
                          reduction_after_all_aic_fixpipe=True,
                          unit_flag_enabled=True, explicit_m_fix_events_per_task=0),
                ordering=dict(aic="task_id = core_id + iteration * launched_aic",
                              within_task="load first L1; prefetch next L1 before current L0 A/B/MMAD; Fixpipe",
                              aiv="wait AIC; all-AIV barrier; serial UB stages: MTE2, Add/Cast, MTE3",
                              pingpong_stages_persist_across_tasks=True),
                caveats=["Source API counts are not lowered scalar instruction counts or hardware PMU counters",
                         "Logical GM requests exclude L2 hits, transaction rounding and memory-controller contention",
                         "Local padded payload bytes describe LoadData geometry, not additional GM reads",
                         "AtlasA2 mixed launch assumes two AIV subblocks per AIC; no runtime topology measurement",
                         "Unit-flag and event dependencies require scheduling; summing phase service times is not latency",
                         "Includes device kernel/reduction only; host launch, allocation, validation and event overhead excluded"])
