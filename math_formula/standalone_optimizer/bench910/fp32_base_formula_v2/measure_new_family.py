#!/usr/bin/env python3
"""Pair clean master with the seven opt-in FP32 family changes."""

import importlib.util
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
SOURCE = ROOT / 'fp32_100_source_family_v1/run.py'
spec = importlib.util.spec_from_file_location('fp32_100_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.OUT = ROOT / 'cases/fp32_base_formula_v2/new_family_pairs'

original_environment = runner.environment


def environment(arm, mode, log_level):
    if arm == 'proposed' and mode == 2:
        env = original_environment('research_default', mode, log_level)
        env['MMV3_RESEARCH_SPLITK_FAMILY'] = 'single'
        return env
    return original_environment(arm, mode, log_level)


runner.environment = environment

# Index, M, N, K, new model family. Indices retain the 100-case assignment.
CASES = (
    (10, 2560, 768, 32768, 2),
    (15, 2560, 1024, 32768, 2),
    (16, 2560, 1024, 49152, 3),
    (20, 2048, 1536, 32768, 2),
    (42, 8192, 4096, 32768, 2),
    (60, 1664, 768, 32768, 3),
    (63, 2432, 768, 32768, 3),
)


def measure(case):
    index, m, n, k, mode = case
    shape = [m, n, k]
    device = 4 + index % 4
    directory = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
    result_path = directory / 'record.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    try:
        audits = {arm: runner.audit(index, shape, device, arm, mode)
                  for arm in ('master', 'research_default', 'proposed')}
        mismatch = {field: (audits['master']['tiling'].get(field),
                            audits['research_default']['tiling'].get(field))
                    for field in runner.TILE_FIELDS
                    if audits['master']['tiling'].get(field) !=
                    audits['research_default']['tiling'].get(field)}
        if mismatch:
            raise RuntimeError(f'unforced research differs from master: {mismatch}')
        measurements = []
        for round_index in range(2):
            order = ('master', 'proposed') if round_index == 0 else ('proposed', 'master')
            for arm in order:
                item = runner.profile(index, shape, device, arm, mode,
                                      round_index, audits[arm])
                measurements.append(item)
                print(index, arm, round_index, item['profile']['median_us'], flush=True)
        record = {'index': index, 'shape': shape, 'device': device,
                  'new_model_mode': mode, 'audits': audits,
                  'measurements': measurements}
        runner.write(result_path, record)
        return record
    except Exception as exc:
        runner.write(directory / 'failure.json',
                     {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    for arm, package in runner.PACKAGES.items():
        library = package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        if runner.sha(library) != runner.HASHES[arm]:
            raise RuntimeError(f'{arm} package host library changed')
    by_device = {d: [c for c in CASES if 4 + c[0] % 4 == d] for d in (4, 5, 6, 7)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lambda cases: [measure(c) for c in cases], cases)
                   for cases in by_device.values() if cases]
        results = [record for f in futures for record in f.result()]
    runner.write(runner.OUT / 'records.json', sorted(results, key=lambda r: r['index']))
    print('completed', len(results), 'changed decisions', flush=True)


if __name__ == '__main__':
    main()
