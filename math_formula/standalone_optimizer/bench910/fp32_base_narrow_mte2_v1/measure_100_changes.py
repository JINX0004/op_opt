#!/usr/bin/env python3
"""Fresh clean-master/forced-family pairs for the three changed FP32 cases."""

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
runner.OUT = ROOT / 'cases/fp32_base_narrow_mte2_selector_changes_v1'

CASES = (
    (97, (4096, 64, 27328), 3, 4),
    (98, (4096, 64, 27392), 3, 5),
    (99, (4096, 64, 27456), 3, 6),
)


def measure(case):
    index, shape, mode, device = case
    directory = runner.OUT / f'device{device}' / f'{index:03d}_{runner.label(shape)}'
    path = directory / 'record.json'
    if path.exists():
        return json.loads(path.read_text())
    try:
        audits = {arm: runner.audit(index, list(shape), device, arm, mode)
                  for arm in ('master', 'research_default', 'proposed')}
        mismatch = {field: (audits['master']['tiling'].get(field),
                            audits['research_default']['tiling'].get(field))
                    for field in runner.TILE_FIELDS
                    if audits['master']['tiling'].get(field) !=
                       audits['research_default']['tiling'].get(field)}
        if mismatch:
            raise RuntimeError(f'unforced research differs from master: {mismatch}')
        measurements = []
        for round_index, order in enumerate((('master', 'proposed'),
                                             ('proposed', 'master'))):
            for arm in order:
                item = runner.profile(index, list(shape), device, arm, mode,
                                      round_index, audits[arm])
                measurements.append(item)
                print(index, arm, round_index, item['profile']['median_us'], flush=True)
        record = {'index': index, 'shape': shape, 'device': device,
                  'new_model_mode': mode, 'audits': audits,
                  'measurements': measurements}
        runner.write(path, record)
        return record
    except Exception as exc:
        runner.write(directory / 'failure.json',
                     {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    for arm, package in runner.PACKAGES.items():
        library = package / 'op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so'
        if runner.sha(library) != runner.HASHES[arm]:
            raise RuntimeError(f'{arm} package host library changed')
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(measure, CASES))
    runner.write(runner.OUT / 'records.json', sorted(records, key=lambda r: r['index']))
    print('completed', len(records), 'paired cases', flush=True)
