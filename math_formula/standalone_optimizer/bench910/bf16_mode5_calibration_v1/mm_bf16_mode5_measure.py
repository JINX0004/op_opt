#!/usr/bin/env python3
"""Measure BF16 mode-5 source/small tiles with AIC pipeline counters."""
import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path('/home/xianfei/solvebase_bench_20261025')
PACKAGE = ROOT / 'ops_nn_bf16_mode5_calib_package'
OUT = ROOT / 'cases/bf16_mode5_calib_v1'
FIELDS = ('Task Duration(us)', 'aicore_time(us)', 'aic_mac_time(us)',
          'aic_mte1_time(us)', 'aic_mte2_time(us)',
          'aic_fixpipe_time(us)', 'aic_scalar_time(us)')

def idle(device):
    snap = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
    if f'No running processes found in NPU {device}' not in snap:
        raise RuntimeError(f'NPU {device} is occupied')

def one(m, device, arm, round_index):
    directory = OUT / f'm{m}_d{device}_r{round_index}_{arm}'
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('MMV3_ONLINE_SELECTOR','MMV3_ONLINE_BF16_EXPERIMENT','MMV3_ONLINE_ROW_READ_PROFILE','MMV3_ONLINE_TILE'):
        env.pop(key, None)
    env.update(CANN_ROOT='/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
               BENCH_API='v3cached', BENCH_CUSTOM_OPP_PATH=str(PACKAGE),
               MMV3_SKIP_LEGACY_OP_CACHE='1', BENCH_EXPECT_SPLIT_CORE_MODE='5',
               BENCH_EXPECT_ND2NZ_A='0', BENCH_EXPECT_ND2NZ_B='0',
               BENCH_PROFILE_KERNEL_TYPE='MIX_AIC',
               MMV3_BF16_MODE5_TILE_PROBE='s' if arm=='small' else 'l')
    idle(device)
    with (directory / 'run.log').open('w') as log:
        result = subprocess.run(['bash', str(ROOT / 'bf16_online_vs_master/run_case.sh'),
                                 str(m), '384', '1536', str(directory), str(device), '10', '8'],
                                cwd=ROOT / 'bf16_online_vs_master', env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    idle(device)
    if result.returncode:
        raise RuntimeError(f'{arm} M={m} failed: {(directory / "run.log").read_text()[-2000:]}')
    tiling = json.loads((directory / 'tiling.json').read_text())
    expected_m = 128 if arm=='small' else 16*((m+319)//320)
    expected_n = 128 if arm=='small' else 384
    if (tiling['tilingEnableSplitCore'] != 5 or tiling['singleCoreM'] != expected_m or
        tiling['singleCoreN'] != expected_n or tiling['singleCoreK'] != 384 or
        tiling['block_dim'] != 20):
        raise RuntimeError(f'unexpected tiling: {tiling}')
    case = json.loads((directory / 'case.json').read_text())
    if case['max_relative_error'] > .02 or case['kernel_count'] != 8:
        raise RuntimeError(f'correctness or count failure: {case}')
    summaries = list((directory / 'profile').glob('PROF_*/mindstudio_profiler_output/op_summary_*.csv'))
    if len(summaries)!=1:
        raise RuntimeError(f'expected one op summary: {summaries}')
    with summaries[0].open(newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['OP Type']=='MatMulV3'][-8:]
    if len(rows)!=8:
        raise RuntimeError(f'expected eight MatMulV3 rows: {len(rows)}')
    counters={key:statistics.median(float(row[key]) for row in rows) for key in FIELDS}
    return {'m':m,'device':device,'arm':arm,'round':round_index,
            'tiling':{k:tiling[k] for k in ('singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','stepM','stepN','stepKa','stepKb','block_dim','tilingEnableSplitCore')},
            'kernel_median_us':case['kernel_median_us'],
            'max_relative_error':case['max_relative_error'],
            'counters_us':counters,'summary_csv':str(summaries[0])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--m',type=int,required=True);p.add_argument('--device',type=int,required=True)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in range(2):
        for arm in (('large','small') if r==0 else ('small','large')):
            row=one(a.m,a.device,arm,r);rows.append(row)
            (OUT / f'm{a.m}_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'m':a.m,'device':a.device,'round':r,'arm':arm,
                              'us':row['kernel_median_us'], 'counters':row['counters_us']}),flush=True)
if __name__=='__main__':main()
