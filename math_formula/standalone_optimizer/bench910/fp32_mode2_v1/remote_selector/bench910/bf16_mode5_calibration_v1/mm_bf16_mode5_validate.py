#!/usr/bin/env python3
"""Paired BF16 source-backed mode-5 validation: clean master vs calibrated online."""
import argparse, json, os, statistics, subprocess
from pathlib import Path
ROOT=Path('/home/xianfei/solvebase_bench_20261025')
OUT=ROOT/'cases/bf16_mode5_validation_v1'
PACKAGES={'master':ROOT/'ops_nn_master_bf16_proto_package',
          'online':ROOT/'ops_nn_bf16_mode5_revised_package'}

def idle(device):
    snap=subprocess.run(['npu-smi','info'],check=True,capture_output=True,text=True).stdout
    if f'No running processes found in NPU {device}' not in snap:
        raise RuntimeError(f'NPU {device} is occupied')

def run(m,device,arm,round_index):
    directory=OUT/f'm{m}_d{device}_r{round_index}_{arm}'
    directory.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ('MMV3_ONLINE_SELECTOR','MMV3_ONLINE_BF16_EXPERIMENT','MMV3_ONLINE_ROW_READ_PROFILE','MMV3_ONLINE_TILE','MMV3_BF16_MODE5_TILE_PROBE'):
        env.pop(key,None)
    env.update(CANN_ROOT='/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0',
               BENCH_API='v3cached',BENCH_CUSTOM_OPP_PATH=str(PACKAGES[arm]),
               MMV3_SKIP_LEGACY_OP_CACHE='1', BENCH_EXPECT_SPLIT_CORE_MODE='5',
               BENCH_EXPECT_ND2NZ_A='0',BENCH_EXPECT_ND2NZ_B='0',
               BENCH_PROFILE_KERNEL_TYPE='MIX_AIC')
    if arm=='online': env.update(MMV3_ONLINE_SELECTOR='1',MMV3_ONLINE_BF16_EXPERIMENT='1')
    idle(device)
    with (directory/'run.log').open('w') as log:
        proc=subprocess.run(['bash',str(ROOT/'bf16_online_vs_master/run_case.sh'),
                             str(m),'384','1536',str(directory),str(device),'5','5'],
                            cwd=ROOT/'bf16_online_vs_master',env=env,stdout=log,stderr=subprocess.STDOUT)
    idle(device)
    if proc.returncode:raise RuntimeError((directory/'run.log').read_text()[-2500:])
    tiling=json.loads((directory/'tiling.json').read_text())
    case=json.loads((directory/'case.json').read_text())
    expected_m=16*((m+319)//320)
    assert tiling['tilingEnableSplitCore']==5 and tiling['singleCoreM']==expected_m
    assert tiling['singleCoreN']==384 and tiling['singleCoreK']==384 and tiling['block_dim']==20
    assert case['max_relative_error']<=.02 and case['kernel_count']==5
    return {'m':m,'device':device,'arm':arm,'round':round_index,
            'median_us':case['kernel_median_us'],'max_relative_error':case['max_relative_error'],
            'tiling':{k:tiling[k] for k in ('tilingEnableSplitCore','singleCoreM','singleCoreN','singleCoreK','baseM','baseN','baseK','stepM','stepN','stepKa','stepKb','block_dim','workspace_bytes')},
            'case_path':str(directory/'case.json')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--m',type=int,required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for r in range(2):
        for arm in (('master','online') if r==0 else ('online','master')):
            row=run(a.m,a.device,arm,r);rows.append(row)
            (OUT/f'm{a.m}_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'m':a.m,'round':r,'arm':arm,'median_us':row['median_us'],'tile':row['tiling']}),flush=True)
    assert rows[0]['tiling']==rows[1]['tiling']==rows[2]['tiling']==rows[3]['tiling']
    med={arm:statistics.median(x['median_us'] for x in rows if x['arm']==arm) for arm in ('master','online')}
    print(json.dumps({'m':a.m,'master_us':med['master'],'online_us':med['online'],'master_over_online':med['master']/med['online']}),flush=True)
if __name__=='__main__':main()
