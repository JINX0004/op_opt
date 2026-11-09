#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path('/home/xianfei/solvebase_bench_20261025')
OUT=ROOT/'cases/fp32_mode5_v1'
CANN=Path('/home/xianfei/miniconda3/envs/triton-project/Ascend/cann-9.0.0')
MASTER=ROOT/'ops_nn_master_bf16_proto_package'
MODE5=ROOT/'ops_nn_fp32_mode5_package'
N128=ROOT/'ops_nn_fp32_mode5_n128_package'

def run(shape,arm,device=4):
    m,n,k=shape
    directory=OUT/f'{m}x{n}x{k}'/arm
    directory.mkdir(parents=True,exist_ok=True)
    result=directory/'result.json'
    if result.exists():
        print(result.read_text(),flush=True);return
    info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    if f'No running processes found in NPU {device}' not in info:raise RuntimeError('device occupied')
    proc=subprocess.run(['bash','-c','source "$1/set_env.sh" >/dev/null 2>&1 && env -0','bash',str(CANN)],capture_output=True,check=True)
    env=dict(item.decode().split('=',1) for item in proc.stdout.split(b'\0') if item)
    for key in ('MMV3_RESEARCH_SPLITK_FAMILY','MMV3_ONLINE_SELECTOR','MMV3_ONLINE_BF16_EXPERIMENT','MMV3_ONLINE_TILE'):
        env.pop(key,None)
    package=MASTER if arm=='master' else N128 if arm=='n128' else MODE5
    env.update(BENCH_CUSTOM_OPP_PATH=str(package),ASCEND_CUSTOM_OPP_PATH=str(package),MMV3_SKIP_LEGACY_OP_CACHE='1',IGNORE_INFER_ERROR='1',ASCEND_GLOBAL_LOG_LEVEL='1')
    env['LD_LIBRARY_PATH']=str(package/'op_api/lib')+':'+env.get('LD_LIBRARY_PATH','')
    if arm!='master':env['MMV3_RESEARCH_SPLITK_FAMILY']='single'
    if arm=='n128':env['MMV3_FP32_MODE5_N128_PROBE']='1'
    cmd=['msprof','--output='+str(directory/'msprof'),'--task-time=on','--ai-core=on','env',str(ROOT/'fp32_bench'),'--api','v3cached','--device',str(device),'--m',str(m),'--n',str(n),'--k',str(k),'--warmup','5','--batches','1','--repetitions','5']
    with (directory/'stdout.log').open('w') as so,(directory/'stderr.log').open('w') as se:
        run=subprocess.run(cmd,env=env,cwd=ROOT,stdout=so,stderr=se,timeout=1200)
    if run.returncode:raise RuntimeError((directory/'stderr.log').read_text()[-2000:])
    event=next(json.loads(line) for line in reversed((directory/'stdout.log').read_text().splitlines()) if line.startswith('{') and 'matmul_v3_event' in line)
    tiling=json.loads(subprocess.run(['python3',str(ROOT/'parse_tiling_log.py'),'--pid',str(event['pid'])],capture_output=True,text=True,env=env,check=True).stdout)
    mode=tiling['tilingEnableSplitCore']
    if mode!=(0 if arm=='master' else 5):raise RuntimeError(f'wrong mode {mode}: {tiling}')
    if arm=='n128' and tiling['singleCoreN']!=128:raise RuntimeError(f'wrong N tile: {tiling}')
    kind='MIX_AIC' if mode==5 else 'KERNEL_AICORE'
    profile=json.loads(subprocess.run(['python3',str(ROOT/'bf16_online_vs_master/parse_profile.py'),str(directory/'msprof'),'--last','5','--kernel-type',kind],capture_output=True,text=True,check=True).stdout)
    obj={'shape':shape,'arm':arm,'device':device,'event':event,'tiling':tiling,'profile':profile}
    result.write_text(json.dumps(obj,indent=2)+'\n')
    print(f'{shape} {arm} mode={mode} tile={tiling["singleCoreM"]}x{tiling["singleCoreN"]}x{tiling["singleCoreK"]} kernel={profile["median_us"]:.3f}us',flush=True)

if __name__=='__main__':
    for spec in sys.argv[1:]:
        m,n,k,arm=spec.split(',')
        run((int(m),int(n),int(k)),arm)
