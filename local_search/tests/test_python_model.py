#!/usr/bin/env python3
"""Check the C++ embedding against frozen predictions and Python API behavior."""
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'local_search/build/bin/local_search'
sys.path.insert(0,str(ROOT/'cost_model'))
from matmul_cost_model import make_config

with gzip.open(ROOT/'cost_model/tests/data/independent_validation.jsonl.gz','rt') as stream:
    records=[json.loads(line) for line in stream]
example=dict(implementation='ops_nn_deterministic_split_k',shape=[16,16,102400],dtype='fp16',cores=24,tiling=dict(base_m=128,base_n=128,base_k=128,step_k=2))
records.append(dict(config=example,expected_us=91.98508811412263))
# Also accept a complete exported static plan, not just compact configurations.
records.append(dict(config=make_config(**example),expected_us=91.98508811412263))
run=subprocess.run([str(BINARY),'--jsonl'],input=''.join(json.dumps(r['config'])+'\n' for r in records),text=True,capture_output=True,timeout=600,check=True)
outputs=[json.loads(line) for line in run.stdout.splitlines()]
assert len(outputs)==len(records)
for record,result in zip(records,outputs):
    assert math.isclose(result['predicted_us'],record['expected_us'],rel_tol=1e-12,abs_tol=1e-10),(record,result)
    assert result['hardware_execution_performed'] is False
assert outputs[-1]['active_compute_cores']==24

# The typed C++ API/default CLI must build the same configuration.
typed=json.loads(subprocess.check_output([str(BINARY),'--predict'],text=True))
assert typed['predicted_us']==outputs[-1]['predicted_us']
semantic=json.loads(subprocess.check_output([str(BINARY),'--predict','--semantic'],text=True))
assert math.isclose(semantic['predicted_us'],93.20267861374995,rel_tol=1e-12)
for update in [dict(dtype='fp32'),dict(cores=1),dict(tiling=dict(base_m=64,base_n=64,base_k=128,step_k=2))]:
    invalid=example|update
    bad=subprocess.run([str(BINARY),'--jsonl'],input=json.dumps(invalid)+'\n',text=True,capture_output=True)
    assert bad.returncode!=0
    assert 'Prediction error:' in bad.stderr
# The GA uses Python's reviewed candidates, including aliases and CATLASS.
for implementation in ['ops_nn_deterministic_split_k','base','scsk','catlass']:
    search=subprocess.run([str(BINARY),'--implementation',implementation],capture_output=True,text=True,check=True,timeout=30)
    assert 'Estimated runtime:' in search.stdout
print(f'Passed {len(records)} calibrated parity cases, semantic/typed API, invalid inputs, and all four searches.')
