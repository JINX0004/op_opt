"""Measure CPU scoring overhead; this does NOT measure or execute an NPU kernel."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from matmul_cost_model import MatmulCostModel

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'examples/fp16_deterministic.json')
    p.add_argument('--repeat',type=int,default=10);a=p.parse_args()
    if a.repeat<1:p.error('--repeat must be positive')
    row=json.loads(a.input.read_text(encoding='utf-8-sig'))
    if isinstance(row,list):p.error('benchmark input must contain one configuration')
    t=time.perf_counter();model=MatmulCostModel();load_ms=1000*(time.perf_counter()-t)
    value=model.predict(row);t=time.perf_counter()
    for _ in range(a.repeat):model.predict(row)
    print(json.dumps(dict(model_load_CPU_ms=load_ms,mean_scoring_CPU_ms=1000*(time.perf_counter()-t)/a.repeat,
                         predicted_kernel_us=value['predicted_us'],repeats=a.repeat,hardware_execution_performed=False),indent=2))

if __name__=='__main__':main()
