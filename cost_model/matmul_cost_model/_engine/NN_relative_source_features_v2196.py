"""Offset-invariant A source row-phase proxies; no bank/page causality claim."""
from math import gcd,log1p,sin,cos,pi
from matmul_cost_model._engine.static_core_path_features_v1773 import NAMES as OLD_NAMES,features as old_features
EXTRA_NAMES=tuple(f'relative_row_phase{w}_{n}' for w in (2048,4096) for n in ('period_log','same_phase_pair_fraction','stride_sin','stride_cos','same_phase_by_core','same_phase_by_burst'))
NAMES=OLD_NAMES+EXTRA_NAMES

def extras(row):
 m,n,k=row['shape'];c=row['cores']
 if row['implementation']=='catlass':bm,_,burst=row['l1_tile']
 else:
  t=row['plan']['tcube'];bm=t['baseM'];burst=t['baseK']*t['stepKa']
 rows=min(m,bm);result=[]
 for w in (2048,4096):
  period=w//gcd(2*k,w);quot,rem=divmod(rows,period)
  equal=(rem*(quot+1)*quot+(period-rem)*quot*(quot-1))/max(1,rows*(rows-1))
  angle=2*pi*((2*k)%w)/w
  result.extend([log1p(period),equal,sin(angle),cos(angle),equal*log1p(c),equal*log1p(min(k,burst)/128)])
 return result

def features(row,pipeline_us,model,path):return old_features(row,pipeline_us,model,path=path)+extras(row)
