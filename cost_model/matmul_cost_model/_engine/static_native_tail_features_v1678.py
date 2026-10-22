"""Exact last-iteration and buffer-cycle geometry, derived only from plan."""
from math import log2,ceil
from matmul_cost_model._engine.static_halfphase_features_v1586 import features as prior_features,NAMES as PRIOR_NAMES,correction
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan
EXTRA_NAMES=('last_MMAD_K_fraction','last_A_L1_K_fraction','last_B_L1_K_fraction',
    'last_A_buffer_cycle_phase','last_B_buffer_cycle_phase','last_L0_cycle_phase',
    'small_MMAD_requires_PIPE_M_barrier','per_core_K_imbalance','A_prefill_work_fraction','B_prefill_work_fraction')
NAMES=PRIOR_NAMES+EXTRA_NAMES

def extra_features(row):
    if row['implementation']=='catlass':return [0.]*len(EXTRA_NAMES)
    p=compile_plan(row);t=row['plan']['tcube'];gg=[j['geometry'] for jobs in p['work'] for j in jobs]
    slotsA=t['depthA1']/t['stepKa'];slotsB=t['depthB1']/t['stepKb'];l0=min(t['dbL0A'],t['dbL0B'])
    def last(k,span):return ((k-1)%span+1)/span
    raw=[max(last(g['k'],g['bk']) for g in gg),
         max(last(g['k'],g['bk']*g['sa']) for g in gg),max(last(g['k'],g['bk']*g['sb']) for g in gg),
         max(last(ceil(g['k']/(g['bk']*g['sa'])),int(slotsA)) for g in gg),
         max(last(ceil(g['k']/(g['bk']*g['sb'])),int(slotsB)) for g in gg),
         max(last(ceil(g['k']/g['bk']),l0) for g in gg),
         max(int(ceil(g['m']/16)*ceil(g['n']/16)<10) for g in gg),
         max(g['k'] for g in gg)/min(g['k'] for g in gg)-1,
         max(min(1,g['bk']*g['sa']*slotsA/g['k']) for g in gg),
         max(min(1,g['bk']*g['sb']*slotsB/g['k']) for g in gg)]
    return [log2(1+v) for v in raw]

def features(row,pipeline_us,model):return prior_features(row,pipeline_us,model)+extra_features(row)
