"""Static work/layout basis for a bounded service-model residual.

No identifier, label, PMU or measured feature enters these descriptors.
No per-shape or per-candidate fitted constant. CAT uses its actual L1/L0
geometry; native paths use their reviewed TCube configuration.
"""
from matmul_cost_model._engine.static_work_features_v1362 import features as work_features,NAMES as WORK_NAMES
EXTRA=('SC_B_cache_disable','global_K_tail_fraction_128','A_row_bytes_phase32','B_row_bytes_phase32')
PAIRS=(('stepA','active'),('stepB','active'),('baseM','max_job_n'),('baseN','max_job_m'),
       ('stepA','max_source_A'),('stepB','max_source_B'),('max_jobs','padding_ratio'),
       ('launched_idle','stepA'),('launched_idle','stepB'),('max_atomic_fraction','active'),
       ('max_job_K','B_row_bytes_phase32'),('stepA','B_row_bytes_phase32'),
       ('stepB','B_row_bytes_phase32'),('SC_B_cache_disable','K'),
       ('SC_B_cache_disable','active'),('global_K_tail_fraction_128','max_mmad'))
BASE_NAMES=tuple(WORK_NAMES)+EXTRA
NAMES=BASE_NAMES+tuple(a+'*'+b for a,b in PAIRS)

def features(row):
    m,n,k=row['shape'];kind=row['implementation'];t=row.get('plan',{}).get('tcube',{})
    disabled=bool(kind=='scsk' and t and m==t['stepM']*t['baseM'])
    x=work_features(row)+[float(disabled),(k%128)/128,((2*k)%32)/32,((2*n)%32)/32]
    return x+[x[BASE_NAMES.index(a)]*x[BASE_NAMES.index(b)] for a,b in PAIRS]
