"""Whitelisted static work features for a smooth analytic residual hypothesis.

No case ID, target timing, PMU, measured baseline or shape-time lookup.
"""
from math import log2,ceil
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan

NAMES=('M','N','K','baseM','baseN','baseK','stepA','stepB','launched','active',
       'max_jobs','max_mmad','max_source_A','max_source_B','max_fix','max_outputs',
       'max_source_MB','total_source_MB','max_padded_GF','padding_ratio',
       'min_job_K','max_job_K','mean_job_K','max_atomic_fraction','A_NZ','B_NZ',
       'aligned_SC_cast','A_slot_KB','B_slot_KB','L0_slots','L0C_slots',
       'max_source_A_rows','max_source_B_rows','A_source_pitch','B_source_pitch',
       'max_job_m','max_job_n','min_job_m','min_job_n','launched_idle')

def features(row):
    # Construct from the explicit static whitelist only; ignore extra fields.
    static={k:row[k] for k in ('implementation','shape','cores')}
    for k in ('plan','tile','l1_tile','l0_tile','execution_profile','transport'):
        if k in row:static[k]=row[k]
    p=compile_plan(static);m,n,k=static['shape'];kind=static['implementation']
    alljobs=[j['geometry'] for js in p['work'] for j in js];g=alljobs[0]
    bm,bn,bk,sa,sb=(g[x] for x in ('bm','bn','bk','sa','sb'))
    bycore=[]
    for js in p['work']:
        gg=[j['geometry'] for j in js]
        if not gg:continue
        mm=sum(ceil(x['k']/x['bk']) for x in gg)
        aa=sum(ceil(x['k']/(x['bk']*x['sa'])) for x in gg)
        bb=sum(ceil(x['k']/(x['bk']*x['sb'])) for x in gg)
        outputs=len({(x['mo'],x['no']) for x in gg})
        padded=sum(ceil(x['m']/16)*16*ceil(x['n']/16)*16*ceil(x['k']/16)*16 for x in gg)
        bycore.append((len(gg),mm,aa,bb,len(gg),outputs,sum(j['source_mb'] for j in js),padded/1e9,
                       sum(x['atomic'] for x in gg)/len(gg)))
    maximum=lambda j:max(x[j] for x in bycore)
    totalpad=sum(ceil(x['m']/16)*16*ceil(x['n']/16)*16*ceil(x['k']/16)*16 for x in alljobs)
    nzA=bool(g['nza']);nzB=bool(g['nzb']);t=static.get('plan',{}).get('tcube',{})
    raw=[m,n,k,bm,bn,bk,sa,sb,p['cores'],p['active_cores'],
         maximum(0),maximum(1),maximum(2),maximum(3),maximum(4),maximum(5),
         maximum(6),p['source_mb'],maximum(7),totalpad/(m*n*k),
         min(x['k'] for x in alljobs),max(x['k'] for x in alljobs),sum(x['k'] for x in alljobs)/len(alljobs),
         maximum(8),int(nzA),int(nzB),int(kind=='scsk' and n%128==0 and not p['has_conversion']),
         2*bm*bk*sa/1024,2*bn*bk*sb/1024,min(t.get('dbL0A',2),t.get('dbL0B',2)),t.get('dbL0C',1),
         min(max(x['k'] for x in alljobs),bk*sa),min(max(x['k'] for x in alljobs),bk*sb),
         2*(bk*sa if nzA else k),2*(bk*sb if nzB else n),
         max(x['m'] for x in alljobs),max(x['n'] for x in alljobs),min(x['m'] for x in alljobs),min(x['n'] for x in alljobs),
         p['cores']-p['active_cores']]
    assert len(raw)==len(NAMES)
    return [log2(1+max(0,x)) for x in raw]

def correction(x,model):
    # Standard-library inference of the fixed formula exp(c+v*tanh(W*x+b)).
    from math import tanh,exp
    z=[(a-b)/s for a,b,s in zip(x,model['center'],model['scale'])]
    h=[tanh(sum(a*w for a,w in zip(z,col))+b) for col,b in zip(model['W_columns'],model['hidden_bias'])]
    return exp(model['output_bias']+sum(a*w for a,w in zip(h,model['output_weights'])))
