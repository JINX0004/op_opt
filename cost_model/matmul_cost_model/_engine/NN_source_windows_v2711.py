"""Static ND row-window crossings and nominal job-wave source sharing.

These are geometric proxies, not observed memory transactions or simultaneous
hardware arrivals. No target measurements or identifiers enter this function.
"""
from math import gcd,ceil,log2
from collections import Counter
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan
from matmul_cost_model._engine.NN_static_budget_features_v2605 import features as old_features,NAMES as OLD_NAMES
EXTRA=('max_A_cross256_k','max_B_cross256_k','max_A_cross512_k','max_B_cross512_k',
       'total_A_cross256_k','total_B_cross256_k','wave_A_readers','wave_B_readers',
       'wave_A_peer_MB','wave_B_peer_MB')
NAMES=OLD_NAMES+EXTRA+tuple(n+'_per_static_us' for n in EXTRA[:6]+EXTRA[8:])

def crossings(rows,width,stride,start,line):
    period=line//gcd(line,stride)
    cycle=[max(0,ceil(((start+j*stride)%line+width)/line)-ceil(width/line)) for j in range(period)]
    q,r=divmod(rows,period)
    return q*sum(cycle)+sum(cycle[:r])

def job_cross(g,operand,line):
    if g['nz'+operand]:return 0
    span=g['bk']*g['s'+operand];full,tail=divmod(g['k'],span)
    K,N=g['global_k'],g['global_n']
    if operand=='a':
        start=512+2*(g['mo']*K+g['ko']);step=2*span
        one=lambda s,e:crossings(g['m'],2*e,2*K,s,line)
    else:
        start=512+2*(g['ko']*N+g['no']);step=2*span*N
        one=lambda s,e:crossings(e,2*g['n'],2*N,s,line)
    period=line//gcd(line,step)
    values=[one(start+i*step,span) for i in range(period)]
    q,r=divmod(full,period)
    return q*sum(values)+sum(values[:r])+(one(start+full*step,tail) if tail else 0)

def geometry(row):
    static={k:row[k] for k in ('implementation','shape','cores')}
    for k in ('plan','tile','l1_tile','l0_tile','execution_profile','transport'):
        if k in row:static[k]=row[k]
    p=compile_plan(static);work=[[j['geometry'] for j in js] for js in p['work']]
    per=[[sum(job_cross(g,a,L) for g in js) for a,L in (('a',256),('b',256),('a',512),('b',512))] for js in work]
    raw=[max(v[i] for v in per)/1000 for i in range(4)]+[sum(v[i] for v in per)/1000 for i in range(2)]
    peers=[1,1];peer_mb=[0.,0.]
    for wave in range(max(map(len,work))):
        jobs=[js[wave] for js in work if len(js)>wave]
        for i,a in enumerate(('a','b')):
            keys=[(g['mo' if a=='a' else 'no'],g['ko'],g['m' if a=='a' else 'n'],
                   min(g['k'],g['bk']*g['s'+a]),g['nz'+a]) for g in jobs]
            counts=Counter(keys)
            for key,count in counts.items():
                peers[i]=max(peers[i],count)
                peer_mb[i]=max(peer_mb[i],(count-1)*2*key[2]*key[3]/1e6)
    return raw+peers+peer_mb

def features(row,static_prediction_us):
    t=float(static_prediction_us)
    if t<=0:raise ValueError('positive internally computed static time required')
    raw=geometry(row)
    return old_features(row,t)+[log2(1+x) for x in raw]+[raw[i]/t for i in (*range(6),8,9)]
