"""Static native-plan BASE prototype; all services are fixed offline constants.

Reviewed schedule: FP16 NN, packed ND, 910B2, no conversion/bias/split-K,
singleCoreM/N == baseM/N, one L2 tile, stepM/N=1. A/B K slabs are independent.
This is a context-effective hypothesis, not independently identified PMU rates.
"""
from dataclasses import asdict, dataclass
import copy
import math
from .external_gemm_kernel import cube_service, mte1_service

WORDS={'usedCoreNum':0,'M':1,'N':2,'Ka':3,'Kb':4,'singleCoreM':5,
       'singleCoreN':6,'singleCoreK':7,'baseM':8,'baseN':9,'baseK':10,
       'depthA1':11,'depthB1':12,'stepM':13,'stepN':14,'iterateOrder':17,
       'shareL1Size':19,'shareL0CSize':20,'shareUbSize':21,'stepKa':26,
       'stepKb':27,'dbL0A':30,'dbL0B':31,'dbL0C':32}

def ceildiv(a,b): return (a+b-1)//b

@dataclass(frozen=True)
class Parameters:
    entry_us: float=1.
    task_us: float=.24
    control_us: float=.04
    copy_setup_us: float=.08
    copy_byte_us: float=.00001
    copy_row_us: float=.0007
    copy_ii_fraction: float=.5
    load_scale: float=1.
    cube_scale: float=1.
    fix_setup_us: float=.08
    fix_byte_us: float=.00002
    shared_byte_us: float=.000001

    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in asdict(self).values()):
            raise ValueError('finite nonnegative offline constants required')
        if not 0<self.copy_ii_fraction<=1:
            raise ValueError('copy interval fraction in (0,1] required')

class Clock:
    def __init__(self): self.ready=self.complete=0.
    def issue(self,ready,latency,ii=None):
        start=max(ready,self.ready)
        self.ready=start+(latency if ii is None else ii)
        self.complete=max(self.complete,start+latency)
        return start,start+latency

def validate_plan(plan):
    """Reject unsupported paths and verify structured metadata against raw ABI."""
    for field in plan:
        if any(x in field.lower() for x in ('time_ms','measured','pmu','actual_us','prediction')):
            raise ValueError('target timing/PMU fields are forbidden')
    if (plan.get('soc')!='Ascend910B2' or plan.get('dtype')!='float16'
        or plan.get('trans_a') is not False or plan.get('trans_b') is not False
        or plan.get('tiling_key')!=65536 or plan.get('tiling_bytes')!=280
        or plan.get('tcube_bytes')!=200):
        raise ValueError('only reviewed 910B2 FP16 NN BASE key65536/ABI280, no conversion')
    m,n,k=(plan[d] for d in ('m','n','k'))
    if any(type(v) is not int or v<=0 or v%16 for v in (m,n,k)):
        raise ValueError('positive 16-aligned M/N/K required')
    if m>4096 or n>4096 or k>8192:
        raise ValueError('prototype geometry bound M/N<=4096,K<=8192')
    raw=plan['raw_words']; t=plan['tcube']
    if len(raw)!=70 or any(type(v) is not int or not 0<=v<2**32 for v in raw):
        raise ValueError('exact uint32 ABI280 required')
    if any(t.get(name)!=raw[index] for name,index in WORDS.items()):
        raise ValueError('raw/structured TCube metadata mismatch')
    if (t['M'],t['N'],t['Ka'],t['Kb'])!=(m,n,k,k):
        raise ValueError('shape and TCube mismatch')
    bm,bn,bk=(t[x] for x in ('baseM','baseN','baseK'))
    if any(v<=0 or v%16 for v in (bm,bn,bk)) or t['singleCoreM']!=bm or t['singleCoreN']!=bn or t['singleCoreK']!=k:
        raise ValueError('one base output tile per Iterate required')
    if t['stepM']!=1 or t['stepN']!=1 or t['iterateOrder']!=0:
        raise ValueError('reviewed unit M/N steps and iterateOrder0 required')
    if raw[15] or any(raw[56:63]) or raw[50:52]!=[1,1] or raw[54]!=0:
        raise ValueError('bias/transposes/conversion/NZ/HF32/L2 split unsupported')
    if raw[52:54]!=[ceildiv(m,bm),ceildiv(n,bn)]:
        raise ValueError('L2 block grid mismatch')
    if (not 1<=t['usedCoreNum']<=24 or plan['block_dim']!=t['usedCoreNum']
        or any(t[x] not in (1,2) for x in ('dbL0A','dbL0B','dbL0C'))):
        raise ValueError('invalid launch or L0 depth')
    for operand in ('A','B'):
        step=t['stepK'+operand.lower()]
        depth=t['depth'+operand+'1']
        if step<=0 or depth%step or depth//step not in (1,2):
            raise ValueError('one or two complete L1 operand slots required')
    if ((t['depthA1']*bm+t['depthB1']*bn)*bk*2>524288
        or bm*bn*4*t['dbL0C']>131072
        or bm*bk*2*t['dbL0A']>65536 or bn*bk*2*t['dbL0B']>65536):
        raise ValueError('L1/L0 capacity exceeded')
    return {name:t[name] for name in WORDS}, (m,n,k)

def tasks_by_core(plan):
    t,(m,n,k)=validate_plan(plan)
    gm,gn=ceildiv(m,t['baseM']),ceildiv(n,t['baseN'])
    cores=t['usedCoreNum']; total=gm*gn; tasks=[]; seen=[]
    for core in range(cores):
        row=[]
        for r in range(total//cores+int(core<total%cores)):
            block=core+r*cores
            mi=block%gm; ni=(block+block//math.lcm(gm,gn))%gn
            seen.append((mi,ni))
            row.append((min(t['baseM'],m-mi*t['baseM']),
                        min(t['baseN'],n-ni*t['baseN']),mi,ni))
        tasks.append(row)
    if len(set(seen))!=total:
        raise ValueError('source block swizzle does not cover output exactly')
    return t,(m,n,k),tasks

def candidate_plans(native):
    """Four reviewed candidate families. Legal footprint is not correctness proof."""
    validate_plan(native)
    result=[]
    for name,settings in [('native',None),('m32n64_k128_s4_c24',(32,64,128,4,24)),
                          ('m64n32_k128_s4_c24',(64,32,128,4,24)),
                          ('m32n64_k128_s1_c12',(32,64,128,1,12))]:
        row=copy.deepcopy(native);row['candidate_name']=name
        if settings:
            bm,bn,bk,step,cores=settings;t=row['tcube'];m,n,k=(row[x] for x in ('m','n','k'))
            t.update(baseM=bm,baseN=bn,baseK=bk,singleCoreM=bm,singleCoreN=bn,singleCoreK=k,
                stepM=1,stepN=1,stepKa=step,stepKb=step,depthA1=2*step,depthB1=2*step,
                dbL0A=2,dbL0B=2,dbL0C=1,usedCoreNum=cores,iterateOrder=0,
                shareL1Size=4*bk*step*(bm+bn),shareL0CSize=4*bm*bn,shareUbSize=0)
            for field,index in WORDS.items():row['raw_words'][index]=t[field]
            row['raw_words'][50:56]=[1,1,ceildiv(m,bm),ceildiv(n,bn),0,0]
            row['block_dim']=cores
            row['resolver']='explicit_legal_BASE_candidate_v206_from_native_problem'
        validate_plan(row);result.append(row)
    return result

def predict_plan(plan,parameters,*,trace=False):
    p=parameters if isinstance(parameters,Parameters) else Parameters(**parameters)
    t,(m,n,k),tasks=tasks_by_core(plan)
    all_cores=[]; total_bytes=0; cache={}
    for core,row in enumerate(tasks):
        signature=tuple((a,b) for a,b,_,_ in row)
        if signature not in cache:
            scalar,dma,load,cube,fix=(Clock() for _ in range(5))
            l0=[0.]*min(t['dbL0A'],t['dbL0B']); cfree=[0.]*t['dbL0C']
            free={x:[0.]*(t['depth'+x.upper()+'1']//t['stepK'+x]) for x in 'ab'}
            seq={'a':0,'b':0}; consumed=0; logical_bytes=0; events=[]
            for ti,(tm,tn,_,_) in enumerate(row):
                _,task_ready=scalar.issue(cfree[ti%len(cfree)],p.task_us)
                acc=cfree[ti%len(cfree)]; source={}; source_slots={}
                for ko in range(0,k,t['baseK']):
                    tk=min(t['baseK'],k-ko)
                    for operand in 'ab':
                        span=t['baseK']*t['stepK'+operand]
                        if ko%span==0:
                            extent=min(span,k-ko)
                            rows,width=(tm,2*extent) if operand=='a' else (extent,2*tn)
                            slot=seq[operand]%len(free[operand]); seq[operand]+=1
                            _,control=scalar.issue(max(task_ready,free[operand][slot]),p.control_us)
                            service=p.copy_setup_us+p.copy_byte_us*rows*width+p.copy_row_us*rows
                            _,source[operand]=dma.issue(control,service,service*p.copy_ii_fraction)
                            source_slots[operand]=slot; logical_bytes+=rows*width
                    _,control=scalar.issue(task_ready,p.control_us)
                    mb,nb,kb=ceildiv(tm,16),ceildiv(tn,16),ceildiv(tk,16)
                    la=p.load_scale*mte1_service(1,0,mb*kb,0)
                    lb=p.load_scale*mte1_service(0,1,0,nb*kb)
                    _,enda=load.issue(max(control,source['a'],l0[consumed%len(l0)]),la,la*.7)
                    _,endb=load.issue(max(control,source['b'],l0[consumed%len(l0)]),lb,lb*.7)
                    for operand,end in (('a',enda),('b',endb)):
                        span=t['baseK']*t['stepK'+operand]
                        if (ko+tk)%span==0 or ko+tk==k:
                            free[operand][source_slots[operand]]=end
                    lc=p.cube_scale*cube_service(tm,tn,tk)
                    # Match the installed small-output PIPE_M condition;
                    # recurrence/II for larger output is an effective prior.
                    ci=lc if mb*nb<10 else p.cube_scale*kb*mb*nb*.9643561962910648/1800.
                    issue,acc_end=cube.issue(max(enda,endb,acc),lc,min(lc,ci))
                    acc=acc_end if mb*nb<10 else issue+ci
                    l0[consumed%len(l0)]=acc_end; consumed+=1
                    if trace: events.append(dict(task=ti,ko=ko,a_ready=source['a'],b_ready=source['b'],mte1_end=max(enda,endb),cube_start=issue,cube_end=acc_end))
                size=ceildiv(tm,16)*16*ceildiv(tn,16)*16*2
                fs,fe=fix.issue(acc_end,p.fix_setup_us+p.fix_byte_us*size)
                cfree[ti%len(cfree)]=fe
            value=dict(body_us=max(c.complete for c in (scalar,dma,load,cube,fix)),
                       logical_source_bytes=logical_bytes,events=events,
                       resources_us=dict(scalar=scalar.complete,mte2=dma.complete,
                                         mte1=load.complete,cube=cube.complete,fixpipe=fix.complete))
            cache[signature]=value
        value=cache[signature]; total_bytes+=value['logical_source_bytes']
        all_cores.append(dict(core=core,task_count=len(row),**value))
    local=max((c['body_us'] for c in all_cores),default=0.)
    shared=total_bytes*p.shared_byte_us
    return dict(model_version='ops_native_BASE_v206_prototype',
        predicted_us=p.entry_us+max(local,shared),matmul_body_us=local,
        shared_service_bound_us=shared,logical_source_bytes=total_bytes,
        active_cores=sum(bool(x) for x in tasks),launched_cores=t['usedCoreNum'],
        per_core=all_cores if trace else None,
        timing_input_required=False,hardware_execution_performed=False,
        accuracy_qualified=False,
        qualification='Native metadata/direct upstream BASE prototype; source schedule and shared-memory envelope are hypotheses. Qualification requires independent validation of this model and candidate domain.')
