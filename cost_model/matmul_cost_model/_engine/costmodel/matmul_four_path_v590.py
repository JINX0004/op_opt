"""Aligned CATLASS MMAD PIPE_M branch, new v498; old301 unchanged.

Four-path finite-buffer max-plus event DAG. No measured inputs or shape lookup.

Implementation-conditioned source service max-plus model: max of engine service bounds within each output/K
chunk, serial chunk boundaries, max over cores/shared source, ordered conversion
and reduction. Effective service constants require independent qualification.
All times are us. Existing frozen versions remain unchanged.
"""
from .matmul_four_path_v320 import validate_prediction_contract
from .source_interval_ownership_v591 import attach_output_geometry
from .fixpipe_packet_geometry_v388 import packet_features
from .fp16_output_features_v402 import packet_features as fp16_packet_features
from math import ceil,gcd,log2,isfinite
from collections import Counter
from functools import lru_cache
from .ops_kernel_features import _conversion, _reduce, STRATEGIES
from .catlass_kernel_features import _reduction

IMPLEMENTATIONS=('catlass','base','scsk','deterministic')
DMA_BASE=('copy_calls','a_mb','b_mb','a_krows','b_krows','b_short_density','b_excess_klines',
     'rowwise_kcalls','nz_mb','b_span_density','b_strided_mb','b_strided_krows',
     'b_dst_span_krows','a_dst_density','b_tail_krows','a_source_excess_klines',
     'preload_b_strided_mb','preload_b_dst_span_krows','preload_packed_density',
     'preload_packed_span_density','nz_copy_calls','a_fill_mb','a_fill_stride_mb','b_cache_disable_mb','b_cache_disable_span_mb','peer_read_mb','peer_short_density','peer_copy_calls','short_b_descriptor_calls','nz_a_packet_span_mb','nd_a_packet_span_mb','b_dst_extra_column_blocks_span_krows','packed_b_partial_c0_krows','peer_b_strided_mb','same_address_b_short_density','b_even_c0_unique_short_lines','b_even_c0_strided_short_density','a_row_crossing_klines','b_row_crossing_klines')
DMA=tuple(i+'.'+x for i in IMPLEMENTATIONS for x in DMA_BASE)
LOAD=('load_kcalls','load_mb','catlass_b_transpose_kcalls','catlass_b_transpose_kblocks')
CUBE=('cube_kcalls','cube_kblocks','small_barriers','deep_kcalls')
SCALAR=('outer_kloops','inner_kloops','base_calls','sc_calls','det_calls','cat_tasks','atomic_calls')
ENGINES={'dma':DMA,'load':LOAD,'cube':CUBE,'scalar':SCALAR}
BOUNDARY=('fix_calls','fix_mb','atomic_mb','first_source_mb','first_source_krows','first_load_mb','first_cube_blocks')
CONVERSION=('copy_kcalls','transfer_mb','vector_kcalls','transdata_krepeats','scalar_kloops','barrier_kcalls')
REDUCTION=('copy_calls','transfer_mb','add_kblocks','cast_kblocks','barrier_calls')
PARAMETER_NAMES=tuple(f'{g}.{x}' for g,fs in ENGINES.items() for x in fs)+tuple('boundary.'+x for x in BOUNDARY)+tuple('conversion.'+x for x in CONVERSION)+tuple('reduction.'+x for x in REDUCTION)+tuple('entry.'+x for x in IMPLEMENTATIONS)+tuple('launch.'+x for x in IMPLEMENTATIONS)+('shared.mb','shared.density','conversion.entry','reduction.entry')
def cd(a,b):return (a+b-1)//b
def r16(x):return cd(x,16)*16
def catlass_split_factor(m,n,k,l1,cores):
    maximum=2 if k<=1024 else 4 if k<=2048 else 8 if k<=4096 else 16
    tiles=cd(m,l1[0])*cd(n,l1[1]);split=max(1,min(cores//tiles,maximum))
    if tiles<cores:
        while split+1<=maximum and cd(tiles*split,cores)>=cd(tiles,cores)*split:split+=1
    split=min(cd(k,l1[2]),split)
    if k>8192:split=max(2,split)
    if k>32768:split=max(4,split)
    return split
def partition(size,cores,ci):
    q,r=divmod(size,cores);s=ci*q+min(ci,r)
    return range(s,s+q+int(ci<r))

def line_excess(rows,width,stride,start):
    period=128//gcd(128,stride);q,r=divmod(rows,period)
    values=[cd((start+i*stride)%128+width,128)-width/128 for i in range(period)]
    return q*sum(values)+sum(values[:r])

def unique_source_lines(rows,width,stride,start):
    """Union of128B address intervals, not a claim about hardware requests.

    Rows are disjoint byte intervals and sorted, hence only consecutive
    intervals can share one boundary line. Count overlap periodically in
    at most128 iterations instead of enumerating K.
    """
    if rows<=0 or width<=0 or stride<width:
        raise ValueError('positive nonoverlapping static source rows required')
    period=128//gcd(128,stride)
    overlaps=[int((start+i*stride+width-1)//128==(start+(i+1)*stride)//128) for i in range(period)]
    q,r=divmod(rows-1,period)
    touched=rows*width/128+line_excess(rows,width,stride,start)
    return int(round(touched-q*sum(overlaps)-sum(overlaps[:r])))


def _chunk(kind,m,n,k,bm,bn,bk,sa,sb,*,global_k,global_n,mo=0,no=0,ko=0,nza=False,nzb=False,atomic=False):
    # The public event predictor re-extracts descriptor-level work. The old
    # per-chunk coarse engine bounds were unused: omit only that redundant
    # computation. Logical traffic and short-row density sums are exact.
    geometry=dict(kind=kind,m=m,n=n,k=k,bm=bm,bn=bn,bk=bk,sa=sa,sb=sb,
                  global_k=global_k,global_n=global_n,mo=mo,no=no,ko=ko,
                  nza=nza,nzb=nzb,atomic=atomic)
    density=0. if nzb else k*max(0.,1-2*n/128)*128/(2*global_n)/1000
    return dict(geometry=geometry,source_mb=2*(m+n)*k/1e6,source_density=density)


@lru_cache(maxsize=64)
def conversion_summary(m,n,k,cores,nza,nzb):
    if not (nza or nzb):return (0.,)*len(CONVERSION)
    fake=dict(m=m,n=n,k=k,block_dim=cores,nd2nz_a=nza,nd2nz_b=nzb)
    result=[0.]*len(CONVERSION)
    for ci in _conversion(fake,192*1024)['per_core']:
        f=ci['features'];values=[(f['mte2_calls']+f['mte3_calls'])/1000,
            (f['mte2_bytes']+f['mte3_bytes'])/1e6,f['nonzero_vector_api_calls']/1000,
            f['transdata_repeats']/1000,f['scalar_loop_iterations']/1000,
            (f['wait_flag_calls']+f['pipe_barrier_calls'])/1000]
        result=[max(a,b) for a,b in zip(result,values)]
    return tuple(result)

def compile_plan(row):
    """Only whitelisted static metadata is read. Labels cannot affect output."""
    kind=row['implementation'];m,n,k=row['shape'];cores=row['cores']
    contract=row.get('plan',row)
    if contract.get('dtype','float16')!='float16' or contract.get('trans_a',False) or contract.get('trans_b',False):
        raise ValueError('current four-path prototype supports FP16 NN only')
    if kind=='deterministic' and cores<2:raise ValueError('single-partial deterministic reducer is unreviewed')
    if kind not in IMPLEMENTATIONS or any(type(x)!=int or x<=0 for x in (m,n,k,cores)):
        raise ValueError('positive static geometry and known implementation required')
    if m>128 or n>128 or k>131072 or k<16 or k%16 or cores>24:
        raise ValueError('reviewed FP16 NN candidate domain M/N<=128,16<=K<=131072,K%16=0,cores<=24')
    work=[[] for _ in range(cores)];conv=[0.]*len(CONVERSION);red=[0.]*len(REDUCTION)
    convert=False;reduce=kind!='base';split=1
    if kind=='catlass':
        l1=row['l1_tile'];l0=row['l0_tile'];bm,bn,ks=l1;bk=l0[2]
        if l1[:2]!=l0[:2] or ks%bk or 4*(bm+bn)*ks>512*1024:
            raise ValueError('unreviewed CATLASS tile footprint')
        split=catlass_split_factor(m,n,k,l1,cores);gm,gn,gk=cd(m,bm),cd(n,bn),cd(k,ks)
        for si in range(split):
            start=(si*(gk//split)+min(si,gk%split))*ks
            size=min((gk//split+int(si<gk%split))*ks,k-start)
            for inner in range(gm*gn):
                block,rem=divmod(inner,3*gn);height=min(3,gm-3*block)
                mi,ni=3*block+rem%height,rem//height
                if block%2:ni=gn-1-ni
                ci=(si*gm*gn+inner)%cores
                work[ci].append(_chunk(kind,min(bm,m-mi*bm),min(bn,n-ni*bn),size,bm,bn,bk,ks//bk,ks//bk,
                    global_k=k,global_n=n,mo=mi*bm,no=ni*bn,ko=start))
        cfg=dict(m=m,n=n,splitk_factor=split,physical_aic=cores,aiv_subblocks_per_aic=2)
        vector,_=_reduction(cfg)
        for ci in vector:
            f=ci['totals'];values=[f.get('mte2_calls',0)+f.get('mte3_calls',0),
                (f.get('reduction_read_bytes',0)+f.get('output_write_bytes',0))/1e6,
                f.get('reduction_add_elements',0)/64/1000,f.get('output_cast_elements',0)/64/1000,
                f.get('hard_event_waits',0)]
            red=[max(a,b) for a,b in zip(red,values)]
    else:
        plan=row['plan'];t=plan['tcube'];bm,bn,bk=(t[x] for x in ('baseM','baseN','baseK'))
        sa,sb=(t[x] for x in ('stepKa','stepKb'));sk=t['singleCoreK'];nza=bool(plan['nd2nz_a']);nzb=bool(plan['nd2nz_b'])
        if kind=='deterministic' and cores>cd(k,sk):
            raise ValueError('deterministic reduction reads every launched core: cores must not exceed K partition count; idle partials are unwritten')
        if t['usedCoreNum']!=cores or [t[x] for x in ('M','N','Ka')]!=[m,n,k]:raise ValueError('static metadata disagreement')
        convert=nza or nzb
        conv=list(conversion_summary(m,n,k,cores,nza,nzb))
        for ci in range(cores):
            if kind=='deterministic':
                for seq,ki in enumerate(partition(cd(k,sk),cores,ci)):
                    work[ci].append(_chunk(kind,m,n,min(sk,k-ki*sk),bm,bn,bk,sa,sb,global_k=k,global_n=n,ko=ki*sk,nza=nza,nzb=nzb,atomic=bool(seq)))
            else:
                gm,gn=cd(m,t['singleCoreM']),cd(n,t['singleCoreN']);total=gm*gn
                if kind=='base':
                    # MatmulBaseBlock::UpdateBasicIndex, pinned calOrder0.
                    # The reviewed candidates have a single L2 tile, so the
                    # first blockIdxStart is0. realRound gives q=ci+j*cores.
                    raw=plan.get('raw_words')
                    if raw is not None and (len(raw)!=70 or raw[50:52]!=[1,1] or raw[54]!=0):
                        raise ValueError('BASE mapping reviewed only for one L2 tile and calOrder0')
                    indices=range(ci,total,cores)
                else:indices=partition(total,cores,ci)
                for ti in indices:
                    if kind=='base':
                        period=gm*gn//gcd(gm,gn)
                        mi,ni=ti%gm,(ti+ti//period)%gn
                    else:mi,ni=ti%gm,ti//gm
                    tm=min(t['singleCoreM'],m-mi*t['singleCoreM']);tn=min(t['singleCoreN'],n-ni*t['singleCoreN'])
                    for ki in range(cd(k,sk)):
                        work[ci].append(_chunk(kind,tm,tn,min(sk,k-ki*sk),bm,bn,bk,sa,sb,
                            global_k=k,global_n=n,mo=mi*t['singleCoreM'],no=ni*t['singleCoreN'],ko=ki*sk,nza=nza,nzb=nzb,atomic=ki>0))
        if reduce:
            if kind=='deterministic':
                span=cd(384*n,2*cores);length=min(m*n,span)
                red=[cores+1,(4*length*cores+2*length)/1e6,
                     (cores-1)*cd(length,64)/1000,cd(length,64)/1000,2*cores+1]
            elif n%128==0 and not convert:
                # Each active AIC's two AIVs cast its output tiles; launch
                # count does not divide the output of a one-tile core.
                candidates=[]
                for jobs in work:
                    seen=set();f=[0.]*len(REDUCTION)
                    for job in jobs:
                        g=job['geometry'];key=(g['mo'],g['no'])
                        if key in seen:continue
                        seen.add(key);length=cd(g['m'],2)*g['n']
                        values=[2,6*length/1e6,0,cd(length,64)/1000,3]
                        f=[a+b for a,b in zip(f,values)]
                    candidates.append(f)
                red=[max(x[i] for x in candidates) for i in range(len(REDUCTION))]
            elif n%64:
                split=max(1,m//(2*cores));height=split
                if m>=2*cores:height=max(split,m-(2*cores-1)*split)
                height=min(m,height);cast_count=height*cd(n,64)*64;length=height*n
                red=[2,(4*cast_count+2*length)/1e6,0,cd(cast_count,64)/1000,5]
            else:
                span=m*n//(2*cores);length=max(span,m*n-(2*cores-1)*span)
                red=[2,6*length/1e6,0,cd(length,64)/1000,5]
    # Same B starting address in a nominal job phase. Different M tiles
    # can request identical B; K-split cores often start at distinct K.
    # Actual overlap can drift; no runtime timing information is used here.
    cohorts=Counter((ji,job['geometry']['no'],job['geometry']['ko'])
                    for jobs in work for ji,job in enumerate(jobs))
    for jobs in work:
        for ji,job in enumerate(jobs):
            g=job['geometry'];g['same_address_b_clients']=cohorts[(ji,g['no'],g['ko'])]
    bypass=(kind=='scsk' and n%128==0 and not convert and row['plan']['tcube']['stepM']*row['plan']['tcube']['baseM']==m)
    for jobs in work:
        for job in jobs:job['geometry']['b_cache_bypass']=bypass
    return dict(implementation=kind,shape=[m,n,k],cores=cores,work=work,conversion=conv,reduction=red,
        has_conversion=convert,has_reduction=reduce,split_factor=split,
        source_mb=sum(j['source_mb'] for c in work for j in c),
        source_density=sum(j['source_density'] for c in work for j in c),
        active_cores=sum(bool(c) for c in work))


PARAMETER_NAMES=tuple(x for x in PARAMETER_NAMES if x not in ('scalar.inner_kloops','scalar.outer_kloops'))+tuple('scalar.'+x+'.'+i for x in ('inner_kloops','outer_kloops') for i in IMPLEMENTATIONS)+tuple('conversion.launch_entry.'+i for i in IMPLEMENTATIONS)+tuple('release.'+i for i in IMPLEMENTATIONS)+tuple('control.copy.'+i for i in IMPLEMENTATIONS)
PARAMETER_NAMES+=tuple('conversion.launch_core.'+i for i in IMPLEMENTATIONS)
PARAMETER_NAMES+=tuple('control.load.'+i for i in IMPLEMENTATIONS)
PARAMETER_NAMES+=('startup.first_source.base','startup.first_source.scsk','sync.scsk_global_peers','startup.first_source.catlass')
PARAMETER_NAMES+=('dma.catlass.rowwise_a_mb',)
PARAMETER_NAMES+=('writeback.fp32.calls', 'writeback.fp32.logical_mb', 'writeback.fp32.partial128_k', 'writeback.fp32.partial128_mb', 'writeback.fp32.fullwin_partial256_mb', 'writeback.fp32.trailing_partial256_mb', 'writeback.fp32.shortwin_partial256_mb', 'writeback.fp32.row_phase256_mb', 'writeback.fp32.packed_start64', 'writeback.fp32.packed_start128', 'writeback.fp32.row_gaps', 'writeback.fp32.atomic_calls', 'writeback.fp32.atomic_logical_mb', 'writeback.fp32.atomic_partial128_k', 'writeback.fp32.atomic_partial128_mb', 'writeback.fp32.atomic_fullwin_partial256_mb', 'writeback.fp32.atomic_trailing_partial256_mb', 'writeback.fp32.atomic_shortwin_partial256_mb', 'writeback.fp32.atomic_row_phase256_mb', 'writeback.fp32.atomic_packed_start64', 'writeback.fp32.atomic_packed_start128', 'writeback.fp32.atomic_row_gaps', 'writeback.fp32.peer_logical_mb')
PARAMETER_NAMES+=('writeback.base.calls', 'writeback.base.logical_mb', 'writeback.base.partial128_k', 'writeback.base.partial128_mb', 'writeback.base.fullwin_partial256_mb', 'writeback.base.trailing_partial256_mb', 'writeback.base.shortwin_partial256_mb', 'writeback.base.row_phase256_mb', 'writeback.base.packed_start64', 'writeback.base.packed_start128', 'writeback.base.row_gaps', 'writeback.base.peer_logical_mb', 'writeback.base.peer128_k', 'writeback.base.peer512_k', 'writeback.fp32.peer128_k', 'writeback.fp32.peer512_k')
PARAMETER_NAMES+=('launch_overlap.base', 'launch_overlap.active_base')
PARAMETER_NAMES+=('startup.first_b.catlass','scalar.scsk_output_sync','sync.scsk_global_calls')+tuple('dma.'+i+'.interval_b_peer_excess_short_density' for i in IMPLEMENTATIONS)
INDEX={name:i for i,name in enumerate(PARAMETER_NAMES)}

class EventGraph:
    """f[e] = max(f[predecessors]) + feature[e] dot fixed constants."""
    def __init__(self):
        self.predecessors=[[]];self.features=[[]]
    def node(self,predecessors=(),features=None,fraction=1.):
        self.predecessors.append(sorted(set(predecessors)-{0}))
        self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v])
        return len(self.features)-1
    def join(self,predecessors):
        p=sorted(set(predecessors)-{0})
        if not p:return 0
        if len(p)==1:return p[0]
        return self.node(p)
    def clock(self):return {'issue':0,'complete':0}
    def issue(self,clock,ready,features,fraction=1.):
        start=self.join([clock['issue'],*ready]);end=self.node([start],features)
        clock['issue']=end if fraction==1 else self.node([start],features,fraction)
        clock['complete']=end if fraction==1 else self.join([clock['complete'],end])
        return end,clock['issue']
    def evaluate(self,theta,sink):
        finish=[0.]*len(self.features)
        for i,(ps,fs) in enumerate(zip(self.predecessors,self.features)):
            finish[i]=max((finish[p] for p in ps),default=0.)+sum(theta[j]*v for j,v in fs)
        return finish[sink]

def _descriptor_features_expanded(g,operand,offset,extent):
    m,n,bm=g['m'],g['n'],g['bm'];kind=g['kind'];global_n=g['global_n'];global_k=g['global_k']
    rows,width=(m,2*extent) if operand=='a' else (extent,2*n)
    nz=g['nza'] if operand=='a' else g['nzb'];f=Counter()
    rowwise=kind=='catlass' and operand=='a' and global_k>=65536
    calls=rows if rowwise else 1
    # CopyNZ2NZImpl: B's NZ global row is K; sourceStride is in 32B
    # blocks. With FP16 C0=16, a >=65535 stride emits one copy per N0.
    if nz and operand=='b' and r16(global_k)-r16(rows)>=65535:calls=cd(n,16)
    # copy_calls/a_mb/b_mb/krows represent ND->NZ only. NZ->NZ uses
    # nz_copy_calls/nz_mb, with independent nonnegative costs.
    f['copy_calls']=0 if nz else calls
    f[operand+'_mb']=0. if nz else rows*width/1e6
    f[operand+'_krows']=0. if nz else rows/1000
    if rowwise:f['rowwise_kcalls']=calls/1000
    if nz:f['nz_mb']=2*r16(rows)*r16(width//2)/1e6;f['nz_copy_calls']=calls
    if operand=='a' and not nz:
        f['a_dst_density']=rows*128/(2*(bm if kind=='catlass' else r16(m)))/1000
        f['a_fill_mb']=rows*width/1e6*min(1.,m/bm)
        f['a_fill_stride_mb']=f['a_fill_mb']*max(0.,log2(global_k/4096))
        f['a_source_excess_klines']=line_excess(rows,width,2*global_k,512+2*(g['mo']*global_k+g['ko']+offset))/1000
        f['a_row_crossing_klines']=max(0.,f['a_source_excess_klines']-rows*(cd(width,128)-width/128)/1000)
    if operand=='b':
        step=g['sb'];span_log=log2(max(1,(g['bk']*step if kind=='catlass' else r16(extent))/128))
        f['b_dst_span_krows']=0. if nz else rows*span_log/1000
        if not nz:f['b_dst_extra_column_blocks_span_krows']=rows*span_log*max(0,cd(n,16)-1)/1000
        if g.get('b_cache_bypass',False):
            f['b_cache_disable_mb']=rows*width/1e6
            f['b_cache_disable_span_mb']=rows*width/1e6*span_log
        if kind=='scsk' and not nz:f['preload_b_dst_span_krows']=rows*span_log/1000
        if not nz:
            density=rows*max(0.,1-width/128)*128/(2*global_n)
            odd_c0=(n//16)%2
            f['b_short_density']=density/1000*odd_c0
            f['b_even_c0_unique_short_lines']=unique_source_lines(rows,width,2*global_n,
                512+2*((g['ko']+offset)*global_n+g['no']))*max(0.,1-width/128)/1000*(1-odd_c0)*int(n==global_n)
            f['b_even_c0_strided_short_density']=density/1000*(1-odd_c0)*int(n<global_n)
            f['b_excess_klines']=line_excess(rows,width,2*global_n,512+2*((g['ko']+offset)*global_n+g['no']))/1000
            f['b_row_crossing_klines']=max(0.,f['b_excess_klines']-rows*(cd(width,128)-width/128)/1000)
            f['b_span_density']=density*span_log/1000
            f['b_strided_mb']=rows*width/1e6*int(n<global_n);f['b_strided_krows']=rows/1000*int(n<global_n)
            f['b_tail_krows']=rows/1000*int(width%32!=0)
            if kind=='scsk':
                f['preload_b_strided_mb']=f['b_strided_mb'];f['preload_packed_density']=density/1000*int(n==global_n)
                f['preload_packed_span_density']=density*span_log/1000*int(n==global_n)
    peers=max(0,g.get('source_clients',1)-1)/24.
    f['peer_read_mb']=rows*width/1e6*peers
    f['peer_short_density']=f.get('b_short_density',0.)*peers
    f['peer_copy_calls']=calls*peers
    if operand=='b' and not nz:
        f['packed_b_partial_c0_krows']=rows/1000*int(n==global_n and n%16!=0)
        f['peer_b_strided_mb']=f.get('b_strided_mb',0.)*peers
        f['same_address_b_short_density']=f.get('b_short_density',0.)*max(0,g.get('same_address_b_clients',1)-1)
    if operand=='b' and not nz:
        f['short_b_descriptor_calls']=calls*max(0.,1-width/128)
    if operand=='a' and nz:
        f['nz_a_packet_span_mb']=f['nz_mb']*log2(max(1,extent/128))
    if operand=='a' and not nz:
        f['nd_a_packet_span_mb']=rows*width/1e6*log2(max(1,extent/128))
    features={'dma.'+kind+'.'+x:f[x] for x in DMA_BASE if f[x]}
    if operand=='b' and not nz and f.get('b_short_density',0.) and g.get('interval_b_peer_excess',0.):
        features['dma.'+kind+'.interval_b_peer_excess_short_density']=f['b_short_density']*g['interval_b_peer_excess']
    if rowwise:features['dma.catlass.rowwise_a_mb']=rows*width/1e6
    return features,calls


# Only derived descriptor work is cached. No model prediction/time/PMU is
# stored.128B source-line functions are invariant to adding128B to the base.
# The cache is bounded and returns copies, so downstream edits cannot leak.
_descriptor_feature_cache={}
def descriptor_features(g,operand,offset,extent):
    aphase=(2*(g['mo']*g['global_k']+g['ko']+offset))%128
    bphase=(2*((g['ko']+offset)*g['global_n']+g['no']))%128
    key=(operand,extent,g['kind'],g['m'],g['n'],g['bm'],g['global_n'],g['global_k'],
        g['nza'],g['nzb'],g['sb'],g['bk'],g.get('b_cache_bypass',False),
        g.get('source_clients',1),g.get('same_address_b_clients',1),
        g.get('interval_b_peer_excess',0.),aphase,bphase)
    saved=_descriptor_feature_cache.get(key)
    if saved is None:
        saved=_descriptor_features_expanded(g,operand,offset,extent)
        if len(_descriptor_feature_cache)>=4096:_descriptor_feature_cache.clear()
        _descriptor_feature_cache[key]=saved
    return dict(saved[0]),saved[1]

def compile_graph(row,graph=None,*,wait_location='pipeline',dma_ii=.5,det_slots=1,scalar_queue_depth=0,expand_microcommands=False):
    validate_prediction_contract(row)
    if type(scalar_queue_depth)!=int or scalar_queue_depth not in (0,1,2,4,8,16,32) or type(expand_microcommands)!=bool:raise ValueError('invalid offline command-credit schedule')
    if wait_location not in ('pipeline','scalar') or not 0<dma_ii<=1:raise ValueError('invalid fixed schedule hypothesis')
    program=attach_output_geometry(row,compile_plan(row));graph=graph or EventGraph();kind=program['implementation'];sinks=[];cache={};descriptor_cache={}
    if kind=='catlass':l1a=l1b=2;l0=2;lc=1
    else:
        t=row['plan']['tcube'];l1a=t['depthA1']//t['stepKa'];l1b=t['depthB1']//t['stepKb'];l0=min(t['dbL0A'],t['dbL0B']);lc=t['dbL0C']
        if kind=='deterministic':l1a=l1b=det_slots
    embedded_cast=kind=='scsk' and row['shape'][1]%128==0 and not program['has_conversion']
    for jobs in program['work']:
        for job in jobs:job['geometry']['source_clients']=program['active_cores']
        # Cache generated schedules with identical descriptor phases. This is
        # computation memoization, never a timing table.
        signature=tuple(tuple(sorted((x,(v%64 if x in ('ko','mo','no') else v)) for x,v in j['geometry'].items())) for j in jobs)
        if signature in cache:sinks.append(cache[signature]);continue
        clocks={x:graph.clock() for x in ('scalar','dma','load','cube','fix')};l0free=[0]*l0;cfree=[0]*lc;aiv=graph.clock();aiv_seen=False
        free={'a':[0]*l1a,'b':[0]*l1b};seq={'a':0,'b':0};inner=0
        credit={p:[0]*max(1,scalar_queue_depth) for p in ('dma','load','cube','fix')}
        issued={p:0 for p in credit}
        def command_ready(pipe):
            return [credit[pipe][issued[pipe]%scalar_queue_depth]] if scalar_queue_depth else []
        def command_done(pipe,end):
            if scalar_queue_depth:
                credit[pipe][issued[pipe]%scalar_queue_depth]=end;issued[pipe]+=1
        def enqueue_load(fs,calls,ready):
            count=calls if expand_microcommands else 1;ends=[]
            for micro in range(count):
                ctrl,_=graph.issue(clocks['scalar'],command_ready('load'),{'control.load.'+kind:calls/count})
                end,_=graph.issue(clocks['load'],[ctrl,*ready],{k:v/count for k,v in fs.items()},.7)
                command_done('load',end);ends.append(end)
            return graph.join(ends)

        for ji,job in enumerate(jobs):
            g=job['geometry'];m,n,k,bk=(g[x] for x in ('m','n','k','bk'))
            feature_signature=tuple(sorted((x,(v%64 if x in ('ko','mo','no') else v)) for x,v in g.items()))
            api={'scalar.'+{'base':'base_calls','scsk':'sc_calls','deterministic':'det_calls','catlass':'cat_tasks'}[kind]:1}
            if g['atomic']:api['scalar.atomic_calls']=1
            graph.issue(clocks['scalar'],[],api);acc=cfree[ji%lc];source={};slots={};copies={}
            for ko in range(0,k,bk):
                kk=min(bk,k-ko)
                if ko%(bk*g['sa'])==0 or ko%(bk*g['sb'])==0:
                    graph.issue(clocks['scalar'],[],{'scalar.outer_kloops.'+kind:.001})
                def issue_source(operand,offset):
                    key=(operand,offset)
                    if key in copies:return copies[key]
                    span=bk*g['s'+operand];extent=min(span,k-offset)
                    slot=seq[operand]%len(free[operand]);seq[operand]+=1
                    feature_key=(feature_signature,operand,offset%64,extent)
                    if feature_key not in descriptor_cache:
                        descriptor_cache[feature_key]=descriptor_features(g,operand,offset,extent)
                    fs,calls=descriptor_cache[feature_key]
                    # First B descriptor only, weighted peer row-interval demand. Distinct
                    # from per-descriptor steady client pressure and from
                    # launched idle cores. An effective offline hypothesis,
                    # not a measured hardware arbitration law.
                    if ji==0 and offset==0 and ((kind=='base' and operand=='b') or (kind in ('scsk','catlass') and operand=='a')):
                        fs=dict(fs)
                        fs['startup.first_source.'+kind]=g['first_b_peer_visits'] if kind=='base' else max(program['active_cores']-1,0)

                    if ji==0 and offset==0 and kind=='catlass' and operand=='b':
                        fs=dict(fs);fs['startup.first_b.catlass']=g['first_b_peer_visits']
                    count=calls if expand_microcommands else 1;ends=[]
                    for micro in range(count):
                        ready=([free[operand][slot]] if wait_location=='scalar' else [])+command_ready('dma')
                        control,_=graph.issue(clocks['scalar'],ready,{'control.copy.'+kind:calls/count})
                        part={name:(value if micro==0 else 0.) if name.startswith('startup.') else value/count for name,value in fs.items()}
                        end,_=graph.issue(clocks['dma'],[control,free[operand][slot]],part,dma_ii)
                        command_done('dma',end);ends.append(end)
                    done=graph.join(ends)
                    copies[key]=(done,slot);return done,slot
                for operand in ('a','b'):
                    span=bk*g['s'+operand]
                    if ko%span==0:source[operand],slots[operand]=issue_source(operand,ko)
                if kind=='catlass' and ko%(bk*g['sa'])==0:
                    # block_mmad_pingpong.hpp loads packet i+1 before the
                    # first L1->L0 read of packet i. Two L1 credits bound it.
                    for operand in ('a','b'):
                        next_offset=ko+bk*g['s'+operand]
                        if next_offset<k:issue_source(operand,next_offset)
                control,_=graph.issue(clocks['scalar'],[],{'scalar.inner_kloops.'+kind:.001})
                mb,nb,kb=cd(m,16),cd(n,16),cd(kk,16)
                fa={'load.load_kcalls':(mb if kind=='catlass' else 1)/1000,'load.load_mb':2*r16(m)*r16(kk)/1e6}
                fb={'load.load_kcalls':(kb if kind=='catlass' else 1)/1000,'load.load_mb':2*r16(n)*r16(kk)/1e6}
                if kind=='catlass':fb.update({'load.catlass_b_transpose_kcalls':kb/1000,'load.catlass_b_transpose_kblocks':kb*nb/1000})
                # CATLASS emits mb separate A and kb separate B LoadData
                # instructions here. Charge their command-control work to
                # Scalar independently of MTE1 service; coefficients fixed
                # offline. This does not claim a measured queue capacity.
                enda=enqueue_load(fa,mb if kind=='catlass' else 1,[control,source['a'],l0free[inner%l0]])
                endb=enqueue_load(fb,kb if kind=='catlass' else 1,[control,source['b'],l0free[inner%l0]])
                for operand,end in (('a',enda),('b',endb)):
                    span=bk*g['s'+operand]
                    if (ko+kk)%span==0 or ko+kk==k:
                        free[operand][slots[operand]]=graph.node([end],{'release.'+kind:1})
                small_barrier=(mb*nb<10) if kind=='catlass' else (m//16)*(n//16)<10
                fc={'cube.cube_kcalls':.001,'cube.cube_kblocks':mb*nb*kb/1000,
                    'cube.small_barriers':int(small_barrier)/1000,'cube.deep_kcalls':int(mb==nb==1 and kk>=512)/1000}
                cube_ready=[enda,endb,acc]
                if scalar_queue_depth:
                    cmd,_=graph.issue(clocks['scalar'],command_ready('cube'),{});cube_ready.append(cmd)
                cend,cissue=graph.issue(clocks['cube'],cube_ready,fc,1. if small_barrier else .7)
                command_done('cube',cend)
                acc=cend if small_barrier else cissue;l0free[inner%l0]=cend;inner+=1
            fix={'boundary.fix_calls':1,'boundary.fix_mb':r16(m)*r16(n)*(2 if kind=='base' else 4)/1e6}
            if g['atomic']:fix['boundary.atomic_mb']=fix['boundary.fix_mb']
            if kind=='base':
                of=fp16_packet_features(m,n,g['output_ldc'],g['output_phase_bytes']);of['calls']=1.
                fix={'writeback.base.'+x:of[x] for x in ('calls', 'logical_mb', 'partial128_k', 'partial128_mb', 'fullwin_partial256_mb', 'trailing_partial256_mb', 'shortwin_partial256_mb', 'row_phase256_mb', 'packed_start64', 'packed_start128', 'row_gaps') if of[x]}
                fix['writeback.base.peer_logical_mb']=max(program['active_cores']-1,0)*of['logical_mb']
                wb_prefix='writeback.base.'
            else:
                # These are native effective services calibrated on316,not
                # literal389 component latencies. BASE FP16 remains separate.
                of=packet_features(m,n,g['output_ldc'],g['output_phase_bytes'])
                of['calls']=1.
                wb_prefix='writeback.'+'fp32'+'.'
                fix={wb_prefix+x:of[x] for x in ('calls', 'logical_mb', 'partial128_k', 'partial128_mb', 'fullwin_partial256_mb', 'trailing_partial256_mb', 'shortwin_partial256_mb', 'row_phase256_mb', 'packed_start64', 'packed_start128', 'row_gaps') if of[x]}
                if g['atomic']:fix.update({wb_prefix+'atomic_'+x:of[x] for x in ('calls', 'logical_mb', 'partial128_k', 'partial128_mb', 'fullwin_partial256_mb', 'trailing_partial256_mb', 'shortwin_partial256_mb', 'row_phase256_mb', 'packed_start64', 'packed_start128', 'row_gaps') if of[x]})
                fix[wb_prefix+'peer_logical_mb']=max(program['active_cores']-1,0)*of['logical_mb']
            fix[wb_prefix+'peer128_k']=g['output_peer128_k']
            fix[wb_prefix+'peer512_k']=g['output_peer512_k']
            fix_ready=[cend]
            if scalar_queue_depth:
                cmd,_=graph.issue(clocks['scalar'],command_ready('fix'),{});fix_ready.append(cmd)
            cfree[ji%lc],_=graph.issue(clocks['fix'],fix_ready,fix)
            command_done('fix',cfree[ji%lc])
            # Upstream SCSK emits PIPE_ALL after the full K loop of an output
            # tile in its aligned N128 path, never after every K partition.
            last=ji+1==len(jobs) or any(jobs[ji+1]['geometry'][x]!=g[x] for x in ('mo','no'))
            if kind=='scsk' and row['shape'][1]%128==0 and not program['has_conversion'] and last:
                clocks['scalar']['issue']=graph.join([c['complete'] for c in clocks.values()])
                graph.issue(clocks['scalar'],[],{'scalar.scsk_output_sync':1})
                # ProcessBlockMKN: AIC notifies after all K for this output;
                # its AIV pair casts that output independently while AIC
                # can continue the next MN output. No AIV->AIC credit here.
                length=cd(g['m'],2)*g['n']
                cast={'reduction.copy_calls':2,'reduction.transfer_mb':6*length/1e6,
                      'reduction.cast_kblocks':cd(length,64)/1000,'reduction.barrier_calls':3}
                if not aiv_seen:cast['reduction.entry']=1;aiv_seen=True
                # Source NotifyEvent<PIPE_FIX> precedes PipeBarrier<PIPE_ALL>:
                # vector can proceed after Fix while Scalar pays output sync.
                graph.issue(aiv,[cfree[ji%lc]],cast)

        sink=graph.join([*[c['complete'] for c in clocks.values()],aiv['complete']]);cache[signature]=sink;sinks.append(sink)
    shared=graph.node(features={'shared.mb':program['source_mb']})
    # Effective startup service can overlap the useful work of active AICs.
    # Keep the original serial entry/per-core term. Zero new rates recover it.
    # BASE with conversion is MIX and deliberately keeps its original topology.
    startup=0
    if kind=='base' and not program['has_conversion']:
        startup_features={'launch_overlap.base':program['cores']}
        startup_features['launch_overlap.active_base']=log2(max(1,program['active_cores']))
        startup=graph.node(features=startup_features)
    body=graph.join([*sinks,shared,startup]);phase={'entry.'+kind:1,'launch.'+kind:program['cores'],
        'conversion.entry':int(program['has_conversion']),'conversion.launch_entry.'+kind:int(program['has_conversion']),
        'conversion.launch_core.'+kind:program['cores']*int(program['has_conversion']),'reduction.entry':int(program['has_reduction'])}
    for prefix,fields,values in [('conversion',CONVERSION,program['conversion']),('reduction',REDUCTION,program['reduction'])]:
        phase.update({prefix+'.'+x:v for x,v in zip(fields,values) if v})
    # Pinned SCSK uses two AIV SyncAll barriers for unaligned inputs or
    # N not128-aligned. Aligned N128 instead uses per-tile CV notification.
    # C220 SyncAll is hardware FFTS+wait, not a software O(C) loop.
    # This peer-count feature is an effective fan-in hypothesis, not a law.
    if embedded_cast:
        for key in ['reduction.entry',*['reduction.'+x for x in REDUCTION]]:phase.pop(key,None)
    global_sync=kind=='scsk' and (program['has_conversion'] or row['shape'][1]%128!=0)
    if global_sync:
        phase['sync.scsk_global_peers']=2*max(2*program['cores']-1,0)
        phase['sync.scsk_global_calls']=2
    sink=graph.node([body],phase)
    return graph,sink,program

def predict(row,model):
    validate_prediction_contract(row,model)
    theta=model['parameters']
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed offline event services')
    graph,sink,program=compile_graph(row,**model.get('schedule',{}))
    value=graph.evaluate([theta[x] for x in PARAMETER_NAMES],sink)
    return dict(predicted_us=value,model_version=model.get('version','four_path_v590'),
        active_cores=program['active_cores'],launched_cores=program['cores'],split_factor=program['split_factor'],
        generated_events=len(graph.features),timing_input_required=False,hardware_execution_performed=False,
        accuracy_qualified=bool(model.get('accuracy_qualified',False)))
