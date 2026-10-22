"""590 source schedule;native descriptor max-service from847,old frozen source unchanged."""
from .matmul_four_path_v590 import *
from .mte2_descriptor_geometry_v1121 import descriptor_features

def compile_graph(row,graph=None,*,wait_location='pipeline',dma_ii=.5,det_slots='source',scalar_queue_depth=0,expand_microcommands=False):
    validate_prediction_contract(row)
    if type(scalar_queue_depth)!=int or scalar_queue_depth not in (0,1,2,4,8,16,32) or type(expand_microcommands)!=bool:raise ValueError('invalid offline command-credit schedule')
    if wait_location not in ('pipeline','scalar') or not 0<dma_ii<=1:raise ValueError('invalid fixed schedule hypothesis')
    program=attach_output_geometry(row,compile_plan(row));graph=graph or EventGraph();kind=program['implementation'];sinks=[];cache={};descriptor_cache={}
    if kind=='catlass':l1a=l1b=2;l0=2;lc=1
    else:
        t=row['plan']['tcube'];l1a=t['depthA1']//t['stepKa'];l1b=t['depthB1']//t['stepKb'];l0=min(t['dbL0A'],t['dbL0B']);lc=t['dbL0C']
        if kind=='deterministic':
            if det_slots=='source':
                if not (t['baseM']==t['baseN']==128 and t['iterateOrder']==1 and t['stepM']==3 and 1<=t['stepKa']==t['stepKb']<=3):raise ValueError('unreviewed deterministic source buffer policy')
                l1a=cd(t['singleCoreM'],t['baseM'])
                l1b=1+((t['depthB1']//(t['stepKb']*t['stepN'])-1)&1)
            else:l1a=l1b=det_slots
    embedded_cast=kind=='scsk' and row['shape'][1]%128==0 and not program['has_conversion']
    for jobs in program['work']:
        for job in jobs:job['geometry']['source_clients']=program['active_cores']
        # Cache generated schedules with identical descriptor phases. This is
        # computation memoization, never a timing table.
        signature=tuple(tuple(sorted((x,(v%256 if x in ('ko','mo','no') else v)) for x,v in j['geometry'].items())) for j in jobs)
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
            feature_signature=tuple(sorted((x,(v%256 if x in ('ko','mo','no') else v)) for x,v in g.items()))
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
                    feature_key=(feature_signature,operand,offset%256,extent)
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
                    count=calls if expand_microcommands or kind!='catlass' else 1;ends=[]
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
                wb_prefix='writeback.'+('fp32' if kind=='catlass' else 'native_fp32')+'.'
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

