"""Static layout/service summaries; no observations, identifiers or shape table."""
from math import log2
from matmul_cost_model._engine.static_cache_flag_features_v1442 import features as old_features,NAMES as OLD_NAMES,correction
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan,attach_output_geometry,packet_features,fp16_packet_features,cd
from matmul_cost_model._engine.costmodel.matmul_source_services_v1516 import resolve,rowwise

EXTRA_NAMES=('CAT_ROWWISE_A_API','pred_fix_work_fraction','pred_fix_calls_fraction',
    'pred_fix_partial256_fraction','pred_fix_atomic_fraction','pred_rowwise_crossing_fraction',
    'logical_output_partial256_fraction','max_output_phase256_fraction',
    'source_padding_fraction','rowwise_copy_count_per_mmad','L1A_buffer_slots','L1B_buffer_slots')
NAMES=OLD_NAMES+EXTRA_NAMES

def features(row,pipeline_us,model):
    x=old_features(row,pipeline_us,model['baseline_model'])
    F,c=resolve(row,model);theta=c['parameters'];p=attach_output_geometry(row,compile_plan(row));kind=row['implementation']
    rw=rowwise(row);values=[]
    # All quantities are reconstructed from the explicit static configuration.
    for jobs in p['work']:
        if not jobs:continue
        fix=calls=partial=atomic=cross=logical_partial=output_bytes=phase=source=padding=source_calls=mmad=0.
        for job in jobs:
            g=job['geometry'];m,n,k,bk=(g[t] for t in ('m','n','k','bk'));mmad+=cd(k,bk)
            prefix='writeback.'+('base' if kind=='base' else 'fp32' if kind=='catlass' else 'native_fp32')+'.'
            of=(fp16_packet_features if kind=='base' else packet_features)(m,n,g['output_ldc'],g['output_phase_bytes'])
            of['calls']=1.;of['peer_logical_mb']=max(p['active_cores']-1,0)*of['logical_mb']
            of['peer128_k']=g['output_peer128_k'];of['peer512_k']=g['output_peer512_k']
            cost=sum(theta.get(prefix+t,0.)*v for t,v in of.items())
            atom=sum(theta.get(prefix+'atomic_'+t,0.)*v for t,v in of.items()) if g['atomic'] else 0.
            fix+=cost+atom;atomic+=atom;calls+=theta.get(prefix+'calls',0.)
            partial+=sum(theta.get(prefix+t,0.)*of.get(t,0.) for t in ('fullwin_partial256_mb','trailing_partial256_mb','shortwin_partial256_mb','row_phase256_mb'))
            size=m*n*(2 if kind=='base' else 4);output_bytes+=size
            logical_partial+=size*((n*(2 if kind=='base' else 4))%256)/256
            phase+=size*(g['output_phase_bytes']%256)/256
            for operand in ('a','b'):
                span=bk*g['s'+operand]
                seen={}
                for off in range(0,k,span):
                    ext=min(span,k-off);rr,cc=(m,ext) if operand=='a' else (ext,n)
                    logical=2*rr*cc;source+=logical; padding+=2*cd(rr,16)*16*cd(cc,16)*16-logical
                    if rw and operand=='a':
                        source_calls+=rr
                        key=(off%256,ext)
                        if key not in seen:seen[key]=F.Schedule.descriptor_features(g,operand,off,ext)[0]
                        fs=seen[key];cross+=theta.get('dma.catlass.a_row_crossing_klines',0.)*fs.get('dma.catlass.a_row_crossing_klines',0.)
        values.append((fix/pipeline_us,calls/pipeline_us,partial/pipeline_us,atomic/pipeline_us,cross/pipeline_us,
                       logical_partial/max(output_bytes,1),phase/max(output_bytes,1),padding/max(source,1),source_calls/max(mmad,1)))
    t=row.get('plan',{}).get('tcube',{})
    slotsA=2 if kind=='catlass' else t['depthA1']/t['stepKa']
    slotsB=2 if kind=='catlass' else t['depthB1']/t['stepKb']
    raw=[int(rw)]+[max(v[j] for v in values) for j in range(9)]+[slotsA,slotsB]
    assert len(raw)==len(EXTRA_NAMES)
    return x+[log2(1+max(0.,v)) for v in raw]
