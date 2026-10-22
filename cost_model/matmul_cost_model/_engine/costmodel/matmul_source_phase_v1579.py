"""Effective source half256 phase and periodic row-transition densities."""
from math import gcd,isfinite
from . import matmul_source_services_v1516 as Prior,matmul_streaming_v918 as Stream
FIELD='dma.catlass.rowwise_kcalls'
def phase_factors(row):
    if not Prior.rowwise(row):return (0.,0.)
    extra=row.get('source_a_offset_extra_bytes',0)
    if type(extra)!=int or extra not in (0,128,256,384):raise ValueError('unreviewed A pointer phase')
    if row['shape'][0]%16 or row['shape'][2]%16:raise ValueError('unreviewed rowwise source-phase domain')
    pitch=(2*row['shape'][2])%256;period=256//gcd(pitch,256)
    bits=[int((extra+i*pitch)%256>=128) for i in range(period)]
    bad=sum(bits)/period;turn=sum(bits[i]!=bits[i-1] for i in range(period))/period
    return bad,turn
def resolve(row,model):
    baseline=dict(model['baseline_model']);baseline['additional_rowwise_A_crossing_us_per_kline']=model['rowwise128_us_per_kline']
    F,c=Prior.resolve(row,baseline);bad=model['bad128_phase_us_per_kcall'];turn=model['phase_transition_us_per_kcall']
    if any(not isfinite(v) or v<0 for v in (bad,turn)):raise ValueError('invalid fixed phase service')
    if Prior.rowwise(row):
        setup=model.get('rowwise_setup_us_per_kcall',c['parameters'][FIELD])
        if FIELD in c['fixed_parameter_values'] or FIELD in F.FIXED_INTERVALS:raise ValueError('hardware service override forbidden')
        if not isfinite(setup) or setup<0:raise ValueError('invalid effective rowwise setup')
        a,b=phase_factors(row);c=dict(c,parameters={**c['parameters'],FIELD:setup+bad*a+turn*b})
    return F,c
def predict(row,model,engine='auto'):
    if engine not in ('auto','periodic','reference'):raise ValueError('unknown CPU engine')
    F,c=resolve(row,model)
    from .matmul_CPU_dispatch_v1006 import periodic_useful
    if engine=='periodic' or (engine=='auto' and periodic_useful(row)):
        from . import matmul_CPU_periodic_v1188 as Periodic
        value=Periodic.predict(F,row,c)
    else:value=Stream.predict(F,row,c)
    a,b=phase_factors(row)
    value.update(model_version=model['version'],static_bad128_phase_fraction=a,static_periodic_phase_transition_density=b,
        source_phase_features_are_effective_NOT_proven_hardware_banks=True,hardware_execution_performed=False,
        timing_input_required=False,accuracy_qualified=False,stable_ranking_qualified=False)
    return value
