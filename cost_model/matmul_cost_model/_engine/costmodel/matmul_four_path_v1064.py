"""Development hypothesis: destination service competes with the ND source bound.

942 added descriptor geometry residuals after the measured ND interval. New
equal-allocation controls show negligible destination-stride effects for the
tested descriptors, while source pitch has a large effect. Here source/dest
residuals represent competing local service bounds, rather than mandatory
serial delays. Peer demand and first-source costs remain explicit. This is
an effective timing hypothesis, not a proved microarchitecture pipeline.
All original source scheduling and fixed component coefficients are retained.
"""
from .matmul_four_path_v942 import *
from . import matmul_four_path_v942 as B
from . import mte2_descriptor_geometry_v942 as D

LOCAL_SOURCE={'a_fill_stride_mb','a_source_excess_klines','a_row_crossing_klines','b_excess_klines','b_row_crossing_klines','b_strided_mb','b_even_c0_strided_short_density','b_even_c0_unique_short_lines'}
LOCAL_DESTINATION={'a_dst_density','b_dst_span_krows','b_dst_extra_column_blocks_span_krows','preload_b_dst_span_krows'}

class EventGraph(B.EventGraph):
    def issue(self,clock,ready,features,fraction=1.):
        if 'mte2.nd.interval.dense_calls' not in features:
            return super().issue(clock,ready,features,fraction)
        start=self.join([clock['issue'],*ready]);branches=[]
        for group in D.II_GROUPS:
            branches.append(self.node([start],{'mte2.nd.interval.'+n:features.get('mte2.nd.interval.'+n,0) for n in group}))
        residual={k:v for k,v in features.items() if not k.startswith('mte2.nd.')}
        for group in (LOCAL_DESTINATION,):
            local={k:v for k,v in residual.items() if k.startswith('dma.') and k.split('.',2)[2] in group}
            if local:
                branches.append(self.node([start],local))
                for k in local:residual.pop(k)
        issued=self.node(branches,residual)
        ends=[self.node([issued],{'mte2.nd.extra_completion.'+n:features.get('mte2.nd.extra_completion.'+n,0) for n in group}) for group in D.EXTRA_GROUPS]
        end=self.join(ends);clock['issue']=issued;clock['complete']=self.join([clock['complete'],end])
        return end,issued

def compile_graph(row,graph=None,**schedule):
    return B.compile_graph(row,graph or EventGraph(),**schedule)

def predict(row,model):
    validate_prediction_contract(row,model)
    theta=model['parameters'];fixed={**FIXED_INTERVALS,**model['component_fixed_mte2'],**model['component_fixed_native_fp32']}
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed offline services')
    if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in fixed.items()):raise ValueError('fixed component service changed')
    graph,sink,p=compile_graph(row,**model['schedule'])
    return dict(predicted_us=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink),model_version=model.get('version','four_path_v1064'),active_cores=p['active_cores'],launched_cores=p['cores'],generated_events=len(graph.features),timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
