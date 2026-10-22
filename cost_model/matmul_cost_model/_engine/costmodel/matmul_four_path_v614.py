"""590 plus source-derived SC embedded Cast32to16 strided descriptor services.

Existing conversion only; no independent vector operator is added. 128B
interval counts describe address geometry, not measured hardware traffic.
"""
from math import isfinite
from . import matmul_four_path_v590 as base
from .source_interval_ownership_v591 import attach_output_geometry
from .matmul_four_path_v320 import validate_prediction_contract
EXTRA_NAMES=('cast.scsk.strided_rows_k','cast.scsk.strided_logical_mb',
    'cast.scsk.read_excess128_k','cast.scsk.write_excess128_k')
PARAMETER_NAMES=base.PARAMETER_NAMES+EXTRA_NAMES
INDEX={x:i for i,x in enumerate(PARAMETER_NAMES)}
compile_plan=base.compile_plan

def cast_features(g,global_n):
    rows=base.cd(g['m'],2);n=g['n'];element_start=g['mo']*global_n+g['no']
    return dict(zip(EXTRA_NAMES,(
        rows/1000 if n<global_n else 0.,6*rows*n/1e6 if n<global_n else 0.,
        base.line_excess(rows,4*n,4*global_n,512+4*element_start)/1000,
        base.line_excess(rows,2*n,2*global_n,512+2*element_start)/1000)))

def _cast_contexts(row):
    if row['implementation']!='scsk' or row['shape'][1]%128:return []
    p=attach_output_geometry(row,compile_plan(row))
    if p['has_conversion']:return []
    seen=set();contexts=[]
    for jobs in p['work']:
        for job in jobs:job['geometry']['source_clients']=p['active_cores']
        signature=tuple(tuple(sorted((x,(v%64 if x in ('ko','mo','no') else v)) for x,v in j['geometry'].items())) for j in jobs)
        if signature in seen:continue
        seen.add(signature)
        for ji,job in enumerate(jobs):
            g=job['geometry']
            if ji+1==len(jobs) or any(jobs[ji+1]['geometry'][x]!=g[x] for x in ('mo','no')):
                contexts.append(cast_features(g,row['shape'][1]))
    return contexts

class EventGraph(base.EventGraph):
    def __init__(self):super().__init__();self.cast_contexts=iter(());self.cast_count=0
    def node(self,predecessors=(),features=None,fraction=1.):
        self.predecessors.append(sorted(set(predecessors)-{0}))
        self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v])
        return len(self.features)-1
    def issue(self,clock,ready,features,fraction=1.):
        if 'reduction.copy_calls' in features and 'reduction.cast_kblocks' in features:
            try:extra=next(self.cast_contexts)
            except StopIteration:raise RuntimeError('SC cast source context exhausted')
            features=dict(features,**extra);self.cast_count+=1
        return super().issue(clock,ready,features,fraction)

def compile_graph(row,graph=None,**schedule):
    validate_prediction_contract(row)
    graph=graph or EventGraph()
    if not isinstance(graph,EventGraph):raise ValueError('614 event graph required')
    contexts=_cast_contexts(row);graph.cast_contexts=iter(contexts)
    result=base.compile_graph(row,graph,**schedule)
    if graph.cast_count!=len(contexts):raise RuntimeError('SC cast source context mismatch')
    return result

def predict(row,model):
    validate_prediction_contract(row,model);theta=model['parameters']
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed offline event services')
    graph,sink,p=compile_graph(row,**model.get('schedule',{}));value=graph.evaluate([theta[x] for x in PARAMETER_NAMES],sink)
    return dict(predicted_us=value,model_version=model.get('version','four_path_v614'),active_cores=p['active_cores'],
        launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),
        timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False)
