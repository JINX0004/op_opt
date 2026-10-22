"""Separate fixed FP16 NN3D steady interval from completion latency.

The interval hypothesis is calibrated on component699, awaiting heldout700.
Extra latency is nonnegative and fitted offline; public inference is exact DAG.
"""
from math import isfinite
from . import matmul_four_path_v665 as base
from .matmul_four_path_v320 import validate_prediction_contract
LATENCY_NAMES=('load3d.a.extra_latency.calls_k','load3d.a.extra_latency.logical_mb',
               'load3d.b.extra_latency.calls_k','load3d.b.extra_latency.logical_mb')
PARAMETER_NAMES=base.PARAMETER_NAMES+LATENCY_NAMES
INDEX={n:i for i,n in enumerate(PARAMETER_NAMES)}
FIXED_INTERVALS={'load3d.a.calls_k':1000/1800.,'load3d.a.logical_mb':2e6/(512*1800.),
                 'load3d.b.calls_k':1000/1800.,'load3d.b.logical_mb':4e6/(512*1800.)}
compile_plan=base.compile_plan
class EventGraph(base.EventGraph):
    def node(self,predecessors=(),features=None,fraction=1.):
        self.predecessors.append(sorted(set(predecessors)-{0}))
        self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v])
        return len(self.features)-1
    def issue(self,clock,ready,features,fraction=1.):
        if self.native_load3d and 'load.load_kcalls' in features:
            operand='a' if self.native_load_count%2==0 else 'b';self.native_load_count+=1
            calls=features['load.load_kcalls'];mb=features['load.load_mb']
            start=self.join([clock['issue'],*ready])
            interval=self.node([start],{'load3d.'+operand+'.calls_k':calls,'load3d.'+operand+'.logical_mb':mb})
            end=self.node([interval],{'load3d.'+operand+'.extra_latency.calls_k':calls,
                                      'load3d.'+operand+'.extra_latency.logical_mb':mb})
            clock['issue']=interval;clock['complete']=self.join([clock['complete'],end])
            return end,interval
        return super().issue(clock,ready,features,fraction)
def compile_graph(row,graph=None,**schedule):
    validate_prediction_contract(row);graph=graph or EventGraph()
    if not isinstance(graph,EventGraph):raise ValueError('701 event graph required')
    # The old fractional issue schedule is replaced only for native3D.
    legacy=schedule.pop('load3d_ii',.7)
    if legacy!=.7:raise ValueError('701 has independent3D interval;legacy load3d_ii cannot be changed')
    return base.compile_graph(row,graph,load3d_ii=.7,**schedule)
def predict(row,model):
    validate_prediction_contract(row,model);theta=model['parameters']
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed services')
    if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in FIXED_INTERVALS.items()):raise ValueError('fixed calibrated3D interval changed')
    graph,sink,p=compile_graph(row,**model.get('schedule',{}));value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
    return dict(predicted_us=value,model_version=model.get('version','four_path_v701'),active_cores=p['active_cores'],
        launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),
        fixed_component_interval=True,first_completion_latency_is_separate=True,
        timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
