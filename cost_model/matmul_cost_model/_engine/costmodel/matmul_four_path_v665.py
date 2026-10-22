"""Separate source-confirmed CANN FP16 Load3D A/B from CATLASS LoadData2D."""
from math import isfinite
from . import matmul_four_path_v614 as base
from .matmul_four_path_v320 import validate_prediction_contract
EXTRA_NAMES=('load3d.a.calls_k','load3d.a.logical_mb','load3d.b.calls_k','load3d.b.logical_mb')
PARAMETER_NAMES=base.PARAMETER_NAMES+EXTRA_NAMES
INDEX={n:i for i,n in enumerate(PARAMETER_NAMES)}
compile_plan=base.compile_plan
class EventGraph(base.EventGraph):
    def __init__(self):super().__init__();self.native_load3d=False;self.native_load_count=0;self.load3d_ii=.7
    def node(self,predecessors=(),features=None,fraction=1.):
        self.predecessors.append(sorted(set(predecessors)-{0}))
        self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v])
        return len(self.features)-1
    def issue(self,clock,ready,features,fraction=1.):
        if self.native_load3d and 'load.load_kcalls' in features:
            operand='a' if self.native_load_count%2==0 else 'b';self.native_load_count+=1
            features=dict(features)
            features['load3d.'+operand+'.calls_k']=features.pop('load.load_kcalls')
            features['load3d.'+operand+'.logical_mb']=features.pop('load.load_mb')
            fraction=self.load3d_ii
        return super().issue(clock,ready,features,fraction)
def compile_graph(row,graph=None,**schedule):
    validate_prediction_contract(row);graph=graph or EventGraph()
    if not isinstance(graph,EventGraph):raise ValueError('665 graph required')
    graph.native_load3d=row['implementation']!='catlass';graph.load3d_ii=schedule.pop('load3d_ii',.7)
    if not 0<graph.load3d_ii<=1:raise ValueError('invalid fixed Load3D issue interval hypothesis')
    return base.compile_graph(row,graph,**schedule)
def predict(row,model):
    validate_prediction_contract(row,model);theta=model['parameters']
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed mode service constants')
    graph,sink,p=compile_graph(row,**model.get('schedule',{}));value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
    return dict(predicted_us=value,model_version=model.get('version','four_path_v665'),active_cores=p['active_cores'],
        launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),
        static_mte1_mode='load2d' if row['implementation']=='catlass' else 'load3d_a_b',
        timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False)
