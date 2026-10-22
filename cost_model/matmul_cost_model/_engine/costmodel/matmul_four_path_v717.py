"""Test partial overlap of BASE startup and active-core work.

The prior whole-body startup maximum can flatten different tiles to a tie.
This is an offline control hypothesis, not a claim of measured dispatch order.
"""
from math import isfinite
from . import matmul_four_path_v701 as base
from .matmul_four_path_v320 import validate_prediction_contract
DELAY_NAMES=('startup.active_delay_core.base','startup.active_delay_log_active.base')
PARAMETER_NAMES=base.PARAMETER_NAMES+DELAY_NAMES
INDEX={n:i for i,n in enumerate(PARAMETER_NAMES)}
FIXED_INTERVALS=base.FIXED_INTERVALS
compile_plan=base.compile_plan
class EventGraph(base.EventGraph):
    def __init__(self):super().__init__();self.startup_node=0;self.startup_features={}
    def plain_node(self,predecessors=(),features=None,fraction=1.):
        self.predecessors.append(sorted(set(predecessors)-{0}))
        self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v])
        return len(self.features)-1
    def node(self,predecessors=(),features=None,fraction=1.):
        ps=list(predecessors)
        if self.startup_node and not features and self.startup_node in ps:
            delayed=self.plain_node([p for p in ps if p!=self.startup_node],self.startup_features)
            return self.plain_node([self.startup_node,delayed])
        node=self.plain_node(ps,features,fraction)
        if features and 'launch_overlap.base' in features:
            self.startup_node=node;self.startup_features=dict(zip(DELAY_NAMES,(
                features['launch_overlap.base'],features.get('launch_overlap.active_base',0.))))
        return node
def compile_graph(row,graph=None,**schedule):
    validate_prediction_contract(row);graph=graph or EventGraph()
    if not isinstance(graph,EventGraph):raise ValueError('717 event graph required')
    return base.compile_graph(row,graph,**schedule)
def predict(row,model):
    validate_prediction_contract(row,model);theta=model['parameters']
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid fixed services')
    if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in FIXED_INTERVALS.items()):raise ValueError('fixed component interval changed')
    graph,sink,p=compile_graph(row,**model.get('schedule',{}));value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
    return dict(predicted_us=value,model_version=model.get('version','four_path_v717'),active_cores=p['active_cores'],launched_cores=p['cores'],
        split_factor=p['split_factor'],generated_events=len(graph.features),
        partial_startup_overlap_hypothesis=True,timing_input_required=False,hardware_execution_performed=False,
        accuracy_qualified=False,stable_ranking_qualified=False)
