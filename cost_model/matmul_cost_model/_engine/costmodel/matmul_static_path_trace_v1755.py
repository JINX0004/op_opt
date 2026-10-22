"""One valid dominant-path decomposition of the static model, never PMU.

Store only a winning parent and sparse own costs for each emitted node.
No target measurement, NumPy, network, subprocess or hardware execution.
"""
from array import array
from math import isfinite
from . import matmul_source_phase_v1579 as Source
from .matmul_streaming_v918 import _Predecessors

GROUPS=('source_service','load_service','cube_service','control_service',
        'writeback_service','conversion_reduction_service','shared_service','boundary_service')

def group(name):
    if name.startswith(('rowwise_source::','rowwise_phase::','ordinaryND_A::')):return 'source_service'
    family=name.split('::')[-1].split('.')[0]
    for families,key in ((('dma','mte2'),'source_service'),(('load','load3d'),'load_service'),
            (('cube',),'cube_service'),(('scalar','control','release','sync'),'control_service'),
            (('writeback',),'writeback_service'),(('conversion','reduction','cast'),'conversion_reduction_service'),
            (('shared','shared_l1'),'shared_service'),(('entry','init','launch','launch_overlap','startup','boundary'),'boundary_service')):
        if family in families:return key
    raise ValueError('unclassified static model term: '+name)

class _TraceFeatures:
    def __init__(self,graph):self.graph=graph
    def __len__(self):return len(self.graph._finish)
    def append(self,features):
        g=self.graph
        if g._pending is None:raise RuntimeError('missing source predecessors')
        parent=max(g._pending,key=g._finish.__getitem__,default=0);cost=[0.]*len(GROUPS);total=0.
        for j,value in features:
            part=g._theta[j]*value;total+=part;cost[g._group[j]]+=part
        g._finish.append(g._finish[parent]+total);g._winner.append(parent)
        for j,value in enumerate(cost):
            if value:g._fi.append(j);g._fv.append(value)
        g._fp.append(len(g._fi));g._pending=None

def graph_type(formula):
    class TraceGraph(formula.EventGraph):
        def __init__(self,theta):
            super().__init__();self._theta=theta;self._finish=array('d',[0.]);self._pending=None
            self._winner=array('I',[0]);self._fp=array('I',[0,0]);self._fi=array('B');self._fv=array('d')
            self._group=tuple(GROUPS.index(group(n)) for n in formula.PARAMETER_NAMES)
            self.predecessors=_Predecessors(self);self.features=_TraceFeatures(self)
        def evaluate(self,theta,sink):
            if tuple(theta)!=self._theta or self._pending is not None:raise ValueError('changed weights or incomplete graph')
            return self._finish[sink]
        def decompose(self,sink):
            parts=[0.]*len(GROUPS);node=sink
            while node:
                for j in range(self._fp[node],self._fp[node+1]):parts[self._fi[j]]+=self._fv[j]
                node=self._winner[node]
            return parts
    return TraceGraph

def predict(row,model):
    F,c=Source.resolve(row,model);F.validate_prediction_contract(row,c);theta=c['parameters']
    if set(theta)!=set(F.PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid offline services')
    fp32=c.get('component_fixed_native_fp32',{})
    if set(fp32)!={*F.FIX_COMPLETION_NAMES,*('writeback.native_fp32.'+n for n in F.FP32_FIELDS)}:raise ValueError('missing fixed FP32 services')
    if set(c.get('component_fixed_mte2',{}))!=set(F.D.PARAMETER_NAMES):raise ValueError('missing fixed MTE2 services')
    fixed={**F.FIXED_INTERVALS,**c['component_fixed_mte2'],**fp32}
    if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in fixed.items()):raise ValueError('fixed component service changed')
    values=tuple(theta[n] for n in F.PARAMETER_NAMES);graph=graph_type(F)(values)
    graph,sink,plan=F.compile_graph(row,graph=graph,**c['schedule']);value=graph.evaluate(values,sink);parts=graph.decompose(sink)
    if abs(sum(parts)-value)>1e-7*max(1,value):raise RuntimeError('static path accounting failed')
    return dict(predicted_us=value,contributions_us=dict(zip(GROUPS,parts)),fractions=dict(zip(GROUPS,(v/value for v in parts))),
        model_version=model['version'],active_cores=plan['active_cores'],launched_cores=plan['cores'],split_factor=plan['split_factor'],
        generated_events=len(graph.features),conversion_phase_us=graph.conversion_us,one_valid_model_path=True,not_PMU_or_real_engine_activity=True,
        timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
