"""Experimental CPU recurrence with certified dominant-path group vectors.

All service constants and residual weights remain frozen. Unsupported or
uncertified repeated paths expand normally. No measurement inputs.
"""
from matmul_cost_model._engine import design
from types import ModuleType
from pathlib import Path
from collections import deque
from math import isfinite
from matmul_cost_model._engine.costmodel import matmul_streaming_v918 as S,matmul_periodic_base_v978 as OB,matmul_periodic_v978 as OP
from matmul_cost_model._engine.costmodel import matmul_CPU_periodic_v1188 as ON,matmul_CPU_catlass_periodic_v1595 as OC
from matmul_cost_model._engine.costmodel import matmul_static_path_trace_v1755 as T,matmul_source_phase_v1579 as Source

class GroupFeatures:
    def __init__(self,g):self.graph=g
    def __len__(self):return len(self.graph._finish)
    def append(self,features):
        g=self.graph;parents=tuple(g._pending);node=len(g._finish)
        parent=max(parents,key=g._finish.__getitem__,default=0)
        own=[0.]*8;cost=0.
        for j,v in features:
            part=g._theta[j]*v;cost+=part;own[g._group_index[j]]+=part
        g._finish.append(g._finish[parent]+cost)
        g._path.append(tuple(a+b for a,b in zip(g._path[parent],own)))
        g._pending=None;g._trace[node]=(parents,cost);g._trace_order.append(node);g._path_own[node]=tuple(own)
        if len(g._trace_order)>6000:
            old=g._trace_order.popleft();del g._trace[old];del g._path_own[old]

def certify_groups(g,first,previous,record,periods):
    ids=record['ids'];v=record['values'];pv=previous['values']
    now=record['group_values'];prev=previous['group_values'];before=first['group_values']
    delta=tuple(tuple(a-b for a,b in zip(x,y)) for x,y in zip(now,prev))
    old=tuple(tuple(a-b for a,b in zip(x,y)) for x,y in zip(prev,before))
    scale=max(1.,max(v));tol=1e-9*scale
    if max(abs(a-b) for x,y in zip(delta,old) for a,b in zip(x,y))>tol:return None
    scalar_delta=tuple(a-b for a,b in zip(v,pv));input_map={n:i for i,n in enumerate(previous['ids'])}
    start=previous['cut'];end=record['cut'];winners=None
    for q in (0,1,periods-1):
        values={n:v[i]+q*scalar_delta[i] for n,i in input_map.items()}
        paths={n:tuple(a+q*b for a,b in zip(now[i],delta[i])) for n,i in input_map.items()}
        values[0]=0.;paths[0]=(0.,)*8;values[g.head]=g._finish[g.head];paths[g.head]=g._path[g.head]
        selected=[]
        for node in range(start,end):
            if node not in g._trace or node not in g._path_own:return None
            parents,cost=g._trace[node]
            if any(p not in values for p in parents):return None
            parent=max(parents,key=values.__getitem__,default=0);selected.append(parent)
            values[node]=values[parent]+cost;paths[node]=tuple(a+b for a,b in zip(paths[parent],g._path_own[node]))
        if winners is not None and winners!=selected:return None
        winners=selected
        for i,n in enumerate(ids):
            if n not in paths:return None
            target=tuple(a+(q+1)*b for a,b in zip(now[i],delta[i]))
            if max(abs(a-b) for a,b in zip(paths[n],target))>tol:return None
    result={n:tuple(a+periods*b for a,b in zip(now[i],delta[i])) for i,n in enumerate(ids)}
    if any(x < -tol or not isfinite(x) for p in result.values() for x in p):return None
    return result

def clone(module,name,text,extra=None):
    q=ModuleType(name);q.__package__='matmul_cost_model._engine.costmodel';q.__file__=module.__file__
    if extra:q.__dict__.update(extra)
    exec(compile(text,name,'exec'),q.__dict__);return q

stream_text=Path(S.__file__).read_text(encoding='utf-8')
stream_text=stream_text.replace('return dict(predicted_us=value,','return dict(path_parts_us=graph._path[sink],predicted_us=value,')
GS=clone(S,'matmul_cost_model._engine.costmodel.private_group_stream1874',stream_text)
original_graph=GS.graph_type
def group_graph(formula):
    Parent=original_graph(formula)
    class Graph(Parent):
        def __init__(self,theta):
            super().__init__(theta)
            self._group_index=tuple(T.GROUPS.index(T.group(n)) for n in formula.PARAMETER_NAMES)
            self._path=[(0.,)*8];self._path_own={};self._trace={};self._trace_order=deque();self.features=GroupFeatures(self)
    return Graph
GS.graph_type=group_graph

def patch_macro(text,job=False):
    if job:
        anchor='record=dict(ji=next_job,ids=ids,values=values,cut=len(self._finish),inner=inner,seq=seq.copy())'
    else:anchor='record=dict(ko=ko_next,ids=ids,values=values,cut=len(self._finish))'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor[:-1]+',group_values=tuple(self._path[n] for n in ids))')
    anchor='  replacements={}';assert text.count(anchor)==1
    text=text.replace(anchor,"  group_future=_certify_groups(self,first,previous,record,periods)\n  if group_future is None:return "+('next_job' if job else 'ko_next')+",inner\n"+anchor)
    anchor='  for n,v in replacements.items():self._finish[n]=v';assert text.count(anchor)==1
    return text.replace(anchor,'  for n,v in replacements.items():self._finish[n]=v;self._path[n]=group_future[n]')

bt=Path(OB.__file__).read_text(encoding='utf-8')
bt=bt.replace('from . import matmul_four_path_v942 as F,matmul_streaming_v918 as Stream','from . import matmul_four_path_v942 as F\nStream=_GROUP_STREAM')
start=bt.index('class TraceFeatures(');end=bt.index('\nBaseStream=',start)
bt=bt[:start]+'TraceFeatures=_GROUP_FEATURES\n'+bt[end:]
GB=clone(OB,'matmul_cost_model._engine.costmodel.private_group_inner1874',patch_macro(bt),dict(_GROUP_STREAM=GS,_GROUP_FEATURES=GroupFeatures,_certify_groups=certify_groups))
pt=Path(OP.__file__).read_text(encoding='utf-8').replace('from . import matmul_periodic_base_v978 as B','B=_GROUP_B')
GP=clone(OP,'matmul_cost_model._engine.costmodel.private_group_jobs1874',patch_macro(pt,job=True),dict(_GROUP_B=GB,_certify_groups=certify_groups))
nt=Path(ON.__file__).read_text(encoding='utf-8')
anchor='from matmul_cost_model._engine.costmodel import matmul_periodic_v978 as P,matmul_periodic_base_v978 as B,matmul_streaming_v918 as Stream'
assert nt.count(anchor)==1;nt=nt.replace(anchor,'P=_GROUP_P;B=_GROUP_B;Stream=_GROUP_STREAM')
GN=clone(ON,'matmul_cost_model._engine.costmodel.private_group_native1874',nt,dict(_GROUP_P=GP,_GROUP_B=GB,_GROUP_STREAM=GS))
ct=Path(OC.__file__).read_text(encoding='utf-8')
anchor='from . import matmul_streaming_v918 as Stream,matmul_periodic_base_v978 as B'
assert ct.count(anchor)==1;ct=ct.replace(anchor,'B=_GROUP_B;Stream=_GROUP_STREAM')
GC=clone(OC,'matmul_cost_model._engine.costmodel.private_group_CAT1874',ct,dict(_GROUP_B=GB,_GROUP_STREAM=GS))

def pipeline(row,model):
    F,c=Source.resolve(row,model)
    v=(GC if row['implementation']=='catlass' else GN).predict(F,row,c)
    parts=v.pop('path_parts_us');p=v['predicted_us']
    if abs(sum(parts)-p)>1e-7*max(1,p):raise AssertionError('group recurrence accounting')
    return dict(v,contributions_us=dict(zip(T.GROUPS,parts)),fractions=dict(zip(T.GROUPS,(x/p for x in parts))),
        one_valid_model_path=True,not_PMU_or_real_engine_activity=True,experimental_group_CPU_recurrence=True)

def predict(row,model):
    from matmul_cost_model._engine.static_core_path_features_v1773 import features
    from matmul_cost_model._engine.static_serial_head_v1785 import serial_head,head_features
    from matmul_cost_model._engine.matmul_hierarchical_residual_v1651 import common_features
    from matmul_cost_model._engine.matmul_static_calibrated_v1797 import evaluate_static_features
    base=pipeline(row,model['baseline_model']);p=base['predicted_us']
    x=features(row,p,model['baseline_model'],path=base)
    value=evaluate_static_features(row['implementation'],model,p,x,common_features(row),head_features(row),serial_head(row,model['baseline_model']))
    return dict(base,pipeline_us=p,predicted_us=value,model_version=model['version'],model_parameters_unchanged=True,
        timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
