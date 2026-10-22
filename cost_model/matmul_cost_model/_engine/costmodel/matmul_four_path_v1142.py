"""BASE event head competes with AIC init; entry is not counted twice.

The component head includes its common envelope. It replaces the old BASE
conversion root, retaining max(init,head) and the exact original source body.
No full-kernel target measurement is required at inference.
"""
from .matmul_four_path_v942 import *
from . import matmul_four_path_v942 as B
from . import matmul_event_head_v1102 as Head

class EventGraph(B.EventGraph):
    pass

def compile_graph(row,graph=None,*,conversion_constants,event_head_constants,**schedule):
    validate_prediction_contract(row);graph=graph or EventGraph()
    if not isinstance(graph,EventGraph):raise ValueError('1142 event graph required')
    if set(conversion_constants)!=set(C.PARAMETER_NAMES):raise ValueError('complete frozen conversion constants required')
    kind=row['implementation'];converted=kind!='catlass' and bool(row['plan']['nd2nz_a'] or row['plan']['nd2nz_b'])
    graph.initialization_kind=kind;graph.initialization_converted=converted
    graph.conversion_us=Head.predict(row,event_head_constants)['head_event_us'] if kind=='base' and converted else 0. if kind=='catlass' else B.conversion_us(*row['shape'],row['cores'],bool(row['plan']['nd2nz_a']),bool(row['plan']['nd2nz_b']),tuple(conversion_constants[n] for n in C.PARAMETER_NAMES))
    init=graph.plain_node(features={'entry.'+kind:1,'launch.'+kind:row['cores']})
    conv=graph.plain_node(features={'conversion.frozen_pipeline_us':graph.conversion_us})
    graph.head=graph.plain_node([init,conv])
    graph.l1_budgets=L.service_budgets(row);graph.native_load3d=kind!='catlass';graph.load3d_ii=schedule.pop('load3d_ii',.7)
    if graph.load3d_ii!=.7:raise ValueError('fixed Load3D interval implementation required')
    contexts=B._cast_contexts(row);graph.cast_contexts=iter(contexts)
    result=Schedule.compile_graph(row,graph,**schedule)
    if graph.cast_count!=len(contexts) or not graph.phase_attached or graph.shared_branches!=len(graph.l1_budgets):raise RuntimeError('source attachment mismatch')
    return result

def predict(row,model):
    from . import matmul_streaming_v918 as Stream
    return Stream.predict(__import__(__name__,fromlist=['EventGraph']),row,model)
