"""717 plus a per-core aggregate L1 read/write service bound.

An offline shared-resource hypothesis motivated by699 background transport.
This is a lower-bound branch in the event graph, not a sum of PMU activities.
No claim is made about a uniquely identified bank map or arbitration policy.
"""
from math import isfinite
from . import matmul_four_path_v717 as base
from .matmul_four_path_v590 import descriptor_features,cd,r16
from .matmul_four_path_v320 import validate_prediction_contract
EXTRA_NAMES=('shared_l1.read_write_mb','shared_l1.transactions_k')
PARAMETER_NAMES=base.PARAMETER_NAMES+EXTRA_NAMES
INDEX={n:i for i,n in enumerate(PARAMETER_NAMES)}
FIXED_INTERVALS=base.FIXED_INTERVALS
compile_plan=base.compile_plan

def _service_budgets_expanded(row):
    program=compile_plan(row);budgets=[];seen=set()
    for jobs in program['work']:
        reads=writes=calls=0.
        for job in jobs:
            g=job['geometry'];m,n,k,bk=(g[x] for x in ('m','n','k','bk'))
            for ko in range(0,k,bk):
                kk=min(bk,k-ko);reads+=2*(r16(m)+r16(n))*r16(kk)
                calls+=cd(m,16)+cd(kk,16) if row['implementation']=='catlass' else 2
            for operand in ('a','b'):
                span=bk*g['s'+operand];width=r16(m if operand=='a' else n)
                for offset in range(0,k,span):
                    extent=min(span,k-offset);writes+=2*width*r16(extent)
                    _,count=descriptor_features(g,operand,offset,extent);calls+=count
        signature=(reads+writes,calls)
        if signature not in seen and jobs:
            seen.add(signature);budgets.append(dict(zip(EXTRA_NAMES,((reads+writes)/1e6,calls/1000))))
    return budgets


def _descriptor_call_count(g,operand,extent):
    rows=g['m'] if operand=='a' else extent
    calls=rows if g['kind']=='catlass' and operand=='a' and g['global_k']>=65536 else 1
    nz=g['nza'] if operand=='a' else g['nzb']
    if nz and operand=='b' and r16(g['global_k'])-r16(rows)>=65535:calls=cd(g['n'],16)
    return calls

def service_budgets(row):
    # Every accumulated quantity is integral. Up to2**53 the original floating
    # additions equal these integer sums exactly. Fall back beyond that bound.
    program=compile_plan(row);budgets=[];seen=set()
    for jobs in program['work']:
        reads=writes=calls=0
        for job in jobs:
            g=job['geometry'];m,n,k,bk=(g[x] for x in ('m','n','k','bk'))
            full,tail=divmod(k,bk);steps=full+int(tail>0)
            reads+=2*(r16(m)+r16(n))*(full*r16(bk)+(r16(tail) if tail else 0))
            if row['implementation']=='catlass':calls+=steps*cd(m,16)+full*cd(bk,16)+(cd(tail,16) if tail else 0)
            else:calls+=2*steps
            for operand in ('a','b'):
                span=bk*g['s'+operand];width=r16(m if operand=='a' else n);packets,last=divmod(k,span)
                writes+=2*width*(packets*r16(span)+(r16(last) if last else 0))
                calls+=packets*_descriptor_call_count(g,operand,span)
                if last:calls+=_descriptor_call_count(g,operand,last)
        if reads+writes>=2**53 or calls>=2**53:return _service_budgets_expanded(row)
        signature=(reads+writes,calls)
        if signature not in seen and jobs:
            seen.add(signature);budgets.append(dict(zip(EXTRA_NAMES,((reads+writes)/1e6,calls/1000))))
    return budgets

class EventGraph(base.EventGraph):
    def __init__(self):super().__init__();self.l1_budgets=[];self.shared_branches=0
    def plain_node(self,predecessors=(),features=None,fraction=1.):
        self.predecessors.append(sorted(set(predecessors)-{0}))
        self.features.append([(INDEX[k],float(v)*fraction) for k,v in (features or {}).items() if v])
        return len(self.features)-1
    def node(self,predecessors=(),features=None,fraction=1.):
        body_join=self.startup_node and not features and self.startup_node in predecessors
        final_phase=features and any(k.startswith('entry.') for k in features)
        if (body_join or final_phase) and not self.shared_branches:
            bounds=[self.plain_node(features=b) for b in self.l1_budgets];self.shared_branches+=len(bounds)
            predecessors=[*predecessors,*bounds]
        return super().node(predecessors,features,fraction)

def compile_graph(row,graph=None,**schedule):
    validate_prediction_contract(row);graph=graph or EventGraph()
    if not isinstance(graph,EventGraph):raise ValueError('740 event graph required')
    graph.l1_budgets=service_budgets(row)
    result=base.compile_graph(row,graph,**schedule)
    if graph.shared_branches!=len(graph.l1_budgets):raise RuntimeError('shared L1 bounds not attached to kernel body')
    return result

def predict(row,model):
    validate_prediction_contract(row,model);theta=model['parameters']
    if set(theta)!=set(PARAMETER_NAMES) or any(not isfinite(v) or v<0 for v in theta.values()):raise ValueError('invalid offline services')
    if any(abs(theta[n]-v)>1e-12*max(1,v) for n,v in FIXED_INTERVALS.items()):raise ValueError('fixed component interval changed')
    graph,sink,p=compile_graph(row,**model.get('schedule',{}));value=graph.evaluate([theta[n] for n in PARAMETER_NAMES],sink)
    return dict(predicted_us=value,model_version=model.get('version','four_path_v740'),active_cores=p['active_cores'],launched_cores=p['cores'],split_factor=p['split_factor'],generated_events=len(graph.features),shared_l1_aggregate_bound_hypothesis=True,timing_input_required=False,hardware_execution_performed=False,accuracy_qualified=False,stable_ranking_qualified=False)
