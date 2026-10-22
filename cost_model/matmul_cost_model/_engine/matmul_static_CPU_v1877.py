"""Static repetition-capacity dispatch; identical frozen coefficients."""
from math import lcm
from matmul_cost_model._engine import design
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import catlass_split_factor
from matmul_cost_model._engine import matmul_static_calibrated_v1797 as Reference
from matmul_cost_model._engine import matmul_group_periodic_v1874 as Group

def use_group_recurrence(row):
    m,n,k=row['shape'];kind=row['implementation']
    if kind=='catlass':
        l1=row['l1_tile'];split=catlass_split_factor(m,n,k,l1,row['cores'])
        packets=(k+l1[2]-1)//l1[2]
        return (packets+split-1)//split>=24
    t=row['plan']['tcube'];bk=t['baseK'];sk=t['singleCoreK']
    if kind=='base':
        period=lcm(bk*t['stepKa']*(t['depthA1']//t['stepKa']),bk*t['stepKb']*(t['depthB1']//t['stepKb']),bk*min(t['dbL0A'],t['dbL0B']),256)
        return k>=14*period
    parts=(k+sk-1)//sk
    if kind=='scsk':return parts>=32
    return (parts+row['cores']-1)//row['cores']>=64

def predict(row,model):
    use=use_group_recurrence(row)
    v=(Group.predict if use else Reference.predict)(row,model)
    return dict(v,CPU_engine1877='group_periodic' if use else 'reference',CPU_engine_selected_from_static_geometry_only=True)
