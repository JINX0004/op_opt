"""Choose an arithmetic engine from static repetition capacity, never timings."""
from math import lcm
def periodic_useful(row):
 kind=row['implementation']
 if kind=='catlass':return False
 t=row['plan']['tcube'];k=row['shape'][2];bk=t['baseK'];sk=t['singleCoreK']
 if min(bk,sk,t['stepKa'],t['stepKb'],t['depthA1'],t['depthB1'],t['dbL0A'],t['dbL0B'])<=0:return False
 if kind=='base':
  a=bk*t['stepKa'];b=bk*t['stepKb'];slots_a=t['depthA1']//t['stepKa'];slots_b=t['depthB1']//t['stepKb'];slots_l0=min(t['dbL0A'],t['dbL0B'])
  if min(slots_a,slots_b)<=0:return False
  period=lcm(a*slots_a,b*slots_b,bk*slots_l0,256)
  return k>=7*period
 partitions=(k+sk-1)//sk
 if kind=='scsk':return partitions>=16
 if kind=='deterministic':return (partitions+row['cores']-1)//row['cores']>=44
 return False
