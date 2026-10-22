"""ND descriptor service work with measured large-packet and128B-edge controls.

Thresholds are static effective packetization proxies, not claimed hardware
cache capacities. No implementation, A/B role, case ID or timing is inspected.
"""
from math import gcd,ceil
from . import nd_descriptor_work_v897 as Old
II_NAMES=Old.II_NAMES+('dense_above32k_mb','dense_above128k_mb','row_cross128_krows')
EXTRA_NAMES=Old.EXTRA_NAMES+('dense_fill8k',)

def crossing_rows(rows,columns,pitch,phase=0):
 width=2*columns;stride=2*pitch;period=128//gcd(128,stride);q,r=divmod(rows,period)
 extra=[max(0,ceil(((phase+j*stride)%128+width)/128)-ceil(width/128)) for j in range(period)]
 return q*sum(extra)+sum(extra[:r])

def work(rows,columns,pitch,phase=0):
 old=Old.work(rows,columns,pitch,phase);ii=[dict(b) for b in old['interval']];extra=[dict(b) for b in old['extra_completion']];mb=2*rows*columns/1e6
 ii[0].update(dense_above32k_mb=max(0.,mb-32768/1e6),dense_above128k_mb=max(0.,mb-131072/1e6))
 crossing=crossing_rows(rows,columns,pitch,phase)/1000
 for b in ii:b['row_cross128_krows']=crossing
 extra[0]['dense_fill8k']=min(1.,2*rows*columns/8192)
 return dict(old,interval=ii,extra_completion=extra,row_cross128_krows=crossing)

