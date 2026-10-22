"""Static column-packet/row-address features;effective service hypothesis only.

For a128/256B column window, combine32B sectors of all output rows within that
window. Windows do not combine with one another. This reflects a bounded format
traversal hypothesis, not a published hardware packet size or cache specification.
"""
from functools import lru_cache

@lru_cache(maxsize=4096)
def packet_features(m,n,ldc,offset_bytes,element_bytes=4):
    if (any(type(v)!=int for v in (m,n,ldc,offset_bytes,element_bytes)) or
        not 0<m<=128 or not 0<n<=128 or ldc<n or offset_bytes<0 or offset_bytes%32 or
        element_bytes not in (2,4) or n*element_bytes%32):
        raise ValueError('positive reviewed32B-sector output required')
    values={}
    for grain in (128,256):
        partial=partial_bytes=full=0;logical=0
        for nc in range(0,n,grain//element_bytes):
            width=min(grain//element_bytes,n-nc)*element_bytes;lines={}
            for row in range(m):
                begin=offset_bytes+(row*ldc+nc)*element_bytes
                assert begin%32==0 and width%32==0
                for off in range(0,width,32):
                    line,within=divmod(begin+off,grain)
                    lines[line]=lines.get(line,0)|(1<<(within//32));logical+=32
            for mask in lines.values():
                if mask==(1<<(grain//32))-1:full+=1
                else:partial+=1;partial_bytes+=mask.bit_count()*32
        assert logical==m*n*element_bytes
        values[f'partial{grain}_k']=partial/1000
        values[f'partial{grain}_mb']=partial_bytes/1e6
        values[f'full{grain}_k']=full/1000
    values['row_phase256_mb']=sum(n*element_bytes/1e6 for row in range(m)
        if (offset_bytes+row*ldc*element_bytes)%512==256)
    values['logical_mb']=m*n*element_bytes/1e6
    return values
