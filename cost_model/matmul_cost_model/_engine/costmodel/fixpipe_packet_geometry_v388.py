"""Descriptor prefix,full/tail column windows and static address features.

Effective format/transport hypotheses,not asserted hardware transactions.
All geometry is static;no target measurement or shape-specific coefficient.
"""
from functools import lru_cache
from .fixpipe_packet_geometry_v386 import packet_features as previous_features

@lru_cache(maxsize=4096)
def packet_features(m,n,ldc,offset_bytes,element_bytes=4):
    f=dict(previous_features(m,n,ldc,offset_bytes,element_bytes))
    counts={kind:0 for kind in ('fullwin','trailing','shortwin')}
    grain=256
    for nc in range(0,n,grain//element_bytes):
        width=min(grain//element_bytes,n-nc)*element_bytes
        kind='fullwin' if width==grain else 'trailing' if nc else 'shortwin';lines={}
        for row in range(m):
            begin=offset_bytes+(row*ldc+nc)*element_bytes
            for off in range(0,width,32):
                line,within=divmod(begin+off,grain);lines[line]=lines.get(line,0)|(1<<(within//32))
        counts[kind]+=sum(mask.bit_count()*32 for mask in lines.values() if mask!=(1<<(grain//32))-1)
    for kind,value in counts.items():f[kind+'_partial256_mb']=value/1e6
    f['packed_start64']=int(ldc==n and offset_bytes%128!=0)
    f['packed_start128']=int(ldc==n and offset_bytes%256==128)
    f['row_gaps']=int(ldc>n)
    return f
