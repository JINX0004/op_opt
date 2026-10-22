"""Static FP16-output sector/window features;not a physical cache specification."""
from functools import lru_cache

@lru_cache(maxsize=4096)
def packet_features(m,n,ldc,offset_bytes):
    if min(m,n)<=0 or max(m,n)>128 or ldc<n or offset_bytes%16 or n%8:raise ValueError('reviewed16B-sector FP16 output required')
    f={'logical_mb':2*m*n/1e6,'row_gaps':int(ldc>n),
        'row_phase256_mb':sum(2*n/1e6 for r in range(m) if (offset_bytes+2*r*ldc)%512==256),
        'packed_start64':int(ldc==n and offset_bytes%128!=0),
        'packed_start128':int(ldc==n and offset_bytes%256==128)}
    for grain in (128,256):
        count=size=0;categories={k:0 for k in ('fullwin','trailing','shortwin')}
        for nc in range(0,n,grain//2):
            width=2*min(grain//2,n-nc);lines={};kind='fullwin' if width==grain else 'trailing' if nc else 'shortwin'
            for r in range(m):
                begin=offset_bytes+2*(r*ldc+nc)
                for off in range(0,width,16):
                    line,within=divmod(begin+off,grain);lines[line]=lines.get(line,0)|(1<<(within//16))
            for mask in lines.values():
                if mask!=(1<<(grain//16))-1:
                    count+=1;size+=mask.bit_count()*16;categories[kind]+=mask.bit_count()*16
        f[f'partial{grain}_k']=count/1000;f[f'partial{grain}_mb']=size/1e6
        if grain==256:
            for kind,value in categories.items():f[kind+'_partial256_mb']=value/1e6
    return f
