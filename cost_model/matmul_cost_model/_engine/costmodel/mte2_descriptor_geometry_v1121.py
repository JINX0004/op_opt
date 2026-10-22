"""Remove an ungrounded aligned-A source-pitch residual; keep ND geometry.

Equal-slot source controls at 32768/49152 element pitches do not support a
positive logarithmic pitch penalty on these aligned rows. The shared fixed
descriptor law already models row/byte/phase work. This is a development
hypothesis restricted to native ND A with 128B-aligned source row pitch.
Other residuals, transport protocols and unaligned source cases are retained.
"""
from .mte2_descriptor_geometry_v942 import *
from . import mte2_descriptor_geometry_v942 as B

def descriptor_features(g,operand,offset,extent):
    features,calls=B.descriptor_features(g,operand,offset,extent)
    if operand=='a' and g['kind']!='catlass' and not g['nza'] and (2*g['global_k'])%128==0:
        features=dict(features);features['dma.'+g['kind']+'.a_fill_stride_mb']=0.
    return features,calls
