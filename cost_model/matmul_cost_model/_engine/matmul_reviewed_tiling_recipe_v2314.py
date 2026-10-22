"""Pure arithmetic reviewed tiling recipe, never a measured-shape table.
Arithmetic copied from pinned design.ops_plan; no path/environment setup.
"""
from matmul_cost_model._engine.costmodel.ops_native_base_v206 import WORDS
CAT={'default':([128,256,256],[128,256,64]),'deepk':([16,16,4096],[16,16,1024]),
     'widek':([16,32,2048],[16,32,512]),'midk':([32,64,1024],[32,64,256])}

def ceil(a,b):return (a+b-1)//b

def align(a,b):return ceil(a,b)*b

def ops_plan(shape,candidate):
    m,n,k=shape;name,kind,bm,bn,bk,s,c=candidate
    det=kind=='deterministic';sk=k if kind=='base' else min(k,bk*s)
    t=dict(usedCoreNum=c,M=m,N=n,Ka=k,Kb=k,singleCoreM=384 if det else bm,
        singleCoreN=n if det else bn,singleCoreK=sk,baseM=bm,baseN=bn,baseK=bk,
        stepM=3 if det else 1,stepN=1,stepKa=s,stepKb=s,
        depthA1=3*s if det else 2*s,depthB1=2*s,dbL0A=2,dbL0B=2,
        dbL0C=2 if det else 1,iterateOrder=1,shareUbSize=0)
    t['shareL1Size']=(t['depthA1']*bm+t['depthB1']*bn)*bk*2
    t['shareL0CSize']=bm*bn*4*t['dbL0C']
    assert t['shareL1Size']<=512*1024
    raw=[0]*70
    for f,i in WORDS.items():raw[i]=t[f]
    raw[22:26]=[1]*4
    raw[50:56]=[1,1,ceil(m,t['singleCoreM']),ceil(n,t['singleCoreN']),0,0]
    raw[56:64]=[0,0,int(k>65535),int(n%16!=0 or n>65535),0,0,0,0]
    raw[66:70]=[min(64,align(m,16)),min(512,align(k,16)),min(64,align(k,16)),min(512,align(n,16))]
    return dict(m=m,n=n,k=k,dtype='float16',trans_a=False,trans_b=False,tcube=t,
        raw_words=raw,nd2nz_a=raw[58],nd2nz_b=raw[59],block_dim=c)
