"""One extra source-reviewed SC cache-hint branch; no timing inputs."""
from matmul_cost_model._engine.static_service_features_v1428 import features as old_features,NAMES as OLD_NAMES,correction
NAMES=OLD_NAMES+('SC_B_CACHE_MODE_DISABLE',)
def flag(row):
    if row['implementation']!='scsk':return 0
    p=row['plan'];t=p['tcube']
    return int(row['shape'][1]%128==0 and not (p['nd2nz_a'] or p['nd2nz_b']) and t['stepM']*t['baseM']==row['shape'][0])
def features(row,pipeline_us,model):return old_features(row,pipeline_us,model)+[float(flag(row))]
