"""100 static geometry/path inputs, no observations or identifiers."""
from matmul_cost_model._engine.static_path_features_v1757 import features as prior_features,NAMES as PRIOR_NAMES,correction
from matmul_cost_model._engine.static_core_geometry_v1772 import extra_features,EXTRA_NAMES
NAMES=PRIOR_NAMES+EXTRA_NAMES
def features(row,pipeline_us,model,path=None):return prior_features(row,pipeline_us,model,path=path)+extra_features(row)
