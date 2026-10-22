"""Same NN formula, computing frozen baseline and 112 static features once."""
from matmul_cost_model._engine.matmul_cpu_predict_cli_v1606 import validate_row
from matmul_cost_model._engine.matmul_static_CPU_v1877 import use_group_recurrence
from matmul_cost_model._engine.matmul_group_periodic_v1874 import pipeline as grouped_pipeline
from matmul_cost_model._engine.costmodel.matmul_static_path_trace_v1755 import predict as reference_pipeline
from matmul_cost_model._engine.static_core_path_features_v1773 import features as old_features
from matmul_cost_model._engine.static_serial_head_v1785 import serial_head,head_features
from matmul_cost_model._engine.matmul_hierarchical_residual_v1651 import common_features
from matmul_cost_model._engine.matmul_static_calibrated_v1797 import evaluate_static_features
from matmul_cost_model._engine.NN_relative_source_features_v2196 import extras
from matmul_cost_model._engine.NN_bounded_predict_v2203 import apply

def predict(row,model):
 validate_row(row);m,n,k=row['shape']
 if m not in range(16,129,16) or n not in (16,24,32,64,96,128) or not 64<=k<=131072:raise ValueError('outside reviewed NN FP16 research domain')
 frozen=model['frozen_base_model'];physical=frozen['baseline_model']
 base=(grouped_pipeline if use_group_recurrence(row) else reference_pipeline)(row,physical);p=base['predicted_us']
 x=old_features(row,p,physical,path=base)
 value=evaluate_static_features(row['implementation'],frozen,p,x,common_features(row),head_features(row),serial_head(row,physical))
 return apply(row['implementation'],value,x+extras(row),model)
