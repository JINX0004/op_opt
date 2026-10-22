"""Full static NN pipeline plus bounded source-window residual, DEV candidate.

The public predictor accepts static operator metadata only. Its denominator
and baseline are calculated internally, never supplied as measured timing.
"""

from copy import deepcopy
from math import exp, isfinite
from matmul_cost_model._engine._imports import import_module
from matmul_cost_model._engine import matmul_group_periodic_v1874 as Group
from matmul_cost_model._engine.NN_thirteen_service_predict_v2578 import (
    Predictor as Service,
)
from matmul_cost_model._engine.NN_semantic_steady_predict_v2424 import physical
from matmul_cost_model._engine.NN_source_window_basis_v2712 import transform, names
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn


def enabled(row):
    return row["implementation"] == "catlass" or not row.get("plan", {}).get("nd2nz_a")


def apply_head(row, computed_service_us, model):
    base = float(computed_service_us)
    if not isfinite(base) or base <= 0:
        raise ValueError("positive computed static service prediction required")
    if not enabled(row):
        return base, 0.0
    spec = model["static_tanh_basis"]
    head = model["head"]
    assert tuple(head["feature_names"]) == names(spec["hidden"])
    x = transform(row, base, spec)
    normal = head["implementation_corrections"][row["implementation"]]
    w = normal["weights"]
    delta = w[0] + sum(
        a * (v - c) / s
        for a, v, c, s in zip(w[1:], x, normal["center"], normal["scale"])
    )
    cap = head["correction_cap"]
    delta = max(-cap, min(cap, delta))
    return base * exp(delta), delta


class Predictor:
    def __init__(self, model):
        self.model = deepcopy(model)
        service = self.model["service_model"]
        self.old = service["old_frozen_model"]
        self.service = Service(
            self.old, service["updated_parameters"], service["graph_version"]
        )
        assert (
            self.model["static_gate"]
            == "CAT or native00/native01; native10/native11 unchanged"
        )
        assert tuple(self.model["head"]["feature_names"]) == names(
            self.model["static_tanh_basis"]["hidden"]
        )
        for normal in self.model["head"]["implementation_corrections"].values():
            assert (
                len(normal["weights"])
                == len(normal["center"]) + 1
                == len(normal["scale"]) + 1
            )
            assert all(
                isfinite(v)
                for key in ("weights", "center", "scale")
                for v in normal[key]
            )
            assert all(v > 0 for v in normal["scale"])
        versions = physical(self.old)["baseline_model"]["baseline_model"][
            "baseline_bundle"
        ]["implementation_formula_versions"]
        for kind, version in versions.items():
            formula = import_module(
                "matmul_cost_model._engine.costmodel.matmul_four_path_v" + str(version)
            )
            (Group.GC if kind == "catlass" else Group.GN).module_for(formula)

    def predict(self, row):
        validate_nn(row)
        base = self.service.predict(row)
        value, delta = apply_head(row, base["predicted_us"], self.model)
        if not isfinite(value) or value <= 0:
            raise ValueError("positive finite static result required")
        return dict(
            predicted_us=value,
            static_service13_us=base["predicted_us"],
            pipeline_us=base["new_pipeline_us"],
            bounded_head_log_correction=delta,
            static_semantic_head_enabled=enabled(row),
            model_version=self.model["version"],
            target_measurements_required=False,
            hardware_execution_performed=False,
            independent_accuracy_qualified=False,
            training_only=True,
        )
