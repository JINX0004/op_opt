"""Semantic replacement heads on computed B13; no target measurements."""

from math import exp, isfinite
from copy import deepcopy
from matmul_cost_model._engine.NN_source10_service_recalibrated_predict_v2876 import (
    Predictor as Parent,
)
from matmul_cost_model._engine.NN_source_window_basis_v2712 import transform, names
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn


def family(row):
    if row["implementation"] == "catlass":
        return "CAT_partial" if row["shape"][1] % 16 else "CAT_ordinary"
    p = row["plan"]
    return "native00" if not (p["nd2nz_a"] or p["nd2nz_b"]) else None


def apply_head(row, b, branch):
    assert isfinite(b) and b > 0
    spec = branch["static_tanh_basis"]
    head = branch["head"]
    assert tuple(head["feature_names"]) == names(spec["hidden"])
    x = transform(row, b, spec)
    c = head["implementation_corrections"][row["implementation"]]
    delta = c["weights"][0] + sum(
        w * (v - a) / s
        for w, v, a, s in zip(c["weights"][1:], x, c["center"], c["scale"])
    )
    cap = head["correction_cap"]
    delta = max(-cap, min(cap, delta))
    return b * exp(delta), delta


class Predictor:
    def __init__(self, model):
        self.model = deepcopy(model)
        self.parent = Parent(model["parent2875_model"])
        self.service = self.parent.service

    def predict(self, row):
        validate_nn(row)
        tag = family(row)
        if tag not in self.model["branches"]:
            return self.parent.predict(row)
        static = {
            k: row[k]
            for k in (
                "implementation",
                "shape",
                "cores",
                "dtype",
                "trans_a",
                "trans_b",
                "plan",
                "tile",
                "l1_tile",
                "l0_tile",
                "execution_profile",
                "transport",
                "source_a_offset_extra_bytes",
            )
            if k in row
        }
        b = self.service.predict(static)
        v, d = apply_head(static, b["predicted_us"], self.model["branches"][tag])
        return dict(
            predicted_us=v,
            computed_static_service13_us=b["predicted_us"],
            bounded_head_log_correction=d,
            semantic_head_family=tag,
            model_version=self.model["version"],
            target_measurements_required=False,
            hardware_execution_performed=False,
            independent_accuracy_qualified=False,
            training_only=True,
        )
