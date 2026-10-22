"""Thirteen existing semantic service constants; stdlib-only CPU inference.
Research hypothesis, no accuracy qualification. Non01 remains exactly frozen.
"""

from copy import deepcopy
from math import isfinite
from matmul_cost_model._engine.NN_semantic_steady_predict_v2424 import (
    physical,
    semantics,
    old_value,
)
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn
from matmul_cost_model._engine import NN_bounded_onepass_predict_v2221 as Old

FIELDS = {
    "base": (
        "entry.base",
        "dma.base.a_dst_density",
        "scalar.outer_kloops.base",
        "scalar.inner_kloops.base",
        "scalar.base_calls",
    ),
    "scsk": ("entry.scsk", "scalar.sc_calls", "scalar.inner_kloops.scsk"),
    "deterministic": (
        "entry.deterministic",
        "dma.deterministic.a_dst_density",
        "scalar.det_calls",
        "scalar.outer_kloops.deterministic",
        "scalar.inner_kloops.deterministic",
    ),
}


def initial_parameters(model):
    s = semantics(physical(model))
    return {
        kind: {name: float(s[kind + ":01"][name]) for name in names}
        for kind, names in FIELDS.items()
    }


class Predictor:
    def __init__(self, old_model, parameters=None, graph_version=942):
        if graph_version not in (942, 1064):
            raise ValueError("reviewed source graph942/1064 required")
        self.old = deepcopy(old_model)
        self.initial = initial_parameters(self.old)
        self.parameters = deepcopy(self.initial if parameters is None else parameters)
        self.graph_version = graph_version
        if set(self.parameters) != set(FIELDS):
            raise ValueError("three native semantic groups required")
        for kind, names in FIELDS.items():
            values = self.parameters[kind]
            if set(values) != set(names) or any(
                not isfinite(v) or v < 0 for v in values.values()
            ):
                raise ValueError(
                    "thirteen finite nonnegative offline constants required"
                )
        self.updated_physical = deepcopy(physical(self.old))
        s = semantics(self.updated_physical)
        for kind, v in self.parameters.items():
            s[kind + ":01"].update(v)
        bundle = self.updated_physical["baseline_model"]["baseline_model"][
            "baseline_bundle"
        ]
        for kind in FIELDS:
            if bundle["implementation_formula_versions"][kind] != 942:
                raise ValueError("original942 baseline required")
            bundle["implementation_formula_versions"][kind] = graph_version

    def predict(self, row):
        validate_nn(row)
        p0, t0 = old_value(row, self.old)
        kind = row["implementation"]
        plan = row.get("plan", {})
        enabled = (
            kind in FIELDS and not plan.get("nd2nz_a") and bool(plan.get("nd2nz_b"))
        )
        changed = enabled and (
            self.graph_version != 942 or self.parameters[kind] != self.initial[kind]
        )
        if changed:
            p1 = (
                Old.grouped_pipeline
                if Old.use_group_recurrence(row)
                else Old.reference_pipeline
            )(row, self.updated_physical)["predicted_us"]
            t1 = t0 * (p1 / p0)
        else:
            p1 = p0
            t1 = t0
        if not all(isfinite(x) and x > 0 for x in [p0, p1, t0, t1]):
            raise ValueError("positive finite static result required")
        return dict(
            predicted_us=t1,
            old_pipeline_us=p0,
            new_pipeline_us=p1,
            old_frozen_prediction_us=t0,
            semantic01_updated=changed,
            graph_version=self.graph_version if enabled else 942,
            target_measurements_required=False,
            hardware_execution_performed=False,
            accuracy_qualified=False,
            static_service_parameters_only=True,
            old_correction_independent_of_updated_graph=True,
        )
