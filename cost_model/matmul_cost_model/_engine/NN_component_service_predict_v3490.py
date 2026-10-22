"""v3477 plus a <=2.5% static BASE64-byte-row calibration; no service refit."""

from math import exp, tanh
from matmul_cost_model._engine.NN_component_service_predict_v3477 import (
    Predictor as Parent,
    features,
    NAMES,
)
from matmul_cost_model._engine.NN_layout_service_repair_predict_v3453 import static_row
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan


def auxiliary(r):
    p = r.get("plan", {})
    return (
        r["implementation"] == "base"
        and r["shape"][1] == 32
        and bool(p.get("nd2nz_a"))
        and not p.get("nd2nz_b")
    )


class Predictor(Parent):
    def evaluate(self, row):
        r, dtype = static_row(row)
        if not auxiliary(r):
            return super().evaluate(row)
        old = self.prior.predict(r)
        path = self.prior.pipeline(r)
        x = features(r, compile_plan(r), path)
        head = self.model.get("BASE_N32_static_head")
        delta = 0.0
        if head:
            z = x
            if head.get("feature_transform") == "static_tanh":
                basis = [
                    (v - a) / s
                    for v, a, s in zip(x, head["basis_center"], head["basis_scale"])
                ]
                z = x + [
                    tanh(sum(w * v for w, v in zip(d, basis)))
                    for d in head["directions"]
                ]
            delta = head["weights"][0] + sum(
                w * (v - a) / s
                for w, v, a, s in zip(
                    head["weights"][1:], z, head["center"], head["scale"]
                )
            )
            delta = max(-head["log_cap"], min(head["log_cap"], delta))
        return (
            dict(
                old,
                predicted_us=old["predicted_us"] * exp(delta),
                prior_prediction_us=old["predicted_us"],
                prior_pipeline_us=path["predicted_us"],
                new_pipeline_us=path["predicted_us"],
                computed_static_budget_us=old["predicted_us"],
                input_dtype=dtype,
                model_version=self.model["version"],
                component_service_repair_applied=True,
                repair_family="base:N32_A_NZ_B_ND",
                static_features=x,
                bounded_log_correction=delta,
                physical_graph_unchanged=True,
                target_measurements_required=False,
                hardware_execution_performed=False,
            ),
            path,
        )
