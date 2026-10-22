"""Component-constrained native source repair and bounded DEV static head.

Frozen predecessor remains intact. Inference reads static geometry only;
shared source service is a resource bound, never a repeated peer-copy fee.
"""

from copy import deepcopy
from math import log2, exp, isfinite
from matmul_cost_model._engine.NN_layout_service_repair_predict_v3453 import (
    Predictor as Prior,
    static_row,
    aggregate_work,
    shared_cost,
    source_cohorts,
)
from matmul_cost_model._engine.NN_semantic_steady_predict_v2424 import semantics
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan
from matmul_cost_model._engine import NN_bounded_onepass_predict_v2221 as Old

NAMES = (
    "log_M",
    "log_N",
    "log_K",
    "log_baseM",
    "log_baseN",
    "log_step",
    "log_launched",
    "log_active",
    "log_jobs",
    "log_mmad",
    "log_descriptors",
    "log_cohort",
    "log_aggregate",
    "log_unique",
    "unique_fraction",
    "valid_M_fraction",
    "valid_N_fraction",
    "K_packet_tail",
    "K_task_tail",
    "conversion_fraction",
    "L1_A_slots",
    "L1_B_slots",
    "L0_slots",
    "source_work_per_active",
    "active_fraction",
    "partial_N_fraction",
)


def family(r):
    if r["implementation"] not in ("base", "deterministic"):
        return None
    p = r["plan"]
    code = str(int(p["nd2nz_a"])) + str(int(p["nd2nz_b"]))
    if r["implementation"] == "base" and r["shape"][1] == 24:
        return "base:" + code
    if r["implementation"] == "deterministic" and p["nd2nz_b"]:
        return "deterministic:" + code
    return None


def features(r, program, path):
    m, n, k = r["shape"]
    t = r["plan"]["tcube"]
    a, u = aggregate_work(program)
    cohort, desc = source_cohorts(program)
    active = program["active_cores"]
    jobs = max(map(len, program["work"]))
    mmads = max(
        sum((j["geometry"]["k"] + t["baseK"] - 1) // t["baseK"] for j in js)
        for js in program["work"]
    )
    span = t["baseK"] * t["stepKa"]
    sk = t["singleCoreK"]
    return [
        log2(1 + x)
        for x in (
            m,
            n,
            k,
            t["baseM"],
            t["baseN"],
            t["stepKa"],
            r["cores"],
            active,
            jobs,
            mmads,
            desc,
            cohort,
            a,
            u,
        )
    ] + [
        u / a if a else 0.0,
        min(m, t["baseM"]) / t["baseM"],
        min(n, t["baseN"]) / t["baseN"],
        (k % span) / span,
        (k % sk) / sk,
        path.get("conversion_phase_us", 0) / path["predicted_us"],
        t["depthA1"] / t["stepKa"],
        t["depthB1"] / t["stepKb"],
        min(t["dbL0A"], t["dbL0B"]),
        a / max(active, 1),
        active / r["cores"],
        (n % 16) / 16,
    ]


class Predictor:
    def __init__(self, model):
        self.model = deepcopy(model)
        self.prior = Prior(model["prior_v3457"])
        self.variants = {}
        self.dynamic = {}
        policy = self.model["source_policy"]
        for kind in ("base", "deterministic"):
            for code in ("00", "01", "10", "11"):
                original = self.prior.originals["01" if code == "01" else "11"]
                variant = deepcopy(original)
                rates = semantics(variant).get(kind + ":" + code)
                if rates is None:
                    continue
                if code == "11":
                    rates["dma." + kind + ".peer_read_mb"] = 0.0
                    rates["dma." + kind + ".peer_copy_calls"] = 0.0
                if code == "01":
                    rates["dma." + kind + ".a_source_excess_klines"] = policy[
                        "A_partial_row_extra_us_per_kline"
                    ]
                rates.update(
                    model.get("fixed_local_service_overrides", {}).get(
                        kind + ":" + code, {}
                    )
                )
                bundle = variant["baseline_model"]["baseline_model"]["baseline_bundle"]
                for c in bundle["implementation_models"].values():
                    weights = c["schedule"]["conversion_constants"]
                    for name in weights:
                        if name.startswith(("mte2.", "mte3.", "vector.", "scalar.")):
                            weights[name] *= policy["conversion_engine_scale"]
                    weights["phase.entry"] = policy["conversion_phase_entry_us"]
                    weights["phase.extra_cores"] = policy["conversion_extra_core_us"]
                    weights["phase.log2_cores"] = policy["conversion_log2_cores_us"]
                self.variants[kind + ":" + code] = variant

    def evaluate(self, row):
        r, dtype = static_row(row)
        key = family(r)
        old = self.prior.predict(r)
        if key is None:
            return (
                dict(
                    old,
                    input_dtype=dtype,
                    component_service_repair_applied=False,
                    model_version=self.model["version"],
                ),
                None,
            )
        program = compile_plan(r)
        a, u = aggregate_work(program)
        variant = self.variants[key]
        if key.endswith("11"):
            bound = shared_cost(program, self.model["source_policy"], a, u)
            scale = bound / program["source_mb"] if program["source_mb"] else 0.0
            tag = (key, round(scale, 14))
            if tag not in self.dynamic:
                changed = deepcopy(variant)
                changed["baseline_model"]["baseline_model"]["baseline_bundle"][
                    "implementation_models"
                ][r["implementation"]]["parameters"]["shared.mb"] = scale
                if len(self.dynamic) > 128:
                    self.dynamic.clear()
                self.dynamic[tag] = changed
            variant = self.dynamic[tag]
        path = (
            Old.grouped_pipeline
            if Old.use_group_recurrence(r)
            else Old.reference_pipeline
        )(r, variant)
        # Keep the previous fixed residual ratio distinct from physical services.
        oldpath = Prior.pipeline(self.prior, r)["predicted_us"]
        budget = old["predicted_us"] * path["predicted_us"] / oldpath
        x = features(r, program, path)
        head = self.model.get("heads", {}).get(key)
        delta = 0.0
        if head:
            if head.get("feature_transform") == "static_tanh":
                z = [
                    sum(
                        w * v
                        for w, v in zip(
                            direction,
                            [
                                (v - a) / s
                                for v, a, s in zip(
                                    x, head["basis_center"], head["basis_scale"]
                                )
                            ],
                        )
                    )
                    for direction in head["directions"]
                ]
                from math import tanh

                x = x + [tanh(v) for v in z]
            assert head["feature_names"] == [
                "bias",
                *NAMES,
                *(
                    ["fixed_tanh_" + str(i) for i in range(len(head["directions"]))]
                    if head.get("feature_transform") == "static_tanh"
                    else []
                ),
            ]
            delta = head["weights"][0] + sum(
                w * (v - a) / s
                for w, v, a, s in zip(
                    head["weights"][1:], x, head["center"], head["scale"]
                )
            )
            delta = max(-head["log_cap"], min(head["log_cap"], delta))
        value = budget * exp(delta)
        if not isfinite(value) or value <= 0:
            raise ValueError("positive fixed static prediction required")
        return (
            dict(
                predicted_us=value,
                input_dtype=dtype,
                model_version=self.model["version"],
                component_service_repair_applied=True,
                repair_family=key,
                prior_prediction_us=old["predicted_us"],
                prior_pipeline_us=oldpath,
                new_pipeline_us=path["predicted_us"],
                computed_static_budget_us=budget,
                static_features=features(r, program, path),
                bounded_log_correction=delta,
                conversion_phase_us=path.get("conversion_phase_us", 0),
                active_cores=program["active_cores"],
                target_measurements_required=False,
                hardware_execution_performed=False,
                independent_accuracy_qualified=False,
            ),
            path,
        )

    def predict(self, row):
        return self.evaluate(row)[0]

    def pipeline(self, row):
        value, path = self.evaluate(row)
        return path if path is not None else self.prior.pipeline(row)
