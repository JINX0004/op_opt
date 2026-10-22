"""Static NN source/tail/conversion repair. No labels or hardware in inference.

Original model is immutable. Recompute the finite-buffer event graph after
service changes, retain the frozen static residual ratio, then optionally apply
a bounded offline head. NZ competition is an aggregate resource bound, not a
per-copy fee repeated for every peer. All quantities are derived from static X.
"""

from copy import deepcopy
from math import log2, exp, isfinite
from matmul_cost_model._engine.NN_complete_semantic_heads_predict_v2908 import (
    Predictor as Parent,
)
from matmul_cost_model._engine.NN_semantic_steady_predict_v2424 import (
    physical,
    semantics,
)
from matmul_cost_model._engine import NN_bounded_onepass_predict_v2221 as Old
from matmul_cost_model._engine.costmodel.matmul_four_path_v590 import compile_plan, r16
from matmul_cost_model._engine.matmul_static_unified_v2342 import validate_nn

FEATURE_NAMES = (
    "bias",
    "log_M",
    "log_K",
    "log_baseM",
    "log_baseN",
    "log_step",
    "log_launched",
    "log_active",
    "log_jobs",
    "tail_fraction",
    "conversion_fraction",
    "log_source_per_core",
    "log_source_aggregate",
)


def head_features(r, budget, x, head):
    if head.get("feature_transform") == "source_window_plus_static":
        from matmul_cost_model._engine.NN_source_window_basis_v2712 import (
            transform,
            names,
        )

        z = transform(r, budget, head["static_tanh_basis"])
        assert head["feature_names"] == [
            "bias",
            *names(head["static_tanh_basis"]["hidden"]),
            *FEATURE_NAMES[1:],
        ]
        return [1.0, *z, *x[1:]]
    assert head["feature_names"] == list(FEATURE_NAMES)
    return x


def static_row(row):
    r = deepcopy(
        {
            k: row[k]
            for k in (
                "implementation",
                "shape",
                "cores",
                "dtype",
                "trans_a",
                "trans_b",
                "batch",
                "plan",
                "tile",
                "l1_tile",
                "l0_tile",
                "execution_profile",
                "transport",
                "source_a_offset_extra_bytes",
                "source_b_offset_extra_bytes",
            )
            if k in row
        }
    )
    dtype = r.get("dtype", "float16")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("Reviewed NN FP16/BF16 only")
    if "plan" in r and r["plan"].get("dtype", dtype) != dtype:
        raise ValueError("Contradictory input/plan dtype")
    r["dtype"] = "float16"
    if "plan" in r:
        r["plan"]["dtype"] = "float16"
    validate_nn(r)
    return r, dtype


def family(r):
    if r["implementation"] not in ("base", "scsk"):
        return None
    p = r["plan"]
    if not p["nd2nz_b"] or r["shape"][1] % 16 == 0:
        return None
    return r["implementation"] + ":" + ("11" if p["nd2nz_a"] else "01")


def auxiliary_family(r):
    p = r.get("plan", {})
    return (
        "base:10_B64"
        if r["implementation"] == "base"
        and p.get("nd2nz_a")
        and not p.get("nd2nz_b")
        and 2 * r["shape"][1] == 64
        else None
    )


def aggregate_work(program):
    aggregate = 0.0
    unique = {}
    for jobs in program["work"]:
        for j in jobs:
            g = j["geometry"]
            m, n, k, mo, no, ko = (g[k] for k in ("m", "n", "k", "mo", "no", "ko"))
            for key, amount in [
                (("a", mo, ko, m, k), 2 * r16(m) * r16(k)),
                (("b", no, ko, n, k), 2 * r16(n) * r16(k)),
            ]:
                aggregate += amount
                unique[key] = amount
    return aggregate / 1e6, sum(unique.values()) / 1e6


def source_cohorts(program):
    """Static per-descriptor source identity; repeated reads within one core
    do not count as concurrent peers. Count real NZ B split descriptors."""
    streams = []
    readers = {}
    for ci, jobs in enumerate(program["work"]):
        requests = []
        for j in jobs:
            g = j["geometry"]
            for operand in ("a", "b"):
                span = g["bk"] * g["s" + operand]
                for offset in range(0, g["k"], span):
                    extent = min(span, g["k"] - offset)
                    split = operand == "b" and r16(g["global_k"]) - r16(extent) >= 65535
                    for part in range((g["n"] + 15) // 16 if split else 1):
                        key = (
                            operand,
                            g["mo"] if operand == "a" else g["no"] + 16 * part,
                            (
                                g["m"]
                                if operand == "a"
                                else min(16, g["n"] - 16 * part) if split else g["n"]
                            ),
                            g["ko"] + offset,
                            extent,
                        )
                        requests.append(key)
                        readers.setdefault(key, set()).add(ci)
        streams.append(requests)
    return max(
        (sum(len(readers[k]) for k in stream) for stream in streams), default=0
    ), max(map(len, streams), default=0)


def shared_cost(program, policy, aggregate, unique):
    cohort, dispatch = source_cohorts(program)
    return max(
        policy.get("NZ_same_address_issue_us_per_descriptor", 0.0) * cohort,
        policy.get("NZ_shared_issue_us_per_descriptor", 0.0) * dispatch
        + policy["NZ_shared_aggregate_us_per_MB"] * aggregate
        + policy["NZ_shared_unique_us_per_MB"] * unique,
    )


def features(r, program, conversion, budget):
    t = r["plan"]["tcube"]
    m, n, k = r["shape"]
    active = program["active_cores"]
    aggregate, _ = aggregate_work(program)
    jobs = max(map(len, program["work"]))
    span = t["baseK"] * t["stepKa"]
    return [
        1.0,
        log2(1 + m),
        log2(1 + k),
        log2(1 + t["baseM"]),
        log2(1 + t["baseN"]),
        log2(1 + t["stepKa"]),
        log2(1 + r["cores"]),
        log2(1 + active),
        log2(1 + jobs),
        (k % span) / span,
        conversion / budget,
        log2(1 + aggregate / max(active, 1)),
        log2(1 + aggregate),
    ]


class Predictor:
    def __init__(self, model):
        self.model = deepcopy(model)
        self.parent = Parent(model["parent_NN2932_model"])
        self.variants = {}
        self.dynamic = {}
        self.originals = {
            "01": self.parent.service.updated_physical,
            "11": physical(self.parent.service.old),
        }
        policy = self.model["service_repair"]
        for kind in ("base", "scsk"):
            for semantic, original in self.originals.items():
                variant = deepcopy(original)
                rates = semantics(variant)[kind + ":" + semantic]
                if semantic == "11":
                    rates["dma." + kind + ".peer_read_mb"] = 0.0
                    rates["dma." + kind + ".peer_copy_calls"] = 0.0
                if semantic == "01":
                    rates["dma." + kind + ".a_source_excess_klines"] = policy[
                        "A_partial_row_extra_us_per_kline"
                    ]
                bundle = variant["baseline_model"]["baseline_model"]["baseline_bundle"]
                for c in bundle["implementation_models"].values():
                    weights = c["schedule"]["conversion_constants"]
                    for name in weights:
                        if name.startswith(("mte2.", "mte3.", "vector.", "scalar.")):
                            weights[name] *= policy["conversion_engine_scale"]
                    weights["phase.entry"] = policy["conversion_phase_entry_us"]
                    weights["phase.extra_cores"] = policy["conversion_extra_core_us"]
                    weights["phase.log2_cores"] = policy["conversion_log2_cores_us"]
                self.variants[kind + ":" + semantic] = variant

    def predict(self, row):
        r, dtype = static_row(row)
        key = family(r)
        old = self.parent.predict(r)
        if key is None:
            aux = auxiliary_family(r)
            head = self.model.get("auxiliary_static_heads", {}).get(aux)
            if head:
                from matmul_cost_model._engine.NN_source_window_basis_v2712 import (
                    transform,
                    names,
                )

                z = [1.0, *transform(r, old["predicted_us"], head["static_tanh_basis"])]
                assert head["feature_names"] == [
                    "bias",
                    *names(head["static_tanh_basis"]["hidden"]),
                ]
                delta = sum(
                    w * (v - a) / s
                    for w, v, a, s in zip(
                        head["weights"], z, head["center"], head["scale"]
                    )
                )
                delta = max(-head["log_cap"], min(head["log_cap"], delta))
                return dict(
                    old,
                    predicted_us=old["predicted_us"] * exp(delta),
                    old_frozen_predicted_us=old["predicted_us"],
                    auxiliary_static_head=aux,
                    auxiliary_log_correction=delta,
                    repair_applied=True,
                    input_dtype=dtype,
                    model_version=self.model["version"],
                    target_measurements_required=False,
                )
            return dict(
                old,
                input_dtype=dtype,
                repair_applied=False,
                model_version=self.model["version"],
                target_measurements_required=False,
            )
        service = self.parent.service.predict(r)
        pold = service["new_pipeline_us"]
        program = compile_plan(r)
        aggregate, unique = aggregate_work(program)
        variant = self.variants[key]
        if key.endswith("11"):
            # The event graph's shared.mb node multiplies logical source_mb. This
            # deterministic work conversion makes its cost beta*actual NZ bytes;
            # beta itself remains a fixed offline constant.
            policy = self.model["service_repair"]
            cost = shared_cost(program, policy, aggregate, unique)
            work_scale = cost / program["source_mb"] if program["source_mb"] else 0.0
            tag = (key, round(work_scale, 14))
            if tag not in self.dynamic:
                changed = deepcopy(variant)
                bundle = changed["baseline_model"]["baseline_model"]["baseline_bundle"]
                bundle["implementation_models"][r["implementation"]]["parameters"][
                    "shared.mb"
                ] = work_scale
                if len(self.dynamic) > 128:
                    self.dynamic.clear()
                self.dynamic[tag] = changed
            variant = self.dynamic[tag]
        path = (
            Old.grouped_pipeline
            if Old.use_group_recurrence(r)
            else Old.reference_pipeline
        )(r, variant)
        pnew = path["predicted_us"]
        budget = old["predicted_us"] * pnew / pold
        x = features(r, program, path["conversion_phase_us"], budget)
        head = self.model.get("heads", {}).get(key)
        delta = 0.0
        if head:
            z = head_features(r, budget, x, head)
            delta = sum(
                w * (v - a) / s
                for w, v, a, s in zip(head["weights"], z, head["center"], head["scale"])
            )
            cap = head["log_cap"]
            delta = max(-cap, min(cap, delta))
        prediction = budget * exp(delta)
        if not isfinite(prediction) or prediction <= 0:
            raise ValueError("Positive static prediction required")
        return dict(
            predicted_us=prediction,
            input_dtype=dtype,
            model_version=self.model["version"],
            repair_applied=True,
            semantic_repair_family=key,
            old_frozen_predicted_us=old["predicted_us"],
            old_pipeline_us=pold,
            repaired_pipeline_us=pnew,
            computed_repaired_budget_us=budget,
            conversion_phase_us=path["conversion_phase_us"],
            bounded_head_log_correction=delta,
            static_features=x,
            active_cores=program["active_cores"],
            NZ_aggregate_source_MB=aggregate,
            NZ_unique_source_MB=unique,
            NZ_shared_bound_us=(
                shared_cost(program, self.model["service_repair"], aggregate, unique)
                if key.endswith("11")
                else 0.0
            ),
            target_measurements_required=False,
            hardware_execution_performed=False,
            independent_accuracy_qualified=False,
        )

    def pipeline(self, row):
        """Same fixed event graph, with every external residual removed."""
        r, dtype = static_row(row)
        key = family(r)
        program = compile_plan(r)
        variant = (
            self.originals[
                (
                    "01"
                    if r["implementation"] != "catlass"
                    and not r["plan"]["nd2nz_a"]
                    and r["plan"]["nd2nz_b"]
                    else "11"
                )
            ]
            if key is None
            else deepcopy(self.variants[key])
        )
        if key and key.endswith("11"):
            aggregate, unique = aggregate_work(program)
            p = self.model["service_repair"]
            variant["baseline_model"]["baseline_model"]["baseline_bundle"][
                "implementation_models"
            ][r["implementation"]]["parameters"]["shared.mb"] = (
                shared_cost(program, p, aggregate, unique) / program["source_mb"]
            )
        path = (
            Old.grouped_pipeline
            if Old.use_group_recurrence(r)
            else Old.reference_pipeline
        )(r, variant)
        return dict(
            path,
            input_dtype=dtype,
            external_residual_applied=False,
            target_measurements_required=False,
            hardware_execution_performed=False,
            model_version=self.model["version"],
        )
