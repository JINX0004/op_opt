"""Public static-input API. No hardware, timing labels, or online fitting."""

from copy import deepcopy
from hashlib import sha256
from itertools import groupby
from math import isfinite
from pathlib import Path
from threading import RLock
import json

DATA = Path(__file__).resolve().parent / "data"
MODEL_SHA256 = "63e3be969dfa453044e9f0b92b9d4166ee8ffb08b485b9724fd6244a5e5b590e"
KINDS = ("catlass", "base", "scsk", "deterministic")
ALIASES = {
    "ops_nn_base": "base",
    "ops_nn_single_core_split_k": "scsk",
    "ops_nn_deterministic_split_k": "deterministic",
}
DTYPES = {
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}
PROFILE = "warm400_gate_targetC"
_LOCK = RLock()  # The frozen recurrence uses shared private module caches.


def model_info():
    """Return provenance, validation scores and support limits, not target data."""
    return json.loads((DATA / "model_info.json").read_text(encoding="utf-8"))


def _identity(implementation, shape, dtype, cores):
    if not isinstance(implementation, str):
        raise ValueError("implementation must be a string")
    kind = ALIASES.get(implementation, implementation)
    if kind not in KINDS:
        raise ValueError("implementation must be catlass, base, scsk or deterministic")
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise ValueError(
            "dtype must be fp16/float16 or bf16/bfloat16; FP32 is not calibrated"
        )
    dtype = DTYPES[dtype]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(type(v) is not int for v in shape)
    ):
        raise ValueError("shape must contain integer [M, N, K]")
    m, n, k = shape
    if not (
        16 <= m <= 128
        and m % 16 == 0
        and n in (16, 24, 32, 64, 96, 128)
        and 64 <= k <= 131072
        and k % 16 == 0
    ):
        raise ValueError(
            "outside reviewed domain: M16..128 /16, N16,24,32,64,96,128, K64..131072 /16"
        )
    if type(cores) is not int or not 1 <= cores <= 24:
        raise ValueError("cores must be an integer in 1..24")
    return kind, list(shape), dtype


def _check_keys(obj, allowed, context):
    if not isinstance(obj, dict):
        raise ValueError(context + " must be a JSON object")
    extra = set(obj) - set(allowed)
    if extra:
        raise ValueError("unknown " + context + " fields: " + ", ".join(sorted(extra)))


def supported_tilings():
    """The actual reviewed templates, not an unconstrained hardware tiler."""
    from ._engine.matmul_static_unified_v2342 import NN_TEMPLATES
    from ._engine.matmul_reviewed_tiling_recipe_v2314 import CAT

    return dict(
        catlass={name: dict(l1_tile=l1, l0_tile=l0) for name, (l1, l0) in CAT.items()},
        **{
            kind: [dict(base_m=m, base_n=n, base_k=128, step_k=s) for m, n, s in values]
            for kind, values in NN_TEMPLATES.items()
        }
    )


def make_config(implementation, shape, dtype, cores, tiling, *, case_id=None):
    """Expand a compact tiling request into the complete reviewed static plan."""
    return expand_config(
        dict(
            implementation=implementation,
            shape=shape,
            dtype=dtype,
            cores=cores,
            tiling=tiling,
            **({"case_id": case_id} if case_id is not None else {})
        )
    )


def expand_config(config):
    """Validate compact requests or complete exported static_input objects.

    Arbitrary TCubeTiling, layouts, offsets and buffers are rejected: silently
    projecting an unknown plan onto a calibrated template would be misleading.
    """
    from ._engine.matmul_reviewed_tiling_recipe_v2314 import CAT, ops_plan
    from ._engine.matmul_static_unified_v2342 import NN_TEMPLATES
    from ._engine.NN_layout_service_repair_predict_v3453 import static_row

    if not isinstance(config, dict):
        raise ValueError("each configuration must be a JSON object")
    if "static_input" in config:
        config = config["static_input"]
    _check_keys(
        config,
        (
            "implementation",
            "shape",
            "dtype",
            "cores",
            "tiling",
            "case_id",
            "trans_a",
            "trans_b",
            "batch",
            "plan",
            "tile",
            "l1_tile",
            "l0_tile",
            "execution_profile",
            "source_a_offset_extra_bytes",
            "source_b_offset_extra_bytes",
        ),
        "configuration",
    )
    for flag in ("trans_a", "trans_b"):
        if flag in config and (type(config[flag]) is not bool or config[flag]):
            raise ValueError(
                flag + ": this release supports NN only; use boolean false"
            )
    if type(config.get("batch", 1)) is not int or config.get("batch", 1) != 1:
        raise ValueError("only batch=1 is supported")
    if config.get("execution_profile", PROFILE) != PROFILE:
        raise ValueError(
            "only the frozen warm400_gate_targetC execution profile is supported"
        )
    for key in ("source_a_offset_extra_bytes", "source_b_offset_extra_bytes"):
        if type(config.get(key, 0)) is not int or config.get(key, 0) != 0:
            raise ValueError("additional source offsets are not qualified")
    kind, shape, dtype = _identity(
        config.get("implementation"),
        config.get("shape"),
        config.get("dtype", "float16"),
        config.get("cores"),
    )
    row = dict(
        implementation=kind,
        shape=shape,
        dtype=dtype,
        cores=config["cores"],
        trans_a=False,
        trans_b=False,
        batch=1,
        execution_profile=PROFILE,
    )
    if "tiling" in config:
        if any(key in config for key in ("plan", "tile", "l1_tile", "l0_tile")):
            raise ValueError(
                "provide either compact tiling or a complete plan, not both"
            )
        t = deepcopy(config["tiling"])
        if kind == "catlass":
            if isinstance(t, str):
                t = {"preset": t}
            _check_keys(t, ("preset", "l1_tile", "l0_tile"), "CATLASS tiling")
            if "preset" in t:
                if len(t) != 1 or t["preset"] not in CAT:
                    raise ValueError(
                        "CATLASS preset must be default, deepk, widek or midk, without contradictory dimensions"
                    )
                name = t["preset"]
            else:
                name = next(
                    (
                        key
                        for key, (l1, l0) in CAT.items()
                        if t.get("l1_tile") == l1 and t.get("l0_tile") == l0
                    ),
                    None,
                )
                if name is None:
                    raise ValueError("unreviewed CATLASS L1/L0 tiling")
            row.update(
                tile=name, l1_tile=list(CAT[name][0]), l0_tile=list(CAT[name][1])
            )
        else:
            _check_keys(t, ("base_m", "base_n", "base_k", "step_k"), "ops-nn tiling")
            if set(t) != {"base_m", "base_n", "base_k", "step_k"} or any(
                type(v) is not int for v in t.values()
            ):
                raise ValueError(
                    "ops-nn tiling requires integer base_m, base_n, base_k, step_k"
                )
            recipe = (t["base_m"], t["base_n"], t["step_k"])
            if t["base_k"] != 128 or recipe not in NN_TEMPLATES[kind]:
                raise ValueError(
                    "unreviewed "
                    + kind
                    + " tiling; inspect supported_tilings() or the info command"
                )
            if kind == "deterministic" and (
                row["cores"] < 2
                or row["cores"]
                > (shape[2] + 128 * t["step_k"] - 1) // (128 * t["step_k"])
            ):
                raise ValueError(
                    "deterministic requires >=2 cores and no more cores than nonempty K partitions"
                )
            row["plan"] = ops_plan(
                shape,
                (
                    "public",
                    kind,
                    t["base_m"],
                    t["base_n"],
                    128,
                    t["step_k"],
                    row["cores"],
                ),
            )
            row["plan"]["dtype"] = dtype
    elif kind == "catlass":
        if "plan" in config:
            raise ValueError("CATLASS does not accept ops-nn plan")
        for key in ("tile", "l1_tile", "l0_tile"):
            if key not in config:
                raise ValueError(
                    "missing CATLASS " + key + "; alternatively use tiling.preset"
                )
            row[key] = deepcopy(config[key])
    else:
        if any(key in config for key in ("tile", "l1_tile", "l0_tile")):
            raise ValueError("ops-nn does not accept CATLASS tiling")
        if "plan" not in config:
            raise ValueError("explicit tiling or complete plan is required")
        row["plan"] = deepcopy(config["plan"])
        if not isinstance(row["plan"], dict):
            raise ValueError("plan must be an object")
        _check_keys(
            row["plan"],
            (
                "m",
                "n",
                "k",
                "dtype",
                "trans_a",
                "trans_b",
                "tcube",
                "raw_words",
                "nd2nz_a",
                "nd2nz_b",
                "block_dim",
            ),
            "plan",
        )
    if kind == "catlass":
        if any(
            not isinstance(row[key], list)
            or len(row[key]) != 3
            or any(type(v) is not int for v in row[key])
            for key in ("l1_tile", "l0_tile")
        ):
            raise ValueError(
                "L1/L0 tiling dimensions must be integer lists of length 3"
            )
    else:
        plan = row["plan"]
        if not isinstance(plan.get("tcube"), dict) or any(
            type(v) is not int for v in plan["tcube"].values()
        ):
            raise ValueError("tcube fields must be integers")
        if not isinstance(plan.get("raw_words"), list) or any(
            type(v) is not int for v in plan["raw_words"]
        ):
            raise ValueError("raw_words must be an integer list")
        if any(
            type(plan.get(k)) is not bool or plan[k] for k in ("trans_a", "trans_b")
        ):
            raise ValueError("plan transpose fields must be boolean false")
    static_row(
        row
    )  # Native raw words and every supported buffer/layout field must agree.
    cid = config.get("case_id")
    if cid is not None and (not isinstance(cid, str) or not cid.strip()):
        raise ValueError("case_id must be a nonempty string")
    row["case_id"] = (
        cid
        if cid is not None
        else "cfg_"
        + sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    )
    return row


def generate_candidates(
    shape, dtype="float16", implementations=KINDS, cores=(1, 2, 4, 8, 16, 24)
):
    """Enumerate reviewed plans; this does not run the upstream host tiler."""
    if isinstance(implementations, str):
        implementations = KINDS if implementations == "all" else (implementations,)
    implementations = tuple(ALIASES.get(k, k) for k in implementations)
    if not implementations or len(set(implementations)) != len(implementations):
        raise ValueError("implementations must be nonempty and unique")
    cores = tuple(cores)
    if not cores or len(set(cores)) != len(cores):
        raise ValueError("cores must be nonempty and unique")
    templates = supported_tilings()
    out = []
    for kind in implementations:
        for c in cores:
            _identity(kind, shape, dtype, c)
            choices = (
                ({"preset": name} for name in templates["catlass"])
                if kind == "catlass"
                else templates[kind]
            )
            for t in choices:
                if kind == "deterministic" and (
                    c < 2
                    or c > (shape[2] + 128 * t["step_k"] - 1) // (128 * t["step_k"])
                ):
                    continue
                out.append(make_config(kind, shape, dtype, c, t))
    return out


class MatmulCostModel:
    """Reusable fixed model. Construct once, then score many static plans."""

    def __init__(self):
        raw = (DATA / "model.json").read_bytes()
        if sha256(raw).hexdigest() != MODEL_SHA256:
            raise RuntimeError("fixed model.json SHA256 mismatch")
        from ._engine.NN_component_service_predict_v3490 import Predictor

        with _LOCK:
            self._engine = Predictor(json.loads(raw))

    def predict(self, config, *, semantic=False, explain=False):
        from ._engine.NN_layout_service_repair_predict_v3453 import static_row
        from ._engine.costmodel.matmul_four_path_v590 import compile_plan

        row = expand_config(config)
        static, _ = static_row(row)
        with _LOCK:
            detail = (
                self._engine.pipeline(row) if semantic else self._engine.predict(row)
            )
            pipeline = (
                detail
                if semantic
                else (self._engine.pipeline(row) if explain else None)
            )
            active = compile_plan(static)["active_cores"]
        value = float(detail["predicted_us"])
        if not isfinite(value) or value <= 0:
            raise ArithmeticError("invalid fixed-model prediction")
        result = dict(
            case_id=row["case_id"],
            implementation=row["implementation"],
            shape=row["shape"],
            dtype=row["dtype"],
            cores=row["cores"],
            active_compute_cores=active,
            predicted_us=value,
            predicted_ms=value / 1000,
            model_version="NN_component_service_v3490_BASE64",
            hardware="Ascend 910B2",
            formula="semantic_pipeline" if semantic else "fixed_calibrated",
            static_input=row,
            execution_assumption=PROFILE,
            target_measurements_required=False,
            hardware_execution_performed=False,
            unseen_input_error_guaranteed=False,
            all_candidate_orders_certified=False,
        )
        if explain:
            result.update(formula_details=detail, semantic_pipeline=pipeline)
        return result

    def predict_many(self, configs, **kwargs):
        return [self.predict(c, **kwargs) for c in configs]

    def compare(self, configs, *, top=5, semantic=False):
        if type(top) is not int or top < 0:
            raise ValueError("top must be a nonnegative integer; 0 means all")
        rows = self.predict_many(configs, semantic=semantic)
        if not rows:
            raise ValueError("no legal candidates")
        if len({r["case_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate case_id in candidate list")
        key = lambda r: (r["implementation"], tuple(r["shape"]), r["dtype"])
        groups = []
        for identity, items in groupby(sorted(rows, key=key), key=key):
            ranked = sorted(items, key=lambda r: r["predicted_us"])
            selected = ranked if top == 0 else ranked[:top]
            groups.append(
                dict(
                    implementation=identity[0],
                    shape=list(identity[1]),
                    dtype=identity[2],
                    candidate_count=len(ranked),
                    ranking=[
                        dict(
                            case_id=r["case_id"],
                            predicted_us=r["predicted_us"],
                            cores=r["cores"],
                            active_compute_cores=r["active_compute_cores"],
                        )
                        for r in selected
                    ],
                )
            )
        return dict(
            total_candidates=len(rows),
            groups=groups,
            predictions=rows,
            stable_order_guaranteed=False,
            target_measurements_required=False,
            hardware_execution_performed=False,
        )
