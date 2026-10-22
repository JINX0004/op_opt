"""Command line interface for standalone prediction and offline evaluation."""

from argparse import ArgumentParser
from pathlib import Path
import json, sys
# from .api import (
#     MatmulCostModel,
#     make_config,
#     generate_candidates,
#     model_info,
#     supported_tilings,
# )
from .evaluation import evaluate, read_measurements


def _load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _list(value):
    return value if isinstance(value, list) else [value]


def _cores(text):
    try:
        values = [int(v) for v in text.split(",")]
    except ValueError:
        raise ValueError("--cores expects comma-separated integers")
    return values


def main(argv=None):
    parser = ArgumentParser(
        description="Ascend 910B2 static matmul model; Python standard library, no NPU required."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("info", "predict", "candidates", "compare", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, help="JSON output file; default stdout")
        if name in ("predict", "compare", "evaluate"):
            p.add_argument(
                "--input",
                type=Path,
                help="Compact/static configuration JSON object or list",
            )
        if name in ("predict", "candidates", "compare"):
            p.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
            p.add_argument("--dtype")
            p.add_argument("--implementation")
            p.add_argument("--cores")
        if name == "predict":
            p.add_argument("--tile", help="CATLASS preset: default/deepk/widek/midk")
            p.add_argument("--base-m", type=int)
            p.add_argument("--base-n", type=int)
            p.add_argument("--base-k", type=int)
            p.add_argument("--step-k", type=int)
            p.add_argument("--case-id")
            p.add_argument("--explain", action="store_true")
        if name in ("predict", "compare"):
            p.add_argument(
                "--semantic",
                action="store_true",
                help="Use static event graph P without external residuals",
            )
        if name == "compare":
            p.add_argument(
                "--top",
                type=int,
                default=5,
                help="Candidates shown per group; 0 means all. predictions always contains all.",
            )
        if name == "evaluate":
            p.add_argument(
                "--predictions",
                type=Path,
                help="Optional existing prediction/compare JSON instead of --input",
            )
            p.add_argument(
                "--measurements",
                type=Path,
                required=True,
                help="CSV: case_id,measured_us[,process_index]",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            result = dict(model=model_info(), supported_tilings=supported_tilings())
        elif args.command == "predict":
            if args.input:
                if any(
                    v is not None
                    for v in (
                        args.shape,
                        args.dtype,
                        args.implementation,
                        args.cores,
                        args.tile,
                        args.base_m,
                        args.base_n,
                        args.base_k,
                        args.step_k,
                        args.case_id,
                    )
                ):
                    raise ValueError(
                        "--input cannot be combined with shape/implementation/cores/tiling options"
                    )
                cases = _load(args.input)
                m = MatmulCostModel()
                values = m.predict_many(
                    _list(cases), semantic=args.semantic, explain=args.explain
                )
                result = values if isinstance(cases, list) else values[0]
            else:
                if (
                    args.shape is None
                    or args.implementation is None
                    or args.cores is None
                ):
                    raise ValueError(
                        "predict requires --input OR --shape --implementation --cores and tiling"
                    )
                cores = _cores(args.cores)
                if len(cores) != 1:
                    raise ValueError(
                        "predict expects one core count; use compare for a core sweep"
                    )
                tiling = (
                    {"preset": args.tile}
                    if args.tile
                    else dict(
                        base_m=args.base_m,
                        base_n=args.base_n,
                        base_k=128 if args.base_k is None else args.base_k,
                        step_k=args.step_k,
                    )
                )
                if args.tile and any(
                    v is not None
                    for v in (args.base_m, args.base_n, args.base_k, args.step_k)
                ):
                    raise ValueError(
                        "CATLASS --tile cannot be combined with ops-nn tiling options"
                    )
                result = MatmulCostModel().predict(
                    make_config(
                        args.implementation,
                        args.shape,
                        args.dtype or "float16",
                        cores[0],
                        tiling,
                        case_id=args.case_id,
                    ),
                    semantic=args.semantic,
                    explain=args.explain,
                )
        elif args.command in ("candidates", "compare"):
            if getattr(args, "input", None):
                if any(
                    v is not None
                    for v in (args.shape, args.dtype, args.implementation, args.cores)
                ):
                    raise ValueError(
                        "--input cannot be combined with shape/dtype/implementation/cores options"
                    )
                cases = _list(_load(args.input))
            else:
                if args.shape is None:
                    raise ValueError("--shape or --input is required")
                cases = generate_candidates(
                    args.shape,
                    args.dtype or "float16",
                    args.implementation or "all",
                    _cores(args.cores or "1,2,4,8,16,24"),
                )
            result = (
                cases
                if args.command == "candidates"
                else MatmulCostModel().compare(
                    cases, top=args.top, semantic=args.semantic
                )
            )
        else:
            if bool(args.input) == bool(args.predictions):
                raise ValueError(
                    "evaluate requires exactly one of --input or --predictions"
                )
            if args.predictions:
                raw = _load(args.predictions)
                preds = (
                    raw["predictions"]
                    if isinstance(raw, dict) and "predictions" in raw
                    else _list(raw)
                )
            else:
                preds = MatmulCostModel().predict_many(_list(_load(args.input)))
            result = evaluate(preds, read_measurements(args.measurements))
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as ex:
        print("error: " + str(ex), file=sys.stderr)
        return 2


if __name__ == "__main__":
    from matmul_cost_model import MatmulCostModel, make_config, generate_candidates

    model = MatmulCostModel()  # 加载固定参数一次，多次复用
    config = make_config(
        implementation="ops_nn_deterministic_split_k",
        shape=[16, 16, 102400], dtype="fp16", cores=24,
        tiling={"base_m": 16, "base_n": 16, "base_k": 2048, "step_k": 1},
    )
    prediction = model.predict(config)
    print(prediction["predicted_us"])
    # raise SystemExit(main())
