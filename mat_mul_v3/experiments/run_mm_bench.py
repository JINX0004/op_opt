#!/usr/bin/env python3
"""Time MatMulV3-style GEMMs on the two experiment datasets.

Backends:
  dryrun  print shapes only
  torch   torch.mm on NPU (needs torch_npu)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
DATASETS = {
    "same_core": EXPERIMENTS / "datasets" / "same_core_k.csv",
    "multi_core": EXPERIMENTS / "datasets" / "multi_core_k.csv",
}

AIC_NUM = 24
SPLIT_K_THRES = 27392
MULTI_CORE_SINGLE_K = 384
MIN_SPLITK_MN64 = 6144
MIN_SPLITK_K = 1000
MAX_SPLITK_K = 4608
M_THRESHOLD = 256
N_THRESHOLD = 256


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            row["trans_a"] = int(row["trans_a"])
            row["trans_b"] = int(row["trans_b"])
            row["m"] = int(row["m"])
            row["n"] = int(row["n"])
            row["k"] = int(row["k"])
            rows.append(row)
    return rows


def classify(row: dict, aic_num: int = AIC_NUM) -> str:
    """Approximate host gates for 24-AIC, no-bias ND MatMulV3 (A2/A3)."""
    m, n, k = row["m"], row["n"], row["k"]
    trans_a, trans_b = bool(row["trans_a"]), bool(row["trans_b"])
    dtype = row["dtype"]
    m_cnt = (m + 127) // 128
    n_cnt = (n + 127) // 128
    k_enough = k >= aic_num * MULTI_CORE_SINGLE_K
    mn_not_enough = m_cnt * n_cnt < aic_num // 2

    scsk = k >= SPLIT_K_THRES
    if scsk and n >= 2048 and m >= 1024:
        return "SINGLE_CORE_SPLIT_K"

    splitk_scene = k_enough and mn_not_enough and not (not trans_a and trans_b)
    small_mn_fp32 = dtype == "float32" and m <= 64 and n <= 64 and k >= MIN_SPLITK_MN64
    trans_mid_k = (
        k >= MIN_SPLITK_K
        and k <= MAX_SPLITK_K
        and m <= M_THRESHOLD
        and n <= N_THRESHOLD
        and trans_a
        and not trans_b
    )
    if splitk_scene or small_mn_fp32 or trans_mid_k or (k >= SPLIT_K_THRES and mn_not_enough):
        return "DETERMINISTIC_SPLIT_K"
    return "BASE"


def bench_torch(row: dict, warmup: int, iters: int, device: str) -> float:
    import torch

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[row["dtype"]]
    m, n, k = row["m"], row["n"], row["k"]
    a_shape = (k, m) if row["trans_a"] else (m, k)
    b_shape = (n, k) if row["trans_b"] else (k, n)
    a = torch.randn(*a_shape, device=device, dtype=dtype)
    b = torch.randn(*b_shape, device=device, dtype=dtype)
    if row["trans_a"]:
        a = a.t()
    if row["trans_b"]:
        b = b.t()

    def sync():
        if device.startswith("npu"):
            torch.npu.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(warmup):
        _ = a @ b
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    sync()
    return (time.perf_counter() - t0) * 1e3 / iters


def tflops(m: int, n: int, k: int, time_ms: float) -> float:
    if time_ms <= 0:
        return 0.0
    return (2.0 * m * n * k) / (time_ms * 1e-3) / 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BASE/same-cube vs multi-core K-split datasets")
    parser.add_argument("--dataset", choices=["same_core", "multi_core", "both"], default="both")
    parser.add_argument("--backend", choices=["dryrun", "torch"], default="dryrun")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    names = ["same_core", "multi_core"] if args.dataset == "both" else [args.dataset]
    out_rows = []
    print("dataset,group,expected,classified,dtype,trans_a,trans_b,m,n,k,time_ms,tflops")
    for name in names:
        for row in load_rows(DATASETS[name]):
            classified = classify(row)
            time_ms = ""
            tf = ""
            if args.backend == "torch":
                time_ms_f = bench_torch(row, args.warmup, args.iters, args.device)
                time_ms = f"{time_ms_f:.4f}"
                tf = f"{tflops(row['m'], row['n'], row['k'], time_ms_f):.4f}"
            line = (
                f"{name},{row['group']},{row['expected']},{classified},{row['dtype']},"
                f"{row['trans_a']},{row['trans_b']},{row['m']},{row['n']},{row['k']},"
                f"{time_ms},{tf}"
            )
            print(line)
            out_rows.append(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        header = "dataset,group,expected,classified,dtype,trans_a,trans_b,m,n,k,time_ms,tflops"
        args.out.write_text(header + "\n" + "\n".join(out_rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
