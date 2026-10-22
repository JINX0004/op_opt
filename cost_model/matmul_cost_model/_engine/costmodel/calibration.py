from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CubeCalibrationPoint:
    cube_blocks: int
    cube_time_us: float
    frequency_mhz: float

    def __post_init__(self) -> None:
        if self.cube_blocks <= 0:
            raise ValueError("cube_blocks must be positive")
        if not math.isfinite(self.cube_time_us) or self.cube_time_us <= 0:
            raise ValueError("cube_time_us must be finite and positive")
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and positive")

    @property
    def cube_cycles(self) -> float:
        # MHz * microseconds is cycles.
        return self.cube_time_us * self.frequency_mhz


@dataclass(frozen=True)
class CubePipelineFit:
    latency_cycles: float
    issue_interval_cycles: float
    r_squared: float
    rmse_cycles: float
    max_abs_error_cycles: float
    point_count: int
    observations: tuple[dict[str, float | int], ...]

    def predict_cycles(self, cube_blocks: int) -> float:
        if cube_blocks <= 0:
            raise ValueError("cube_blocks must be positive")
        return self.latency_cycles + (cube_blocks - 1) * self.issue_interval_cycles

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_cube_pipeline(points: Iterable[CubeCalibrationPoint]) -> CubePipelineFit:
    """Fit ``T = Lblock + (blocks - 1) * IIblock`` by ordinary least squares."""

    values = tuple(points)
    if len(values) < 3:
        raise ValueError("at least three Cube calibration points are required")
    if len({point.cube_blocks for point in values}) < 2:
        raise ValueError("at least two distinct cube block counts are required")

    xs = [float(point.cube_blocks - 1) for point in values]
    ys = [point.cube_cycles for point in values]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    if intercept <= 0 or slope <= 0:
        raise ValueError(
            "non-physical Cube fit: latency and issue interval must both be positive"
        )

    predictions = [intercept + slope * x for x in xs]
    residuals = [actual - predicted for actual, predicted in zip(ys, predictions)]
    squared_error = sum(value**2 for value in residuals)
    total_variance = sum((value - y_mean) ** 2 for value in ys)
    r_squared = 1.0 - squared_error / total_variance if total_variance else 1.0
    observations = tuple(
        {
            "cube_blocks": point.cube_blocks,
            "observed_cycles": point.cube_cycles,
            "predicted_cycles": predicted,
            "residual_cycles": residual,
        }
        for point, predicted, residual in zip(values, predictions, residuals)
    )
    return CubePipelineFit(
        latency_cycles=intercept,
        issue_interval_cycles=slope,
        r_squared=r_squared,
        rmse_cycles=math.sqrt(squared_error / len(values)),
        max_abs_error_cycles=max(abs(value) for value in residuals),
        point_count=len(values),
        observations=observations,
    )


def load_cube_points_jsonl(path: str | Path) -> tuple[CubeCalibrationPoint, ...]:
    """Load normalized profile records emitted by the calibration collection step."""

    points = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                shape = payload["shape"]
                profile = payload["profile"]
                points.append(
                    CubeCalibrationPoint(
                        cube_blocks=int(shape["cube_blocks"]),
                        cube_time_us=float(profile["cube_time_us"]),
                        frequency_mhz=float(profile["current_freq_mhz"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid Cube profile record at line {line_number}: {error}") from error
    return tuple(points)


def save_cube_pipeline_fit(path: str | Path, fit: CubePipelineFit) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(fit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class TransferCalibrationPoint:
    mode: str
    transfer_bytes: int
    active_time_us: float
    frequency_mhz: float
    main_memory_read_kb: float | None = None
    l2_read_hit_rate_pct: float | None = None

    def __post_init__(self) -> None:
        if not self.mode:
            raise ValueError("transfer mode must not be empty")
        if self.transfer_bytes <= 0:
            raise ValueError("transfer_bytes must be positive")
        if not math.isfinite(self.active_time_us) or self.active_time_us <= 0:
            raise ValueError("active_time_us must be finite and positive")
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and positive")

    @property
    def active_cycles(self) -> float:
        return self.active_time_us * self.frequency_mhz


@dataclass(frozen=True)
class TransferPipelineFit:
    mode: str
    fixed_cycles: float
    cycles_per_byte: float
    effective_bandwidth_gbps: float
    r_squared: float
    rmse_cycles: float
    max_abs_error_cycles: float
    point_count: int
    observations: tuple[dict[str, float | int], ...]

    def predict_cycles(self, transfer_bytes: int) -> float:
        if transfer_bytes <= 0:
            raise ValueError("transfer_bytes must be positive")
        return self.fixed_cycles + transfer_bytes * self.cycles_per_byte

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_transfer_pipeline(
    points: Iterable[TransferCalibrationPoint],
) -> TransferPipelineFit:
    """Fit active cycles as a fixed cost plus a linear byte-transfer cost."""

    values = tuple(points)
    if len(values) < 3:
        raise ValueError("at least three transfer calibration points are required")
    modes = {point.mode for point in values}
    if len(modes) != 1:
        raise ValueError("all transfer calibration points must have the same mode")
    if len({point.transfer_bytes for point in values}) < 2:
        raise ValueError("at least two distinct transfer sizes are required")

    xs = [float(point.transfer_bytes) for point in values]
    ys = [point.active_cycles for point in values]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    if intercept < 0 or slope <= 0:
        raise ValueError(
            "non-physical transfer fit: fixed cost must be non-negative and byte cost positive"
        )
    predictions = [intercept + slope * x for x in xs]
    residuals = [actual - predicted for actual, predicted in zip(ys, predictions)]
    squared_error = sum(value**2 for value in residuals)
    total_variance = sum((value - y_mean) ** 2 for value in ys)
    r_squared = 1.0 - squared_error / total_variance if total_variance else 1.0
    frequency_mhz = sum(point.frequency_mhz for point in values) / len(values)
    bytes_per_cycle = 1.0 / slope
    observations = tuple(
        {
            "transfer_bytes": point.transfer_bytes,
            "observed_cycles": point.active_cycles,
            "predicted_cycles": predicted,
            "residual_cycles": residual,
        }
        for point, predicted, residual in zip(values, predictions, residuals)
    )
    return TransferPipelineFit(
        mode=next(iter(modes)),
        fixed_cycles=intercept,
        cycles_per_byte=slope,
        effective_bandwidth_gbps=bytes_per_cycle * frequency_mhz / 1000.0,
        r_squared=r_squared,
        rmse_cycles=math.sqrt(squared_error / len(values)),
        max_abs_error_cycles=max(abs(value) for value in residuals),
        point_count=len(values),
        observations=observations,
    )


def load_transfer_points_jsonl(
    path: str | Path, mode: str
) -> tuple[TransferCalibrationPoint, ...]:
    valid_modes = {
        "raw_mte2",
        "nd2nz_mte2",
        "raw_mte2_reuse",
        "nd2nz_mte2_reuse",
        "mte1_load",
        "mte1_a_nt",
        "mte1_b_nt",
    }
    if mode not in valid_modes:
        raise ValueError(f"unsupported transfer mode: {mode}")
    points = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") != "ok":
                continue
            experiment = payload.get("experiment", {})
            if experiment.get("mode") != mode:
                continue
            profile = payload.get("profile", {})
            try:
                if mode in {"mte1_load", "mte1_a_nt", "mte1_b_nt"}:
                    transfer_bytes = int(profile["requested_mte1_bytes"])
                    active_time_us = float(profile["mte1_time_us"])
                else:
                    transfer_bytes = int(profile["requested_mte2_bytes"])
                    active_time_us = float(profile["mte2_time_us"])
                points.append(
                    TransferCalibrationPoint(
                        mode=mode,
                        transfer_bytes=transfer_bytes,
                        active_time_us=active_time_us,
                        frequency_mhz=float(profile["current_freq_mhz"]),
                        main_memory_read_kb=float(profile["main_memory_read_kb"]),
                        l2_read_hit_rate_pct=float(profile["l2_read_hit_rate_pct"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid transfer profile record at line {line_number}: {error}"
                ) from error
    return tuple(points)


def save_transfer_pipeline_fit(path: str | Path, fit: TransferPipelineFit) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(fit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
