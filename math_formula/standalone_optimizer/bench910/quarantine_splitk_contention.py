#!/usr/bin/env python3
"""Move timed cases exposed to a concurrent device-0 job out of the sample."""

from datetime import datetime
from pathlib import Path
import shutil


ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/splitk_family_selection_v1')
TIMING = ROOT / 'timing'
QUARANTINE = ROOT / 'quarantine_device_contention_20261030'
# npu-smi/ps observed process 2116901 (spmm_csr_aicore) start time.
CUTOFF = datetime(2026, 10, 30, 11, 18, 47).timestamp()


def main() -> None:
    for round_dir in sorted(TIMING.glob('r*')):
        for shape_dir in sorted(round_dir.iterdir()):
            if not shape_dir.is_dir():
                continue
            for arm_dir in sorted(shape_dir.iterdir()):
                if not arm_dir.is_dir():
                    continue
                case = arm_dir / 'case.json'
                # Round 1 began after the competing process started. Round
                # 0 cases whose profile completed after the start are also
                # excluded. Any unfinished case is excluded conservatively.
                exposed = (round_dir.name != 'r0' or not case.exists() or
                           case.stat().st_mtime >= CUTOFF)
                if exposed:
                    target = QUARANTINE / round_dir.name / shape_dir.name / arm_dir.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise RuntimeError(f'quarantine target exists: {target}')
                    shutil.move(str(arm_dir), str(target))
                    print('quarantined', target.relative_to(ROOT))


if __name__ == '__main__':
    main()
