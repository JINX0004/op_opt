#!/usr/bin/env python3
"""Freeze mode-2/5/6 reduced-MIP predictions before source timing."""

import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OPT = HERE.parent.parent
MANIFEST = HERE / 'manifest.json'
MODEL = HERE / 'model.cpp'
OUT = HERE / 'model_predictions.csv'
META = HERE / 'model_predictions_metadata.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    shapes = [tuple(shape) for shape in manifest['shapes']]
    assert len(shapes) == len(set(shapes))
    binary = Path('/tmp/mmopt_single_mode_model')
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(OPT), str(MODEL),
                    str(OPT / 'build/debug/libmmopt_mip.a'),
                    str(OPT / 'build/debug/libmmopt.a'), '-lhighs', '-o', str(binary)],
                   check=True)
    stdin = ''.join(' '.join(map(str, shape)) + '\n' for shape in shapes)
    result = subprocess.run([str(binary)], input=stdin, text=True,
                            capture_output=True, check=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert [(int(r['m']), int(r['n']), int(r['k'])) for r in rows] == shapes
    OUT.write_text(result.stdout)
    META.write_text(json.dumps({
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': sha(MANIFEST),
        'predictions_sha256': sha(OUT),
        'model_cpp_sha256': sha(MODEL),
        'solver_cpp_sha256': sha(OPT / 'mip_solver.cpp'),
        'finalizer_cpp_sha256': sha(OPT / 'plan_finalizer.cpp'),
        'source_timing_started': False,
    }, indent=2, sort_keys=True) + '\n')
    print(f'frozen {len(rows)} predictions')


if __name__ == '__main__':
    main()
