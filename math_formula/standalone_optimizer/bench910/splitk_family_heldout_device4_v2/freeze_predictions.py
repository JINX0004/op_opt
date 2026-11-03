#!/usr/bin/env python3
"""Freeze measured-output family predictions before device timing."""

import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).parent
SOLVER = HERE.parent.parent / 'build/debug/mmopt_splitk_family_model'
MANIFEST = HERE / 'manifest.json'
OUT = HERE / 'model_predictions.csv'
META = HERE / 'model_predictions_metadata.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text())
    shapes = [tuple(shape) for shape in manifest['shapes']]
    assert len(shapes) == len(set(shapes))
    stdin = ''.join(' '.join(map(str, shape)) + '\n' for shape in shapes)
    result = subprocess.run([str(SOLVER)], input=stdin, text=True,
                            capture_output=True, check=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == len(shapes)
    assert [(int(row['m']), int(row['n']), int(row['k']))
            for row in rows] == shapes
    assert all(row['selected'] in ('single', 'deterministic') for row in rows)
    OUT.write_text(result.stdout)
    metadata = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': digest(MANIFEST),
        'predictions_sha256': digest(OUT),
        'solver_cpp_sha256': digest(HERE.parent.parent / 'mip_solver.cpp'),
        'profile_cpp_sha256': digest(HERE.parent / 'splitk_family_model.cpp'),
        'source_timing_started': False,
    }
    META.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(f'frozen {len(rows)} predictions in {OUT}')


if __name__ == '__main__':
    main()
