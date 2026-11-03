#!/usr/bin/env python3
"""Print compact progress for the 100-shape remote comparison."""

import json
from pathlib import Path


ROOT = Path('/home/xianfei/solvebase_bench_20261025/cases/online_100_vs_master_fp16_v1')


def main():
    result = {}
    for device in range(4, 8):
        path = ROOT / f'device{device}/summary.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        log = ROOT / f'device{device}.log'
        result[device] = {
            'completed': len(rows),
            'failures': [(row['shape_index'], row['error'][:100])
                         for row in rows if row.get('error')],
            'changed': sum(row.get('arms', {}).get('default', {}).get('mode') !=
                           row.get('arms', {}).get('online', {}).get('mode')
                           for row in rows if not row.get('error')),
            'last': log.read_text().splitlines()[-1] if log.exists()
            and log.stat().st_size else '',
        }
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
