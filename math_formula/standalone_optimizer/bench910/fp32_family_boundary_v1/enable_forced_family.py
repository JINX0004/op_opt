#!/usr/bin/env python3
"""Add BASE to the isolated source tiler's existing research family switch."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    path = args.source
    body = path.read_text()
    old = '''        } else if (std::strcmp(researchFamily, "deterministic") == 0) {
            tilingSelect_ = TilingCalcSelect::DETERMINISTIC_SPLIT_K;
        }
'''
    new = '''        } else if (std::strcmp(researchFamily, "deterministic") == 0) {
            tilingSelect_ = TilingCalcSelect::DETERMINISTIC_SPLIT_K;
        } else if (std::strcmp(researchFamily, "base") == 0) {
            tilingSelect_ = TilingCalcSelect::BASE;
        }
'''
    if new in body:
        print(f'already patched: {path}')
        return
    if body.count(old) != 1 or 'MMV3_RESEARCH_SPLITK_FAMILY' not in body:
        raise RuntimeError('expected isolated family probe was not found')
    path.write_text(body.replace(old, new, 1))
    print(f'patched: {path}')


if __name__ == '__main__':
    main()
