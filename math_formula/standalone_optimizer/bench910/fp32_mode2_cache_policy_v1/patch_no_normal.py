#!/usr/bin/env python3
"""Variant control: leave policy 0's source cache-hint calls untouched."""

import sys
from pathlib import Path


header = Path(sys.argv[1])
source = header.read_text()
before = '''                aGlobal_.SetL2CacheHint((policy == 1.0f || policy == 3.0f) ?
                    CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL);
                bGlobal_.SetL2CacheHint((policy == 2.0f || policy == 3.0f) ?
                    CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL);'''
after = '''                if (policy == 1.0f || policy == 3.0f) {
                    aGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                }
                if (policy == 2.0f || policy == 3.0f) {
                    bGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                }'''
assert source.count(before) == 1
header.write_text(source.replace(before, after, 1))
