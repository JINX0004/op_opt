#!/usr/bin/env python3
"""Patch only an isolated generated MatMulV3 FP32 device-header copy."""

import sys
from pathlib import Path


header = Path(sys.argv[1])
text = header.read_text()
before = '''        //  SetL2CacheHint 的逻辑: 左、右矩阵仅搬运一次的情形
        if (IS_NKM && (block_.matmulTilingData_->matmulTiling.stepN *
            block_.matmulTilingData_->matmulTiling.baseN == block_.matmulTilingData_->matmulTiling.N)) {
            aGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        } else if (!IS_NKM && (block_.matmulTilingData_->matmulTiling.stepM *
            block_.matmulTilingData_->matmulTiling.baseM == block_.matmulTilingData_->matmulTiling.M)) {
            bGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }'''
after = '''        // Research-only FP32 mode-2 cache policy. The benchmark writes a
        // 0..3 marker into A row 0's physical padding at logical K. It is
        // never part of the matrix multiplication. Policy bit 0 disables
        // A caching; bit 1 disables B caching. All four combinations use
        // this same device binary and host tiling library.
        if constexpr (sizeof(typename A_TYPE::T) == sizeof(float)) {
            if (block_.matmulTilingData_->matmulTiling.Kb == 48384 &&
                block_.matmulTilingData_->matmulTiling.Ka > 48384 &&
                block_.matmulTilingData_->matmulTiling.M >= 2048) {
                const float policy = aGlobal_.GetValue(48384);
                aGlobal_.SetL2CacheHint((policy == 1.0f || policy == 3.0f) ?
                    CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL);
                bGlobal_.SetL2CacheHint((policy == 2.0f || policy == 3.0f) ?
                    CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL);
            } else {
                if (IS_NKM && (block_.matmulTilingData_->matmulTiling.stepN *
                    block_.matmulTilingData_->matmulTiling.baseN ==
                    block_.matmulTilingData_->matmulTiling.N)) {
                    aGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                } else if (!IS_NKM &&
                    (block_.matmulTilingData_->matmulTiling.stepM *
                     block_.matmulTilingData_->matmulTiling.baseM ==
                     block_.matmulTilingData_->matmulTiling.M)) {
                    bGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                }
            }
        } else {
            if (IS_NKM && (block_.matmulTilingData_->matmulTiling.stepN *
                block_.matmulTilingData_->matmulTiling.baseN ==
                block_.matmulTilingData_->matmulTiling.N)) {
                aGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            } else if (!IS_NKM &&
                (block_.matmulTilingData_->matmulTiling.stepM *
                 block_.matmulTilingData_->matmulTiling.baseM ==
                 block_.matmulTilingData_->matmulTiling.M)) {
                bGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
        }'''
assert text.count(before) == 1, 'source cache-hint branch changed'
header.write_text(text.replace(before, after, 1))
