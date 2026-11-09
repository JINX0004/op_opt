#!/usr/bin/env python3
"""Limit an isolated diagnostic build to the audited tiling key 65568."""

import sys
from pathlib import Path


path = Path(sys.argv[1])
source = path.read_text()
start = source.index('\n#else\nASCENDC_TPL_SEL(\n    /* Base */')
end = source.index('\n#endif\n\n#endif // MAT_MUL_V3_TILING_KEY_H', start)
selection = '''
#else
// Research build: the four audited shapes all emit key 65568.
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
        ASCENDC_TPL_UINT_SEL(LOADMODE, ASCENDC_TPL_UI_LIST, MAT_MUL_V3_BASE_FULLLOAD),
        ASCENDC_TPL_UINT_SEL(SPLITCOREMODE, ASCENDC_TPL_UI_LIST, MAT_MUL_V3_SINGLE_CORE_SPLIT_K),
        ASCENDC_TPL_UINT_SEL(FIXOPTI, ASCENDC_TPL_UI_LIST, MAT_MUL_V3_BASE_FIXOPTI),
        ASCENDC_TPL_UINT_SEL(MIXND2NZ, ASCENDC_TPL_UI_LIST, MAT_MUL_V3_MIXND2NZ_FALSE),
        ASCENDC_TPL_UINT_SEL(SPECIALOPT, ASCENDC_TPL_UI_LIST, MAT_MUL_V3_K_NOT_SHIFT)), );'''
path.write_text(source[:start] + selection + source[end:])
