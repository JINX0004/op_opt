def validate_row(r):
    if not isinstance(r, dict):
        raise ValueError('each static input must be an object')
    shape = r.get('shape', [])
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('shape requires three positive integers')
    if shape[0] > 128 or shape[1] > 128 or shape[2] % 16:
        raise ValueError('current research scope M,N<=128 and K multiple of16')
    if r.get('dtype','float16') not in ('fp16','float16'):
        raise ValueError('unsupported top-level dtype')
    if r.get('trans_a',False) or r.get('trans_b',False) or r.get('batch',1) != 1:
        raise ValueError('transpose and batch are unvalidated')
    if r.get('implementation') not in ('catlass','base','scsk','deterministic'):
        raise ValueError('unknown implementation')
    if type(r.get('cores')) is not int or not 1 <= r['cores'] <= 24:
        raise ValueError('cores must be an integer in1..24')
    if r['implementation'] != 'catlass':
        p = r.get('plan', {})
        if p.get('dtype') != 'float16' or p.get('trans_a') or p.get('trans_b'):
            raise ValueError('native plan must declare float16 NN')
        if [p.get(k) for k in ('m','n','k')] != shape:
            raise ValueError('native plan and shape disagree')
        if r['implementation'] == 'deterministic' and r['cores'] < 2:
            raise ValueError('single-partial DET is unvalidated')
