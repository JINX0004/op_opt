"""Reviewed NN contract and candidate arithmetic, no NT research dependency."""

from matmul_cost_model._engine.matmul_reviewed_tiling_recipe_v2314 import ops_plan, CAT

NN_TEMPLATES = {
    "base": [
        (16, 16, 4),
        (32, 64, 1),
        (32, 64, 4),
        (32, 64, 8),
        (64, 32, 4),
        (128, 128, 4),
    ],
    "scsk": [
        (16, 16, 4),
        (32, 64, 2),
        (32, 64, 4),
        (32, 64, 8),
        (64, 32, 4),
        (128, 128, 4),
    ],
    "deterministic": [(128, 128, 1), (128, 128, 2), (128, 128, 3)],
}


def validate_nn(r):
    shape = r.get("shape", [])
    kind = r.get("implementation")
    c = r.get("cores")
    if (
        len(shape) != 3
        or any(type(x) is not int or x % 16 for x in (shape[0], shape[2]))
        or not (
            16 <= shape[0] <= 128
            and shape[1] in (16, 24, 32, 64, 96, 128)
            and type(shape[1]) is int
            and 64 <= shape[2] <= 131072
        )
    ):
        raise ValueError("outside reviewed NN research domain")
    if (
        r.get("dtype", r.get("plan", {}).get("dtype", "float16")) != "float16"
        or r.get("trans_a")
        or r.get("trans_b")
        or r.get("batch", 1) != 1
    ):
        raise ValueError("reviewed FP16 NN batch1 only")
    if type(c) is not int or not 1 <= c <= 24:
        raise ValueError("cores must be integer1..24")
    if r.get("source_a_offset_extra_bytes", 0) or r.get(
        "source_b_offset_extra_bytes", 0
    ):
        raise ValueError("extra source offsets unverified by this API")
    if r.get("execution_profile", "warm400_gate_targetC") != "warm400_gate_targetC":
        raise ValueError("different execution profile unverified")
    if kind == "catlass":
        tile = r.get("tile")
        if (
            tile not in CAT
            or (list(r.get("l1_tile", [])), list(r.get("l0_tile", []))) != CAT[tile]
        ):
            raise ValueError("unreviewed CAT tile")
    elif kind in NN_TEMPLATES:
        p = r.get("plan", {})
        t = p.get("tcube", {})
        recipe = (t.get("baseM"), t.get("baseN"), t.get("stepKa"))
        if recipe not in NN_TEMPLATES[kind] or t.get("baseK") != 128:
            raise ValueError("unreviewed NN native template")
        bm, bn, step = recipe
        if kind == "deterministic" and (
            c < 2 or c > (shape[2] + 128 * step - 1) // (128 * step)
        ):
            raise ValueError("illegal deterministic split count")
        expected = ops_plan(shape, ("reviewed", kind, bm, bn, 128, step, c))
        for key in (
            "m",
            "n",
            "k",
            "dtype",
            "trans_a",
            "trans_b",
            "tcube",
            "raw_words",
            "nd2nz_a",
            "nd2nz_b",
            "block_dim",
        ):
            if p.get(key) != expected[key]:
                raise ValueError("inconsistent/unreviewed NN plan field: " + key)
    else:
        raise ValueError("unknown implementation")


def candidates(implementation, shape, cores, trans_b=False):
    if trans_b:
        raise ValueError("NT is outside this standalone release")
    rows = []
    for c in cores:
        if implementation == "catlass":
            for tile, (l1, l0) in CAT.items():
                rows.append(
                    dict(
                        implementation=implementation,
                        shape=list(shape),
                        cores=c,
                        dtype="float16",
                        trans_a=False,
                        trans_b=False,
                        tile=tile,
                        l1_tile=list(l1),
                        l0_tile=list(l0),
                    )
                )
        else:
            if implementation not in NN_TEMPLATES:
                raise ValueError("unknown implementation")
            for bm, bn, step in NN_TEMPLATES[implementation]:
                if implementation == "deterministic" and (
                    c < 2 or c > (shape[2] + 128 * step - 1) // (128 * step)
                ):
                    continue
                rows.append(
                    dict(
                        implementation=implementation,
                        shape=list(shape),
                        cores=c,
                        dtype="float16",
                        trans_a=False,
                        trans_b=False,
                        plan=ops_plan(
                            shape, ("reviewed", implementation, bm, bn, 128, step, c)
                        ),
                    )
                )
    for row in rows:
        validate_nn(row)
    return rows
