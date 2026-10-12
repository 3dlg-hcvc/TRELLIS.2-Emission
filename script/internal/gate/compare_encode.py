"""Compare two encode-check runs (encode_check.sh): every latent's coords and feats, and the
normalization stats. Exact by default; --envelope <run> <run> ... allows, per measure (feats max
difference, stats max difference), the largest value between any two runs of unchanged code.

  python compare_encode.py --a <run> --b <run> [--envelope <run> <run> ...]

Verdict (exit 0 for the first two): IDENTICAL, WITHIN_NOISE, DIFFERENT, OUTSIDE_NOISE, INCOMPLETE
(a latent missing on one side, coords or shapes differ, a non-finite value), EMPTY.
"""
import argparse, itertools, json, math, os, sys
import numpy as np


class Incomplete(Exception):
    pass


def measure(a, b):
    da, db = (os.path.join(r, "emission_latents", "gate") for r in (a, b))
    na = sorted(f for f in os.listdir(da) if f.endswith(".npz")); nb = sorted(f for f in os.listdir(db) if f.endswith(".npz"))
    if not na:
        raise Incomplete("EMPTY")
    if na != nb:
        raise Incomplete(f"latent sets differ: {sorted(set(na) ^ set(nb))[:5]}")
    feats = 0.0
    for f in na:
        za, zb = np.load(os.path.join(da, f)), np.load(os.path.join(db, f))
        if not np.array_equal(za["coords"], zb["coords"]):
            raise Incomplete(f"{f}: coords differ")
        x, y = za["feats"].astype(np.float64), zb["feats"].astype(np.float64)
        if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise Incomplete(f"{f}: feats shape or non-finite")
        feats = max(feats, float(np.abs(x - y).max()) if x.size else 0.0)
    sa, sb = (json.load(open(os.path.join(d, "normalization_stats.json"))) for d in (da, db))
    if sorted(sa) != sorted(sb):
        raise Incomplete(f"stats fields differ: {sorted(set(sa) ^ set(sb))}")
    stats = 0.0
    for k in sorted(sa):
        if isinstance(sa[k], str) or isinstance(sb[k], str):
            if sa[k] != sb[k]:
                raise Incomplete(f"stats {k}: {sa[k]!r} != {sb[k]!r}")
            continue
        x, y = np.asarray(sa[k], float), np.asarray(sb[k], float)
        if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise Incomplete(f"stats {k}: shape or non-finite")
        stats = max(stats, float(np.abs(x - y).max()))
    return {"feats": feats, "stats": stats}, len(na)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--envelope", nargs="+", default=[])
    args = ap.parse_args()
    try:
        m, n = measure(args.a, args.b)
    except Incomplete as e:
        print(f"VERDICT {'EMPTY' if str(e) == 'EMPTY' else 'INCOMPLETE'}: {e}"); sys.exit(1)
    print(f"latents={n} feats_max_diff={m['feats']:.3e} stats_max_diff={m['stats']:.3e}")
    if m["feats"] == 0 and m["stats"] == 0:
        verdict = "IDENTICAL"
    elif len(args.envelope) < 2:
        verdict = "DIFFERENT"
    else:
        bound = {"feats": 0.0, "stats": 0.0}
        for x, y in itertools.combinations(args.envelope, 2):
            em, _ = measure(x, y)
            bound = {k: max(bound[k], em[k]) for k in bound}
        verdict = "OUTSIDE_NOISE" if any(m[k] > bound[k] for k in m) else "WITHIN_NOISE"
    print(f"VERDICT {verdict}"); sys.exit(0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1)


if __name__ == "__main__":
    main()
