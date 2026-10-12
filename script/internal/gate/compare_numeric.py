"""Numeric comparison of two gate runs, and whether every difference is within the noise of
unchanged code on the same GPU type (REFACTOR_TODO.md, How we work).

  python compare_numeric.py --a <baseline run> --b <candidate run> [--shas_file <shas>]
      [--envelope <run> <run> [<run> ...]] [--csv <per-shape.csv>]

A run is a directory with pred_voxels/<sha>.npz (keys coords, pred, and gt_recon, gt_emission unless
--pred_only). Shapes: --shas_file, else every npz on either side. Per shape it reports byte identity of
the npz, and per key equal coords, then the mean and the max absolute difference. It also scores both
runs' `pred` against the BASELINE's gt_emission with the evaluator's metric functions
(`evaluation/pointsampled/metrics.py`: emission mask at eps 1/255, IoU, MAE, PSNR with its 40 dB cap),
so a metric change reflects the prediction alone. These are voxel-level numbers; the published
evaluator samples mesh surface points, which this script does not do.

The envelope: every pair of the --envelope runs (unchanged code, same GPU type as a and b) is compared
like a candidate; per measure, the largest per-shape value of any pair is the bound. A candidate is
within noise when no shape exceeds the bound in any measure.

NaN and infinity never pass: a non-finite value in an array or a metric makes that measure infinite,
except a metric that is NaN on both sides (MAE of a shape without voxels), which counts as equal.

Verdict (exit 0 for the first two):
  IDENTICAL      every npz byte-identical, nothing missing
  WITHIN_NOISE   nothing missing, every difference within the envelope
  DIFFERENT      differences and no envelope given
  OUTSIDE_NOISE  a shape exceeds the envelope in a measure
  INCOMPLETE     a shape is missing on one side, or a shape's coords or keys differ
  EMPTY          no shape to compare
"""
import argparse
import csv
import itertools
import math
import os
import sys

import numpy as np

REPO = os.environ.get("LIGHTGEN_REPO", "/local-scratch/localhome/dya78/code/lightgen")
sys.path.insert(0, REPO)
from evaluation.pointsampled.metrics import emission_mask, point_iou, point_mae, point_psnr  # noqa: E402

EPS = 1.0 / 255.0
KEYS = ("pred", "gt_recon", "gt_emission")
METRICS = ("iou", "mae", "psnr")
MEASURES = [f"{k}_{s}" for k in KEYS for s in ("mean", "max")] + [f"d{m}" for m in METRICS]


def shapes_of(a, b, shas_file):
    if shas_file:
        with open(shas_file) as f:
            return [s.strip() for s in f if s.strip()]
    names = set()
    for root in (a, b):
        d = os.path.join(root, "pred_voxels")
        names |= {n[:-4] for n in os.listdir(d) if n.endswith(".npz")}
    return sorted(names)


def metrics(pred, gt):
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        return {m: math.inf for m in METRICS}
    pm, gm = emission_mask(pred, EPS), emission_mask(gt, EPS)
    return {"iou": point_iou(pm, gm), "mae": point_mae(pred, gt), "psnr": point_psnr(pred, gt)}


def metric_delta(x, y):
    if math.isnan(x) and math.isnan(y):
        return 0.0
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.inf
    return abs(x - y)


def compare_shape(a, b, sha):
    """Returns (status, row): status 'identical', 'differ' or 'incomplete'."""
    pa, pb = (os.path.join(r, "pred_voxels", sha + ".npz") for r in (a, b))
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        return "incomplete", {"sha": sha, "note": "missing " + ("a" if not os.path.isfile(pa) else "b")}
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        same_bytes = fa.read() == fb.read()
    za, zb = np.load(pa), np.load(pb)
    for z, side in ((za, "a"), (zb, "b")):
        if "coords" not in z.files or "pred" not in z.files:
            return "incomplete", {"sha": sha, "note": f"{side}: keys {sorted(z.files)}"}
        if not all(np.isfinite(z[k]).all() for k in z.files if k != "coords"):
            return "incomplete", {"sha": sha, "note": f"{side}: a non-finite value"}
    if same_bytes:
        return "identical", {"sha": sha, "note": "bytes equal", **{m: 0.0 for m in MEASURES}}
    if set(za.files) != set(zb.files) or "coords" not in za.files or "pred" not in za.files:
        return "incomplete", {"sha": sha, "note": f"keys {sorted(za.files)} vs {sorted(zb.files)}"}
    if za["coords"].shape != zb["coords"].shape or not np.array_equal(za["coords"], zb["coords"]):
        return "incomplete", {"sha": sha, "note": "coords differ"}
    row = {"sha": sha, "note": ""}
    for k in KEYS:
        if k not in za.files:
            row[f"{k}_mean"] = row[f"{k}_max"] = 0.0
            continue
        x, y = za[k].astype(np.float64), zb[k].astype(np.float64)
        if x.shape != y.shape:
            return "incomplete", {"sha": sha, "note": f"{k} shape {x.shape} vs {y.shape}"}
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            row[f"{k}_mean"] = row[f"{k}_max"] = math.inf
            continue
        d = np.abs(x - y)
        row[f"{k}_mean"] = float(d.mean()) if d.size else 0.0
        row[f"{k}_max"] = float(d.max()) if d.size else 0.0
    gt = za["gt_emission"] if "gt_emission" in za.files else None
    if gt is None:
        row.update({f"d{m}": 0.0 for m in METRICS})
        row["note"] = "no gt_emission: metrics skipped"
    else:
        ma, mb = metrics(za["pred"], gt), metrics(zb["pred"], gt)
        for m in METRICS:
            row[f"d{m}"] = metric_delta(ma[m], mb[m])
            row[f"a_{m}"], row[f"b_{m}"] = ma[m], mb[m]
    return "differ", row


def compare_runs(a, b, shas):
    out = {"identical": 0, "differ": 0, "incomplete": 0, "rows": []}
    for sha in shas:
        status, row = compare_shape(a, b, sha)
        out[status] += 1
        row["status"] = status
        out["rows"].append(row)
    return out


def envelope(runs, shas):
    bound = {m: 0.0 for m in MEASURES}
    for r1, r2 in itertools.combinations(runs, 2):
        res = compare_runs(r1, r2, shas)
        if res["incomplete"]:
            sys.exit(f"envelope pair {r1} / {r2} is incomplete: {res['incomplete']} shapes")
        for row in res["rows"]:
            for m in MEASURES:
                if math.isinf(row[m]):
                    sys.exit(f"envelope pair {r1} / {r2}: {row['sha']} {m} is not finite")
                bound[m] = max(bound[m], row[m])
    return bound


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline run")
    ap.add_argument("--b", required=True, help="candidate run")
    ap.add_argument("--shas_file")
    ap.add_argument("--envelope", nargs="+", default=[], help="two or more runs of unchanged code")
    ap.add_argument("--csv")
    args = ap.parse_args()

    shas = shapes_of(args.a, args.b, args.shas_file)
    if not shas:
        print("VERDICT EMPTY: no shape to compare")
        sys.exit(1)
    res = compare_runs(args.a, args.b, shas)
    rows = res["rows"]
    print(f"shapes={len(shas)} identical={res['identical']} differ={res['differ']} incomplete={res['incomplete']}")
    for row in rows:
        if row["status"] == "incomplete":
            print(f"  INCOMPLETE {row['sha']}: {row['note']}")
    diff_rows = [r for r in rows if r["status"] == "differ"]
    if diff_rows:
        for m in MEASURES:
            print(f"  max over shapes {m:18s} {max(r[m] for r in diff_rows):.3e}")
        for m in METRICS:
            a_vals = [r[f"a_{m}"] for r in diff_rows if f"a_{m}" in r]
            b_vals = [r[f"b_{m}"] for r in diff_rows if f"b_{m}" in r]
            if a_vals:
                print(f"  mean {m} over differing shapes: a {np.nanmean(a_vals):.6f}  b {np.nanmean(b_vals):.6f}")
    if args.csv:
        cols = ["sha", "status", "note"] + MEASURES + [f"{s}_{m}" for s in "ab" for m in METRICS]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    if res["incomplete"]:
        verdict = "INCOMPLETE"
    elif not diff_rows:
        verdict = "IDENTICAL"
    elif len(args.envelope) < 2:
        verdict = "DIFFERENT"
    else:
        bound = envelope(args.envelope, shas)
        over = [(r["sha"], m, r[m], bound[m]) for r in diff_rows for m in MEASURES if r[m] > bound[m]]
        for sha, m, v, bnd in over[:10]:
            print(f"  OVER {sha} {m} {v:.3e} > {bnd:.3e}")
        verdict = "OUTSIDE_NOISE" if over else "WITHIN_NOISE"
    print(f"VERDICT {verdict}")
    sys.exit(0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1)


if __name__ == "__main__":
    main()
