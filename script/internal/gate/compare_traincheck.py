"""Compare two training-check records (train-check.sh), optionally against the noise of unchanged code.

  python compare_traincheck.py --a <record.jsonl> --b <record.jsonl> [--envelope <rec> <rec> [<rec> ...]]

Matched in order: the training steps (global step, epoch and the batch's sha256 list must be equal;
measured: lr, loss, grad_norm, wsum, and the first step's loss on its own), the validations (global step equal; measured: every val/* metric,
every EMA copy's sum), the checkpoint summary (keys, steps, tensor counts and every string must be equal;
measured: every number in it) and the log record (the SAVE_START and RESUME_AFTER_SAVE step lists must be
equal). A measure is the absolute difference; NaN or infinity on one side is an infinite difference, on
both sides none. The envelope is, per measure, the largest difference between any pair of the given
records of unchanged code on the same GPU type. With an envelope the candidate (--b) is treated as one
more run: per measure its distance to the NEAREST envelope run must not exceed the envelope. (Measured
against --a alone, a fifth unchanged run of the DiT, whose flash-attention backward drifts chaotically,
often lands slightly outside four runs' largest pairwise difference.) The first step's loss must match
exactly in any case.

Verdict (exit 0 for the first two): IDENTICAL, WITHIN_NOISE, DIFFERENT (differences, no envelope),
OUTSIDE_NOISE, INCOMPLETE (a record or a matched field missing or unequal), EMPTY (no training step).
"""
import argparse
import itertools
import json
import math
import sys


class Incomplete(Exception):
    pass


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    by = {k: [r for r in recs if r["kind"] == k] for k in ("train", "val", "ckpt", "log", "fit_end")}
    validate(path, by)
    return by


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(path, by):
    """A record that cannot be a complete, sane run is INCOMPLETE, whatever the other side holds."""
    t = by["train"]
    if not t:
        raise Incomplete(f"{path}: no training step")
    gs = [r.get("gs") for r in t]
    if gs != list(range(gs[0], gs[0] + len(gs))):
        raise Incomplete(f"{path}: training steps not consecutive: {gs[:3]}...{gs[-3:]}")
    for r in t:
        for f in ("lr", "loss", "wsum"):
            if not finite(r.get(f)):
                raise Incomplete(f"{path}: step {r.get('gs')}: {f}={r.get(f)!r}")
        # grad_norm may be inf: under fp16 AMP an overflowing step has infinite gradients, which the
        # GradScaler skips (the VAE's step 3); it must be a number, and inf on one side only differs.
        g = r.get("grad_norm")
        if not isinstance(g, (int, float)) or isinstance(g, bool) or math.isnan(g):
            raise Incomplete(f"{path}: step {r.get('gs')}: grad_norm={g!r}")
        if not r.get("shas"):
            raise Incomplete(f"{path}: step {r.get('gs')}: no sha256 list")
    if len(by["fit_end"]) != 1 or by["fit_end"][0].get("gs") != gs[-1]:
        raise Incomplete(f"{path}: fit_end {by['fit_end']} does not close at step {gs[-1]}")
    if len(by["ckpt"]) != 1 or by["ckpt"][0].get("global_step") != gs[-1]:
        raise Incomplete(f"{path}: the checkpoint summary is not of step {gs[-1]}")
    if not by["val"] or len(by["log"]) != 1:
        raise Incomplete(f"{path}: no validation or no log record")


def delta(x, y):
    if x is None and y is None:
        return 0.0
    if x is None or y is None:
        return math.inf
    x, y = float(x), float(y)
    if math.isnan(x) and math.isnan(y):
        return 0.0
    if not (math.isfinite(x) and math.isfinite(y)):
        return 0.0 if x == y else math.inf
    return abs(x - y)


def walk(a, b, path, out):
    """Numbers -> out[path] = delta; everything else must be equal."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        if a != b:
            raise Incomplete(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out[path] = max(out.get(path, 0.0), delta(a, b))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a) != sorted(b):
            raise Incomplete(f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}")
        for k in a:
            walk(a[k], b[k], f"{path}.{k}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise Incomplete(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
        return
    raise Incomplete(f"{path}: types {type(a).__name__} / {type(b).__name__}")


def compare(ra, rb):
    """Returns {measure: max delta}; raises Incomplete."""
    a, b = load(ra), load(rb)
    for k in ("train", "val", "ckpt", "log", "fit_end"):
        if len(a[k]) != len(b[k]):
            raise Incomplete(f"{k}: {len(a[k])} vs {len(b[k])} records")
    m = {}
    for x, y in zip(a["train"], b["train"]):
        for f in ("gs", "epoch", "shas"):
            if x.get(f) != y.get(f):
                raise Incomplete(f"train step {x.get('gs')}: {f} {x.get(f)} != {y.get(f)}")
        for f in ("lr", "loss", "grad_norm", "wsum"):
            m[f"train.{f}"] = max(m.get(f"train.{f}", 0.0), delta(x.get(f), y.get(f)))
    # The first step's loss on its own: the forward pass is deterministic, so unchanged code repeats it
    # exactly even where the backward pass (the DiT's flash-attention) does not; its envelope is 0.
    m["train.loss.first"] = delta(a["train"][0].get("loss"), b["train"][0].get("loss"))
    for x, y in zip(a["val"], b["val"]):
        if x["gs"] != y["gs"]:
            raise Incomplete(f"val at gs {x['gs']} vs {y['gs']}")
        walk(x["metrics"], y["metrics"], "val.metrics", m)
        walk(x["ema"], y["ema"], "val.ema", m)
    for x, y in zip(a["ckpt"], b["ckpt"]):
        walk({k: v for k, v in x.items() if k != "file"}, {k: v for k, v in y.items() if k != "file"}, "ckpt", m)
    for x, y in zip(a["log"], b["log"]):
        if x != y:
            raise Incomplete(f"log records differ: {x} vs {y}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--envelope", nargs="+", default=[])
    args = ap.parse_args()
    try:
        m = compare(args.a, args.b)
    except Incomplete as e:
        print(f"INCOMPLETE: {e}")
        print("VERDICT INCOMPLETE" if "no training step" not in str(e) else "VERDICT EMPTY")
        sys.exit(1)
    n_train = len(load(args.a)["train"])
    nonzero = {k: v for k, v in m.items() if v != 0.0}
    print(f"train steps={n_train} measures={len(m)} nonzero={len(nonzero)}")
    for k, v in sorted(nonzero.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k:60s} {v:.3e}")
    if not nonzero:
        verdict = "IDENTICAL"
    elif m.get("train.loss.first", 0.0) != 0.0:
        print("  the first step's loss differs: the forward pass, the data or the loaded state changed")
        verdict = "OUTSIDE_NOISE" if len(args.envelope) >= 2 else "DIFFERENT"
    elif len(args.envelope) < 2:
        verdict = "DIFFERENT"
    else:
        bound = {}
        for x, y in itertools.combinations(args.envelope, 2):
            try:
                em = compare(x, y)
            except Incomplete as e:
                sys.exit(f"envelope pair {x} / {y} incomplete: {e}")
            for k, v in em.items():
                if math.isinf(v):
                    sys.exit(f"envelope pair {x} / {y}: {k} is not finite")
                bound[k] = max(bound.get(k, 0.0), v)
        if bound.get("train.loss.first", 0.0) != 0.0:
            sys.exit("envelope runs differ in the first step's loss: unchanged code must repeat it exactly")
        nearest = dict(m)
        for e in args.envelope:
            try:
                em = compare(e, args.b)
            except Incomplete as exc:
                sys.exit(f"candidate vs envelope run {e} incomplete: {exc}")
            nearest = {k: min(v, em.get(k, v)) for k, v in nearest.items()}
        over = [(k, v, bound.get(k, 0.0)) for k, v in nearest.items() if v > bound.get(k, 0.0)]
        for k, v, bnd in over[:10]:
            print(f"  OVER {k} {v:.3e} > {bnd:.3e}")
        verdict = "OUTSIDE_NOISE" if over else "WITHIN_NOISE"
    print(f"VERDICT {verdict}")
    sys.exit(0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1)


if __name__ == "__main__":
    main()
