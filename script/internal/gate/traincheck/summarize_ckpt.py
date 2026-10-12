"""One JSON line summarizing a Lightning checkpoint the training check saved (appended to its record).

  python summarize_ckpt.py <ckpt>

Recorded: the top-level keys, global_step, epoch; the state_dict's key count and, per prefix (the first
two dotted components, e.g. models.decoder, training_models.decoder), the number of tensors and the
sum of their values; per EMA rate the same for its copy; the sum of every optimizer state tensor; the
lr-scheduler, sampler, elastic-controller and grad-clip states with tensors replaced by their sums; the
callbacks' keys; for the state_dict and each EMA copy also a sha256 over every tensor's key, shape and
dtype (a renamed tensor changes it). Sums are float64.
"""
import json
import sys

import torch


def tsum(t):
    return float(t.detach().double().sum()) if t.numel() else 0.0


def plain(v):
    """A JSON-able copy: tensors become {"sum", "shape"}."""
    if isinstance(v, torch.Tensor):
        return {"sum": tsum(v), "shape": list(v.shape)}
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return repr(type(v).__name__)


def tensor_index(sd):
    """sha256 over every tensor's key, shape and dtype, sorted: a renamed or reshaped tensor changes it."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        if isinstance(t, torch.Tensor):
            h.update(f"{k}|{tuple(t.shape)}|{t.dtype}\n".encode())
    return h.hexdigest()


def prefix_sums(sd):
    out = {}
    for k, t in sd.items():
        if not isinstance(t, torch.Tensor):
            continue
        p = ".".join(k.split(".")[:2])
        n, s = out.get(p, (0, 0.0))
        out[p] = (n + 1, s + tsum(t))
    return {p: {"n": n, "sum": s} for p, (n, s) in sorted(out.items())}


def main():
    path = sys.argv[1]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rec = {"kind": "ckpt", "file": path.rsplit("/", 1)[-1], "keys": sorted(ck.keys()),
           "global_step": ck.get("global_step"), "epoch": ck.get("epoch")}
    sd = ck.get("state_dict", {})
    rec["state_dict"] = {"n": len(sd), "index": tensor_index(sd), "prefixes": prefix_sums(sd)}
    ema = ck.get("ema") or {}
    rec["ema"] = {"rates": ema.get("rates"),
                  "state": {r: {"index": tensor_index(b), "prefixes": prefix_sums(b)} for r, b in (ema.get("state") or {}).items()}}
    opt = []
    for st in ck.get("optimizer_states", []):
        total, n = 0.0, 0
        for ps in st.get("state", {}).values():
            for t in ps.values():
                if isinstance(t, torch.Tensor):
                    total += tsum(t); n += 1
        opt.append({"tensors": n, "sum": total, "param_groups": plain(st.get("param_groups"))})
    rec["optimizer_states"] = opt
    for key in ("lr_schedulers", "sampler_state", "elastic_controller", "grad_clip"):
        if key in ck:
            rec[key] = plain(ck[key])
    rec["callbacks"] = sorted(str(k) for k in (ck.get("callbacks") or {}))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
