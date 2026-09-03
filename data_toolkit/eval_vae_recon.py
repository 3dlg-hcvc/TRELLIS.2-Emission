#!/usr/bin/env python
"""Emission reconstruction: our finetuned emission VAEs vs the untouched pretrained PBR VAE.

The question this answers: does finetuning the TRELLIS.2 PBR VAE on emission buy anything
over using the pretrained PBR VAE as-is for emission? Every arm sees the SAME shapes, the
SAME per-voxel targets and the SAME deterministic recipe; only weights and input packing
differ.

The comparison is TWO rows -- `pretrained_pad` vs `ours_pbr2emission` -- and that is what the
default invocation runs and what summary.md shows. The other arms are optional controls
(`--pretrained_arms`, `--sets`, `--arms`), kept for the record.

Arms (one CSV row per arm x shape):

  pretrained_pad           pretrained tex_{enc,dec}_next_dc_f16c32_fp16, emission packed into
                           the base_color slot, metallic/roughness/alpha = constants (-1,+1,+1)
                           -- byte-for-byte the packing EmissionPbrFinetuneTrainer trains with
                           (emission_pbr_finetune.py:_pack_emission_as_pbr). Output channels
                           0:3 are the emission prediction.
  pretrained_realpbr       same weights, same base_color slot, but metallic/roughness/alpha are
                           the shape's REAL PBR -- the pretrained VAE at its least handicapped.
  pretrained_albedo_ctrl   control row: real 6-ch PBR in, reconstruct real ALBEDO. Shows the
                           pretrained VAE is healthy on its own domain, so a gap on emission is
                           an emission gap, not a loading bug.
  ours_pbr2emission        the shipped pbr->emission VAE (agentic generation): 3-ch emission
                           encoder, decoder(cat(z_emission, z_pbr)) with z_pbr from the frozen
                           pretrained PBR encoder. EmissionVaeTrainer semantics.
  ours_albedo2emission_v2  albedo->emission VAE `_v2_cosine`: SAME architecture and SAME
                           packing as pretrained_pad, only the weights differ (284/284 tensors
                           shape-identical to the pretrained files). The clean finetune-only row.
  ours_albedo2emission_v3  albedo->emission `_v3`: pretrained_pad packing, but its decoder is
                           PBR-conditioned (latent_channels 64, z_pbr concat). Secondary row.

Recipe, identical for every arm:
  * deterministic: encoder mean (sample_posterior=False), no dropout, no autocast, fp32 end
    to end (the pretrained files are fp16 on disk; they are cast on load). Verified: two
    independent runs are bit-identical on every metric (2026-08-27, 80 arm x shape pairs);
  * strict weight loading, tensor counts printed -- a silent partial load would invalidate
    the whole table -- and a content fingerprint of every loaded weight set recorded in
    `manifest.json`, which `--resume` must match exactly;
  * per-shape metrics first, then averaged across shapes with equal weight (a 2M-voxel shape
    does not outvote a 20k one); the marginal table is computed on the COMMON set of shapes
    every arm scored, so arms are always compared like for like;
  * PSNR in [0,1] emission space on the clamped prediction, capped at 40 dB like the
    point-sampled evaluator; "lit" = GT max channel > 1/255 (the evaluator's IoU eps); an
    empty GT-lit mask gives NaN L1_lit and the 40 dB cap for PSNR_lit, also as the evaluator.

Numbers here are NOT comparable to the training-time `val/loss`: that one is fp16 autocast +
a sampled posterior + the KL term, and it was measured on the validation split.

Checkpoints resolve to `outputs/<run>/ckpts/<file>` (the local cache) and fall back to the
jupiter copy under /cs/3dlg-jupiter-project/lightgen/outputs/<run>/ when the local one is
absent (mmap over NFS reads only the ~3 GB of model tensors, not the optimizer state).

Run (from TRELLIS2/, conda env `trellis2`):
  conda run --no-capture-output -n trellis2 python data_toolkit/eval_vae_recon.py \
      --shapes ../evaluation/newdata_eval/eval_sets/test_agentic_clean.txt \
      --out_dir reports/vae_recon_test204 2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-eval_vae_recon.log
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import yaml

import o_voxel
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder, SparseUnetVaeEncoder
from trellis2.modules import sparse as sp
from trellis2.trainers.builders import build_models

PBR_ATTRS = ("base_color", "metallic", "roughness", "alpha")  # 3+1+1+1 = 6, dataset order
# metallic=0, roughness=1, alpha=1 in [-1,1] -- EmissionPbrFinetuneTrainer.PAD_{METALLIC,ROUGHNESS,ALPHA}
PAD_MRA = (-1.0, 1.0, 1.0)
EPS_LIT = 1.0 / 255.0  # the point-sampled evaluator's IoU threshold (strict >)
PSNR_CAP = 40.0
PRETRAINED_REPO = "microsoft/TRELLIS.2-4B"
PRETRAINED_ENC = "ckpts/tex_enc_next_dc_f16c32_fp16"
PRETRAINED_DEC = "ckpts/tex_dec_next_dc_f16c32_fp16"
JUPITER_OUTPUTS = "/cs/3dlg-jupiter-project/lightgen/outputs"
METRIC_VERSION = 3  # bump when any metric definition or CSV column changes; recorded in manifest.json

ARM_ORDER = ["pretrained_pad", "pretrained_realpbr", "pretrained_albedo_ctrl",
             "ours_pbr2emission", "ours_albedo2emission_v2", "ours_albedo2emission_v3"]
# What each arm reconstructs. For an "albedo" target the lit / IoU columns are albedo-mask
# quantities (non-black albedo voxels), not emission-mask ones -- the table marks such rows.
ARM_TARGET = {"pretrained_albedo_ctrl": "albedo"}
SETS = {  # set name -> (default run dir, default ckpt file, arms it produces)
    "pretrained": (None, None, ["pretrained_pad", "pretrained_realpbr", "pretrained_albedo_ctrl"]),
    "pbr2emission": ("outputs/emission_vae_pbr2emission_74k_v3_agentic", "ckpts/step0035200-0.0058.ckpt", ["ours_pbr2emission"]),
    "albedo2emission_v2": ("outputs/emission_vae_albedo2emission_74k_v2_cosine", "ckpts/step0034800-0.0056.ckpt", ["ours_albedo2emission_v2"]),
    "albedo2emission_v3": ("outputs/emission_pbr_vae_ft_74k_v3", "ckpts/step0034800-0.0055.ckpt", ["ours_albedo2emission_v3"]),
}

METRIC_COLS = [
    "l1_all", "mae01_all", "psnr_all", "l1_lit", "mae01_lit", "psnr_lit", "iou",
    "n_vox", "n_lit", "lit_frac", "gt_mean01_lit", "pred_mean01_lit", "pred_max01",
]
CSV_COLS = ["arm", "sha256", "status", "target", "seconds"] + METRIC_COLS


# ----------------------------------------------------------------------------- data


def read_shape(data_root: str, sha: str, resolution: int = 256):
    """Read one shape exactly as LightGenSLatEmission.get_instance does, minus the parquet.

    Returns coords [N,4] (batch col 0), emission [N,3] and pbr [N,6] in [-1,1] fp32.
    The emission and PBR grids are asserted to share coordinates -- the dataset relies on
    that ("verified at voxelisation time"); here it is checked because nothing else would.
    """
    p_e = os.path.join(data_root, f"emission_voxels_{resolution}", f"{sha}.vxz")
    p_p = os.path.join(data_root, f"pbr_voxels_{resolution}", f"{sha}.vxz")
    coords_e, attr_e = o_voxel.io.read_vxz(p_e, num_threads=4)
    coords_p, attr_p = o_voxel.io.read_vxz(p_p, num_threads=4)
    if not torch.equal(coords_e, coords_p):
        raise RuntimeError(f"{sha}: emission/PBR voxel coords differ "
                           f"({tuple(coords_e.shape)} vs {tuple(coords_p.shape)})")
    emit = attr_e["emissive"].float() / 255.0 * 2.0 - 1.0
    pbr = torch.cat([attr_p[a].float() / 255.0 * 2.0 - 1.0 for a in PBR_ATTRS], dim=-1)
    assert emit.shape[1] == 3 and pbr.shape[1] == 6, (emit.shape, pbr.shape)
    batch_col = torch.zeros(coords_e.shape[0], 1, dtype=coords_e.dtype)
    coords = torch.cat([batch_col, coords_e], dim=-1)
    return coords, emit, pbr


def pack_pad(emit: torch.Tensor) -> torch.Tensor:
    """[N,3] emission -> [N,6] PBR layout with the trainer's constant metallic/roughness/alpha."""
    pad = torch.tensor([PAD_MRA], device=emit.device, dtype=emit.dtype).expand(emit.shape[0], -1)
    return torch.cat([emit, pad], dim=-1)


def pack_realpbr(emit: torch.Tensor, pbr: torch.Tensor) -> torch.Tensor:
    """[N,3] emission into the base_color slot, the shape's real metallic/roughness/alpha behind it."""
    return torch.cat([emit, pbr[:, 3:6]], dim=-1)


# ----------------------------------------------------------------------------- metrics


def psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return PSNR_CAP
    return min(PSNR_CAP, 10.0 * math.log10(1.0 / mse))


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Per-shape reconstruction metrics. pred/target: [N,3] in [-1,1] (pred unclamped).

    l1_*     mean |pred - target| in [-1,1] space, the training-loss definition (loss_utils.l1_loss)
    mae01_*  mean |clamp(pred01) - target01| in [0,1] space
    psnr_*   10 log10(1/MSE) in [0,1] space on the clamped prediction, capped at PSNR_CAP
    *_lit    the same restricted to GT-lit voxels (max channel > EPS_LIT). With no lit voxel:
             L1/MAE/means are NaN and psnr_lit is PSNR_CAP -- the point-sampled evaluator's
             empty-GT-mask convention (evaluation/pointsampled/pipeline.py).
    iou      voxel IoU of (pred01 max channel > EPS_LIT) vs GT-lit; empty union -> 1.0
    """
    assert pred.shape == target.shape and pred.shape[1] == 3, (pred.shape, target.shape)
    pred = pred.float()
    target = target.float()
    t01 = (target + 1.0) * 0.5
    p01 = ((pred + 1.0) * 0.5).clamp_(0.0, 1.0)

    gt_lit = t01.max(dim=1).values > EPS_LIT
    pr_lit = p01.max(dim=1).values > EPS_LIT
    n = int(target.shape[0])
    n_lit = int(gt_lit.sum().item())
    inter = int((gt_lit & pr_lit).sum().item())
    union = int((gt_lit | pr_lit).sum().item())
    iou = 1.0 if union == 0 else inter / union

    out = {
        "l1_all": (pred - target).abs().mean().item(),
        "mae01_all": (p01 - t01).abs().mean().item(),
        "psnr_all": psnr_from_mse(((p01 - t01) ** 2).mean().item()),
        "iou": iou,
        "n_vox": n,
        "n_lit": n_lit,
        "lit_frac": n_lit / max(1, n),
        "pred_max01": p01.max().item(),
    }
    if n_lit > 0:
        pl, tl, p01l, t01l = pred[gt_lit], target[gt_lit], p01[gt_lit], t01[gt_lit]
        out.update({
            "l1_lit": (pl - tl).abs().mean().item(),
            "mae01_lit": (p01l - t01l).abs().mean().item(),
            "psnr_lit": psnr_from_mse(((p01l - t01l) ** 2).mean().item()),
            "gt_mean01_lit": t01l.mean().item(),
            "pred_mean01_lit": p01l.mean().item(),
        })
    else:
        out.update({"l1_lit": float("nan"), "mae01_lit": float("nan"), "psnr_lit": PSNR_CAP,
                    "gt_mean01_lit": float("nan"), "pred_mean01_lit": float("nan")})
    return out


# ----------------------------------------------------------------------------- models


def fingerprint_state(state: Dict[str, torch.Tensor]) -> str:
    """sha256 over every tensor's bytes in key order: identifies WHICH weights were scored."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _strict_load(module: torch.nn.Module, state: Dict[str, torch.Tensor], label: str) -> str:
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"[load] {label}: missing={len(missing)} unexpected={len(unexpected)} "
                           f"-- refusing a partial load. missing[:3]={list(missing)[:3]} "
                           f"unexpected[:3]={list(unexpected)[:3]}")
    fp = fingerprint_state(state)
    print(f"[load] {label}: {len(state)} tensors, strict OK, sha256 {fp[:16]}")
    return fp


def resolve_pretrained(name: str, explicit: Optional[str]) -> str:
    """Path prefix for `{prefix}.json` / `{prefix}.safetensors`, both verified to exist.
    Offline: the HF cache only."""
    if explicit:
        prefix = explicit
    else:
        from huggingface_hub import try_to_load_from_cache
        f = try_to_load_from_cache(PRETRAINED_REPO, f"{name}.safetensors")
        if not isinstance(f, str):
            raise FileNotFoundError(f"{PRETRAINED_REPO}/{name}.safetensors not in the HF cache; "
                                    f"pass --pretrained_enc/--pretrained_dec explicitly")
        prefix = f[: -len(".safetensors")]
    for ext in (".json", ".safetensors"):
        if not os.path.exists(prefix + ext):
            raise FileNotFoundError(f"pretrained artifact missing: {prefix}{ext}")
    return prefix


def resolve_run_dir(run: str, ckpt_rel: str) -> str:
    """Local cache first, then the jupiter master copy (AGENTS.md: jupiter is the source)."""
    if os.path.exists(os.path.join(run, ckpt_rel)) and os.path.exists(os.path.join(run, "config.yaml")):
        return run
    alt = os.path.join(JUPITER_OUTPUTS, os.path.basename(run.rstrip("/")))
    if os.path.exists(os.path.join(alt, ckpt_rel)) and os.path.exists(os.path.join(alt, "config.yaml")):
        print(f"[resolve] {run}/{ckpt_rel} not local -> reading the jupiter copy {alt}")
        return alt
    raise FileNotFoundError(f"neither {run} nor {alt} holds {ckpt_rel} + config.yaml")


def load_pretrained_pair(enc_prefix: str, dec_prefix: str, device):
    """The pretrained PBR VAE, built fp32 from its own json (use_fp16 forced off) and loaded strictly.

    trellis2.models.from_pretrained loads with strict=False -- not acceptable for a measurement.
    Returns (encoder, decoder, {"encoder": sha256, "decoder": sha256}).
    """
    from safetensors.torch import load_file
    out, fps = [], {}
    for prefix, cls, name in ((enc_prefix, SparseUnetVaeEncoder, "encoder"), (dec_prefix, SparseUnetVaeDecoder, "decoder")):
        cfg = json.load(open(prefix + ".json"))
        assert cfg["name"] == cls.__name__, (cfg["name"], cls.__name__)
        args = dict(cfg["args"])
        args["use_fp16"] = False
        m = cls(**args)
        fps[name] = _strict_load(m, load_file(prefix + ".safetensors"), os.path.basename(prefix))
        out.append(m.float().eval().to(device))
    return out[0], out[1], fps


def load_run_pair(config_path: str, ckpt_path: str, device):
    """A Lightning run's encoder/decoder: architecture from its config.yaml, raw (non-EMA)
    `models.encoder.*` / `models.decoder.*` weights from the checkpoint -- the same tensors
    vis_utils.load_vae_from_handle and the predictors use, so this matches the shipped numbers.
    Returns (encoder, decoder, {"encoder": sha256, "decoder": sha256})."""
    cfg = yaml.safe_load(open(config_path))
    models = build_models(cfg["models"])
    sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    fps = {}
    run_label = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    for name in ("encoder", "decoder"):
        pre = f"models.{name}."
        own = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        fps[name] = _strict_load(models[name], own, f"{run_label}/{name}")
        models[name].float().eval().to(device)
    return models["encoder"], models["decoder"], fps


# ----------------------------------------------------------------------------- arms

ArmFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
# (coords, emit, pbr) -> (pred [N,3] in [-1,1], target [N,3] in [-1,1])


def _encode_mean(enc, coords, feats) -> sp.SparseTensor:
    return enc(sp.SparseTensor(feats, coords), sample_posterior=False, return_raw=False)


def _decode_checked(dec, z: sp.SparseTensor, coords: torch.Tensor) -> torch.Tensor:
    y = dec(z)
    if not torch.equal(y.coords, coords):
        raise RuntimeError(f"decoder output coords != input coords "
                           f"({tuple(y.coords.shape)} vs {tuple(coords.shape)})")
    return y.feats


def make_pbrvae_arms(enc, dec) -> Dict[str, ArmFn]:
    """Arms for a 6-in/6-out PBR-layout VAE (pretrained, or the albedo->emission _v2 finetune)."""
    def pad(coords, emit, pbr):
        return _decode_checked(dec, _encode_mean(enc, coords, pack_pad(emit)), coords)[:, :3], emit

    def realpbr(coords, emit, pbr):
        return _decode_checked(dec, _encode_mean(enc, coords, pack_realpbr(emit, pbr)), coords)[:, :3], emit

    def albedo_ctrl(coords, emit, pbr):
        return _decode_checked(dec, _encode_mean(enc, coords, pbr), coords)[:, :3], pbr[:, :3]

    return {"pad": pad, "realpbr": realpbr, "albedo_ctrl": albedo_ctrl}


def make_pbrcond_arm(enc, dec, pbr_enc, in_packing: str) -> ArmFn:
    """decoder(cat(z_emission, z_pbr)) with z_pbr from the frozen pretrained PBR encoder.

    in_packing: "emission3" (pbr->emission chain: 3-ch encoder) or "pad6" (albedo->emission _v3:
    6-ch encoder with the constant pad). Output channels 0:3 are emission either way.
    """
    def arm(coords, emit, pbr):
        x = emit if in_packing == "emission3" else pack_pad(emit)
        z = _encode_mean(enc, coords, x)
        z_pbr = _encode_mean(pbr_enc, coords, pbr)
        if not torch.equal(z.coords, z_pbr.coords):
            raise RuntimeError("z_emission / z_pbr coord mismatch")
        zc = z.replace(torch.cat([z.feats, z_pbr.feats.to(z.feats.dtype)], dim=-1))
        return _decode_checked(dec, zc, coords)[:, :3], emit
    return arm


# ----------------------------------------------------------------------------- manifest


def load_manifest(path: str) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}


def manifest_check_or_set(manifest: dict, key: str, value, resume: bool, path: str) -> None:
    """On a fresh run, record `value`; on --resume, require it to equal what was recorded.

    This is what makes `--resume` safe: rows keyed (arm, sha256) carry no weight identity,
    so without this a resume against different checkpoints or data would silently mix runs.
    """
    if key in manifest:
        if manifest[key] != value:
            raise RuntimeError(f"[manifest] {key} differs from the recorded run in {path}:\n"
                               f"  recorded: {json.dumps(manifest[key], sort_keys=True)[:300]}\n"
                               f"  now:      {json.dumps(value, sort_keys=True)[:300]}\n"
                               f"Refusing to mix results. Use a fresh --out_dir.")
    elif resume and key.startswith(("shapes", "data_root", "metric_version")):
        raise RuntimeError(f"[manifest] --resume but {path} has no '{key}' entry -- the CSV predates "
                           f"the manifest and cannot be verified; use a fresh --out_dir.")
    else:
        manifest[key] = value
    json.dump(manifest, open(path, "w"), indent=2, sort_keys=True)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- driver


def check_csv_header(csv_path: str) -> None:
    """A resumed CSV must have exactly today's columns: rows are appended positionally."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    if header != CSV_COLS:
        raise RuntimeError(f"{csv_path} header {header} != current CSV_COLS {CSV_COLS}; "
                           f"refusing to append mismatched rows -- use a fresh --out_dir")


def read_done(csv_path: str) -> set:
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    done.add((row["arm"], row["sha256"]))
    return done


def run_set(set_name: str, arms: Dict[str, ArmFn], shas: List[str], data_root: str,
            writer: csv.DictWriter, fh, done: set, device) -> Dict[str, int]:
    todo = [(a, s) for s in shas for a in arms if (a, s) not in done]
    print(f"[{set_name}] {len(arms)} arms x {len(shas)} shapes; {len(todo)} to run, "
          f"{len(arms) * len(shas) - len(todo)} already in CSV")
    n_ok, n_oom, n_err, t0 = 0, 0, 0, time.time()
    for i, sha in enumerate(shas):
        pending = [a for a in arms if (a, sha) not in done]
        if not pending:
            continue
        coords, emit, pbr = read_shape(data_root, sha)
        coords, emit, pbr = coords.to(device), emit.to(device), pbr.to(device)
        for arm in pending:
            t1 = time.time()
            row = {"arm": arm, "sha256": sha, "status": "ok", "target": ARM_TARGET.get(arm, "emission")}
            try:
                with torch.no_grad():
                    pred, target = arms[arm](coords, emit, pbr)
                    row.update(compute_metrics(pred, target))
                n_ok += 1
            except torch.OutOfMemoryError:
                row["status"] = "oom"
                n_oom += 1
                print(f"  [OOM] {arm} {sha} ({coords.shape[0]} voxels)")
            except Exception as e:  # noqa: BLE001 -- recorded, counted, never swallowed silently
                # A poisoned CUDA context would corrupt every later row: re-raise on any GPU-side
                # error (by message) and on a context that no longer synchronizes.
                if any(k in str(e).lower() for k in ("cuda", "cudnn", "cublas", "nvrtc", "triton", "device-side", "illegal memory")):
                    raise
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    raise e
                row["status"] = f"error:{type(e).__name__}"
                n_err += 1
                print(f"  [ERR] {arm} {sha}: {e}")
            row["seconds"] = round(time.time() - t1, 3)
            writer.writerow(row)
            fh.flush()
            torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i + 1 == len(shas):
            print(f"  [{set_name}] {i + 1}/{len(shas)} shapes, {time.time() - t0:.0f}s elapsed")
    return {"ok": n_ok, "oom": n_oom, "error": n_err}


PAIRED_METRICS = ("mae01_all", "mae01_lit", "psnr_lit", "iou")
LOWER_IS_BETTER = {"l1_all", "mae01_all", "l1_lit", "mae01_lit"}


def paired_stats(a: np.ndarray, b: np.ndarray, lower_is_better: bool, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of arm `a` against reference `b` on the SAME shapes.

    delta = a - b per shape; mean delta with a percentile-bootstrap 95% CI over shapes, and
    the fraction of shapes where `a` is strictly better. Rows are paired by shape, so this is
    what actually supports "arm A beats arm B" -- the marginal means alone do not.
    """
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    better = (d < 0) if lower_is_better else (d > 0)
    return {"n": int(len(d)), "mean_delta": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "win_frac": float(better.mean()), "median_delta": float(np.median(d))}


def summarize(csv_path: str, expected_shas: List[str], paired_ref: Optional[str] = None,
              only_arms: Optional[List[str]] = None) -> Tuple[str, dict, List[str]]:
    """Marginal table on the COMMON shapes every arm scored (like for like), paired deltas vs
    a reference arm, and a list of warnings that is persisted with the table.

    Only rows whose sha256 is in `expected_shas` count; duplicates (arm, sha256) keep the last
    row, so a --resume that re-scored a pair supersedes the earlier row.
    """
    expected = set(expected_shas)
    rows = [r for r in csv.DictReader(open(csv_path)) if r["sha256"] in expected]
    ok: Dict[str, Dict[str, dict]] = {}
    bad: Dict[str, set] = {}
    target_of: Dict[str, str] = {}
    for r in rows:  # last row per (arm, sha) wins
        target_of[r["arm"]] = r.get("target") or ARM_TARGET.get(r["arm"], "emission")
        if r["status"] == "ok":
            ok.setdefault(r["arm"], {})[r["sha256"]] = r
            bad.get(r["arm"], set()).discard(r["sha256"])
        else:
            bad.setdefault(r["arm"], set()).add(r["sha256"])
            ok.get(r["arm"], {}).pop(r["sha256"], None)
    seen = set(ok) | set(bad)
    warnings: List[str] = []
    for a in sorted(seen - set(ok)):
        warnings.append(f"{a}: 0 successful rows ({len(bad[a])} failed) -- arm OMITTED from the table")
    arms = [a for a in ARM_ORDER if a in ok] + sorted(set(ok) - set(ARM_ORDER))
    if only_arms:
        arms = [a for a in arms if a in only_arms]
    if not arms:
        return "(no rows)", {}, warnings + ["no successful rows for the expected shapes"]

    common = set.intersection(*(set(ok[a]) for a in arms))
    if len(common) < len(expected):
        warnings.append(f"marginal table uses the {len(common)} shapes ALL arms scored, out of {len(expected)} "
                        f"expected; per-arm successful counts: " + ", ".join(f"{a}={len(ok[a])}" for a in arms))
    for a in arms:
        if bad.get(a):
            warnings.append(f"{a}: {len(bad[a])} shapes failed (OOM/error) and are excluded")

    def col(a, k):
        v = np.array([float(ok[a][s][k]) for s in sorted(common)], dtype=np.float64)
        return v[~np.isnan(v)]

    summary = {"n_expected": len(expected), "n_common": len(common), "arms": {}}
    lines = [f"Marginal means on the {len(common)} shapes every arm scored (mean / median). "
             f"MAE is in [0,1] emission space (the paper's convention); L1 is the same error in the "
             f"[-1,1] training space (= 2x MAE unless the decoder overshoots); PSNR in [0,1], cap {PSNR_CAP:.0f} dB; "
             f"'lit' columns are GT-mask quantities (GT max channel > 1/255), like the evaluator's `_gton` columns.", "",
             "| arm | n | failed | MAE all | L1 [-1,1] all | PSNR all | MAE lit | L1 [-1,1] lit | PSNR lit | IoU | pred/GT mean (lit) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    albedo_rows = False
    for a in arms:
        s = {"n_ok": len(ok[a]), "n_common": len(common), "n_bad": len(bad.get(a, ())), "target": target_of.get(a, "emission")}
        for k in METRIC_COLS:
            v = col(a, k)
            s[k] = {"mean": float(v.mean()) if len(v) else float("nan"),
                    "median": float(np.median(v)) if len(v) else float("nan"), "n": int(len(v))}
        summary["arms"][a] = s
        f = lambda k, fmt: f"{s[k]['mean']:{fmt}} / {s[k]['median']:{fmt}}"
        n_lit_shapes = s["l1_lit"]["n"]  # shapes with at least one GT-lit voxel (NaN otherwise)
        lit_n = f" (n={n_lit_shapes})" if n_lit_shapes != len(common) else ""
        if lit_n:
            warnings.append(f"{a}: lit columns average over {n_lit_shapes} of {len(common)} shapes "
                            f"(the rest have no GT-lit voxel); psnr_lit counts them at the {PSNR_CAP:.0f} dB cap")
        mark = ""
        if s["target"] == "albedo":
            mark, albedo_rows = " †", True
        lines.append(
            f"| {a}{mark} | {len(common)} | {s['n_bad']} | {f('mae01_all', '.4f')} | {f('l1_all', '.4f')} | {f('psnr_all', '.2f')} | "
            f"{f('mae01_lit', '.4f')}{lit_n} | {f('l1_lit', '.4f')} | {f('psnr_lit', '.2f')} | {f('iou', '.3f')} | "
            f"{s['pred_mean01_lit']['mean']:.3f} / {s['gt_mean01_lit']['mean']:.3f} |")
    if albedo_rows:
        lines += ["", "† target = albedo (control row): its lit / IoU / pred-GT-mean columns are ALBEDO-mask "
                      "quantities (non-black albedo voxels ≈ the whole surface), not emission-mask ones."]

    if paired_ref and paired_ref in ok:
        lines += ["", f"Paired vs `{paired_ref}` (same shapes; delta = arm - ref; 95% bootstrap CI over shapes; "
                      f"win = fraction of shapes where the arm is strictly better):", "",
                  "| arm | metric | n | mean delta [95% CI] | median delta | win |", "|---|---|---|---|---|---|"]
        for a in arms:
            if a == paired_ref:
                continue
            pair_common = sorted(set(ok[a]) & set(ok[paired_ref]))
            summary["arms"][a]["paired_vs"] = {"ref": paired_ref, "n_common": len(pair_common)}
            for k in PAIRED_METRICS:
                pa = np.array([float(ok[a][s_][k]) for s_ in pair_common])
                pb = np.array([float(ok[paired_ref][s_][k]) for s_ in pair_common])
                keep = ~(np.isnan(pa) | np.isnan(pb))
                st = paired_stats(pa[keep], pb[keep], k in LOWER_IS_BETTER)
                summary["arms"][a]["paired_vs"][k] = st
                lines.append(f"| {a} | {k} | {st['n']} | {st['mean_delta']:+.4f} [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] "
                             f"| {st['median_delta']:+.4f} | {st['win_frac']:.0%} |")
    return "\n".join(lines), summary, warnings


def write_summary(a, csv_path: str, shas: List[str], totals: dict) -> None:
    only = [x.strip() for x in a.arms.split(",") if x.strip()] if a.arms else None
    table, summary, warnings = summarize(csv_path, shas, paired_ref=a.paired_ref or None, only_arms=only)
    header = (f"eval set: {a.shapes} ({len(shas)} shapes)  |  recipe: encoder mean, fp32, strict load, "
              f"per-shape then mean; MAE in [0,1], L1 in [-1,1], PSNR [0,1] cap {PSNR_CAP:.0f} dB; "
              f"lit = GT max channel > 1/255\n"
              f"runs this invocation: {totals}\n")
    warn_block = "".join(f"\n[WARN] {w}" for w in warnings)
    print("\n" + header + "\n" + table + warn_block)
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write(header + "\n" + table + "\n" + (warn_block + "\n" if warn_block else ""))
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump({"shapes_file": a.shapes, "n_shapes": len(shas), "runs": totals,
                   "warnings": warnings, **summary}, f, indent=2)
    print(f"[out] {csv_path}\n[out] {os.path.join(a.out_dir, 'summary.md')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", required=True, help="txt, one sha256 per line (the eval set)")
    ap.add_argument("--data_root", default="data/lightgen_74k_newbake")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sets", default="pretrained,pbr2emission",
                    help=f"comma list of model sets to run, in order; all: {','.join(SETS)}")
    ap.add_argument("--pretrained_arms", default="pad",
                    help="which pretrained packings to run: comma list of pad,realpbr,albedo_ctrl")
    ap.add_argument("--arms", default="pretrained_pad,ours_pbr2emission",
                    help="arms shown in summary.md ('' = every arm in the CSV)")
    ap.add_argument("--limit", type=int, default=0, help="first N shapes only (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (arm, shape) pairs already ok in the CSV; weights/data must match manifest.json")
    ap.add_argument("--summarize_only", action="store_true", help="no GPU work; rebuild summary.{md,json} from the CSV")
    ap.add_argument("--paired_ref", default="pretrained_pad", help="reference arm for the paired-delta table ('' to skip)")
    ap.add_argument("--pretrained_enc", default=None, help="path prefix; default = HF cache")
    ap.add_argument("--pretrained_dec", default=None)
    for s, (run, ckpt, _) in SETS.items():
        if run:
            ap.add_argument(f"--{s}_run", default=run)
            ap.add_argument(f"--{s}_ckpt", default=ckpt)
    a = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, "per_shape.csv")
    manifest_path = os.path.join(a.out_dir, "manifest.json")

    shas = [l.strip() for l in open(a.shapes) if l.strip()]
    if a.limit:
        shas = shas[: a.limit]
    if len(set(shas)) != len(shas):
        raise ValueError(f"{a.shapes}: duplicate sha256 entries")

    if a.summarize_only:
        write_summary(a, csv_path, shas, {})
        return

    # ---- preflight: every artifact must exist before a single row is written
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    unknown = [s for s in sets if s not in SETS]
    if unknown:
        raise ValueError(f"unknown sets {unknown}; known: {list(SETS)}")
    missing = [s for s in shas if not os.path.exists(os.path.join(a.data_root, "pbr_voxels_256", f"{s}.vxz"))
               or not os.path.exists(os.path.join(a.data_root, "emission_voxels_256", f"{s}.vxz"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} shapes lack voxels under {a.data_root}: {missing[:3]} ...")
    print(f"[data] {len(shas)} shapes from {a.shapes}")
    enc_prefix = resolve_pretrained(PRETRAINED_ENC, a.pretrained_enc)
    dec_prefix = resolve_pretrained(PRETRAINED_DEC, a.pretrained_dec)
    run_dirs = {}
    for s in sets:
        if s != "pretrained":
            run_dirs[s] = resolve_run_dir(getattr(a, f"{s}_run"), getattr(a, f"{s}_ckpt"))

    # ---- manifest: what this out_dir was scored with; --resume must match it exactly
    if not a.resume and os.path.exists(csv_path):
        raise FileExistsError(f"{csv_path} exists; pass --resume to extend it or pick a new --out_dir")
    manifest = load_manifest(manifest_path)
    manifest_check_or_set(manifest, "metric_version", METRIC_VERSION, a.resume, manifest_path)
    manifest_check_or_set(manifest, "shapes", {"file": os.path.abspath(a.shapes), "n": len(shas),
                                               "sha256_of_list": hashlib.sha256("\n".join(shas).encode()).hexdigest()},
                          a.resume, manifest_path)
    manifest_check_or_set(manifest, "data_root", os.path.abspath(a.data_root), a.resume, manifest_path)

    if a.resume:
        check_csv_header(csv_path)
    done = read_done(csv_path) if a.resume else set()
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
    if fh.tell() == 0:
        writer.writeheader()

    device = torch.device("cuda")
    # The pretrained encoder doubles as the frozen z_pbr conditioning encoder of the
    # PBR-conditioned decoders, so it stays resident across sets.
    print(f"[pretrained] {enc_prefix}\n[pretrained] {dec_prefix}")
    pt_enc, pt_dec, pt_fps = load_pretrained_pair(enc_prefix, dec_prefix, device)
    manifest_check_or_set(manifest, "weights/pretrained",
                          {"encoder": enc_prefix, "decoder": dec_prefix, "fingerprint": pt_fps}, a.resume, manifest_path)

    totals = {}
    for set_name in sets:
        if set_name == "pretrained":
            wanted = [x.strip() for x in a.pretrained_arms.split(",") if x.strip()]
            arms = {f"pretrained_{k}": v for k, v in make_pbrvae_arms(pt_enc, pt_dec).items() if k in wanted}
            assert arms, f"--pretrained_arms {a.pretrained_arms!r} matches nothing (pad,realpbr,albedo_ctrl)"
            totals[set_name] = run_set(set_name, arms, shas, a.data_root, writer, fh, done, device)
            continue
        run = run_dirs[set_name]
        ckpt = os.path.join(run, getattr(a, f"{set_name}_ckpt"))
        cfg = os.path.join(run, "config.yaml")
        print(f"[{set_name}] {ckpt}\n[{set_name}] {cfg}")
        enc, dec, fps = load_run_pair(cfg, ckpt, device)
        manifest_check_or_set(manifest, f"weights/{set_name}",
                              {"ckpt": os.path.abspath(ckpt), "config": os.path.abspath(cfg),
                               "config_sha256": file_sha256(cfg), "fingerprint": fps}, a.resume, manifest_path)
        if set_name == "pbr2emission":
            assert enc.in_channels == 3, enc.in_channels
            arms = {"ours_pbr2emission": make_pbrcond_arm(enc, dec, pt_enc, "emission3")}
        elif set_name == "albedo2emission_v2":
            assert enc.in_channels == 6 and dec.from_latent.weight.shape[1] == 32, "expected the 32-ch-latent _v2 decoder"
            arms = {"ours_albedo2emission_v2": make_pbrvae_arms(enc, dec)["pad"]}
        elif set_name == "albedo2emission_v3":
            assert enc.in_channels == 6 and dec.from_latent.weight.shape[1] == 64, "expected the PBR-conditioned 64-ch _v3 decoder"
            arms = {"ours_albedo2emission_v3": make_pbrcond_arm(enc, dec, pt_enc, "pad6")}
        totals[set_name] = run_set(set_name, arms, shas, a.data_root, writer, fh, done, device)
        del arms, enc, dec  # `arms` closures hold enc/dec -- without dropping them nothing is freed
        torch.cuda.empty_cache()
    fh.close()
    write_summary(a, csv_path, shas, totals)


if __name__ == "__main__":
    main()
