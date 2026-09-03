#!/usr/bin/env python
"""Dump one `eval_vae_recon.py` arm's per-voxel reconstruction as evaluator-ready npz.

`eval_vae_recon.py` answers "how well does each VAE reconstruct emission?" in its own
voxel-space units. This script runs the SAME arm, with the SAME weights and the SAME
deterministic recipe, but writes the per-voxel prediction to disk in the layout the parent
repo's point-sampled evaluator reads -- `<out_dir>/<sha>.npz` with `coords` int32 [N,3] and
`pred` float32 [N,3] in [0,1] -- so the ablation can be reported in the paper's units instead
of a second set of numbers nobody can line up against Table 1.

Nothing about the recipe is reimplemented here: the shape reader, the packing, the weight
loading (strict, fp32, fingerprinted) and the arm closures are imported from eval_vae_recon.
The only thing this file adds is the [-1,1] -> [0,1] cast and the npz write, which mirror
`predict_emission_voxels_twostream.py:to01` byte for byte -- that is what makes the dumped
`pred` comparable with the `gt_recon` key those predictors write for the shipped VAE.

The default arm is `pretrained_pad`: untouched pretrained PBR VAE, emission packed into the
base_color slot with metallic/roughness/alpha at the constants (-1,+1,+1), output channels
0:3 read back as emission. That is the counterpart of the shipped emission VAE's already
scored `pbr_recon_v3_agentic` row.

`--compare_dir` is the guard that matters: the evaluator samples surface points and looks
them up in this coordinate grid, so a dump whose coords differ from the row it is compared
against is not a like-for-like comparison however similar the numbers look. Point it at the
existing score dir and every shape's coords must match exactly.

Run (from TRELLIS2/, conda env `trellis2`):
  conda run --no-capture-output -n trellis2 python data_toolkit/dump_vae_recon_voxels.py \
      --shapes ../evaluation/newdata_eval/eval_sets/test_agentic_clean.txt \
      --out_dir ../evaluation/newdata_eval/out_test/score/pbr_recon_pretrained \
      --compare_dir ../evaluation/newdata_eval/out_test/score/pbr_recon_v3_agentic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

import eval_vae_recon as E


def to01(feats3: torch.Tensor) -> np.ndarray:
    """[-1,1] -> [0,1], clamped, fp32 -- the predictors' `to01`, so the dumps are one format."""
    return (feats3[:, :3] * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", required=True, help="txt, one sha256 per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default="data/lightgen_74k_newbake")
    ap.add_argument("--arm", default="pretrained_pad",
                    choices=["pretrained_pad", "pretrained_realpbr"],
                    help="which pretrained-PBR-VAE packing to dump")
    ap.add_argument("--compare_dir", default=None,
                    help="an existing score dir; every shape's coords must match it exactly")
    ap.add_argument("--pretrained_enc", default=None, help="path prefix; default = HF cache")
    ap.add_argument("--pretrained_dec", default=None)
    a = ap.parse_args()

    torch.manual_seed(0)
    shas = [l.strip() for l in open(a.shapes) if l.strip()]
    if len(set(shas)) != len(shas):
        raise ValueError(f"{a.shapes}: duplicate sha256 entries")
    missing = [s for s in shas
               if not os.path.exists(os.path.join(a.data_root, "pbr_voxels_256", f"{s}.vxz"))
               or not os.path.exists(os.path.join(a.data_root, "emission_voxels_256", f"{s}.vxz"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} shapes lack voxels under {a.data_root}: "
                                f"{missing[:3]} ...")
    os.makedirs(a.out_dir, exist_ok=True)

    enc_prefix = E.resolve_pretrained(E.PRETRAINED_ENC, a.pretrained_enc)
    dec_prefix = E.resolve_pretrained(E.PRETRAINED_DEC, a.pretrained_dec)
    print(f"[data] {len(shas)} shapes from {a.shapes}\n[pretrained] {enc_prefix}\n"
          f"[pretrained] {dec_prefix}")
    device = torch.device("cuda")
    enc, dec, fps = E.load_pretrained_pair(enc_prefix, dec_prefix, device)
    arm = E.make_pbrvae_arms(enc, dec)[a.arm.removeprefix("pretrained_")]

    t0, n_done = time.time(), 0
    for i, sha in enumerate(shas):
        coords, emit, pbr = E.read_shape(a.data_root, sha)
        coords, emit, pbr = coords.to(device), emit.to(device), pbr.to(device)
        with torch.no_grad():
            pred, _target = arm(coords, emit, pbr)
        coords_np = coords[:, 1:].int().cpu().numpy()          # drop the batch column
        pred01 = to01(pred)
        assert pred01.shape == coords_np.shape, (pred01.shape, coords_np.shape)
        if a.compare_dir:
            ref = os.path.join(a.compare_dir, f"{sha}.npz")
            if not os.path.exists(ref):
                raise FileNotFoundError(f"--compare_dir has no {sha}.npz")
            rc = np.load(ref)["coords"]
            if rc.shape != coords_np.shape or not np.array_equal(rc, coords_np):
                raise RuntimeError(
                    f"{sha}: coords differ from {a.compare_dir} "
                    f"({rc.shape} vs {coords_np.shape}). The two dumps would be scored on "
                    f"different grids -- not a like-for-like comparison.")
        np.savez(os.path.join(a.out_dir, f"{sha}.npz"), coords=coords_np, pred=pred01)
        n_done += 1
        if (i + 1) % 50 == 0 or i + 1 == len(shas):
            print(f"  {i + 1}/{len(shas)} shapes, {time.time() - t0:.0f}s")
        torch.cuda.empty_cache()

    json.dump({"arm": a.arm, "shapes_file": os.path.abspath(a.shapes), "n_shapes": n_done,
               "data_root": os.path.abspath(a.data_root),
               "weights": {"encoder": enc_prefix, "decoder": dec_prefix, "fingerprint": fps},
               "compare_dir": os.path.abspath(a.compare_dir) if a.compare_dir else None,
               "value_convention": "pred = clamp((x+1)/2, 0, 1), fp32; coords int32 [N,3]"},
              open(os.path.join(a.out_dir, "dump_manifest.json"), "w"), indent=2, sort_keys=True)
    print(f"\nDONE -> {a.out_dir}  ({n_done}/{len(shas)} npz written)")


if __name__ == "__main__":
    main()
