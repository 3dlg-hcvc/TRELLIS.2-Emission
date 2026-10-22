#!/usr/bin/env python3
"""Encode pbr_voxels_<res>/<sha>.vxz into PBR latents using the pretrained
PBR encoder from TRELLIS.2-4B. Parquet-driven, skips existing outputs.

Writes to `<data_root>/pbr_latents/<enc_name>_<res>/<sha>.npz`.

Usage:
    python data_toolkit/encode_pbr_from_voxels.py \
        --data_root <root> \
        --parquet <root>/df_lightgenbench.parquet \
        --resolution 256
"""
from __future__ import annotations
import argparse
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel  # noqa: E402

import trellis2.models as models  # noqa: E402
import trellis2.modules.sparse as sp  # noqa: E402

torch.set_grad_enabled(False)

PBR_ATTRS = ["base_color", "metallic", "roughness", "alpha"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--enc_pretrained", default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    voxel_dir = os.path.join(args.data_root, f"pbr_voxels_{args.resolution}")
    if not os.path.isdir(voxel_dir):
        raise SystemExit(f"missing pbr_voxels_{args.resolution}/ at {voxel_dir}")

    latent_name = f"{args.enc_pretrained.split('/')[-1]}_{args.resolution}"
    save_dir = os.path.join(args.data_root, "pbr_latents", latent_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading PBR encoder: {args.enc_pretrained}")
    # Pretrained weights are fp16 per the ckpt name; force fp32 so it matches
    # fp32 inputs from the .vxz → float() → sparse tensor pipeline (Triton's
    # tl.dot asserts matching dtypes).
    encoder = models.from_pretrained(args.enc_pretrained).eval().float().to(args.device)

    df = pd.read_parquet(args.parquet)
    shas_all = list(df.index.astype(str))
    # Shard by rank: each worker processes a contiguous slice of shas.
    start = len(shas_all) * args.rank // args.world_size
    end = len(shas_all) * (args.rank + 1) // args.world_size
    shas = shas_all[start:end]
    print(f"Samples total: {len(shas_all)}  rank {args.rank}/{args.world_size} → {len(shas)}")

    ok, missing, skipped = 0, 0, 0
    for i, sha in enumerate(shas):
        out_path = os.path.join(save_dir, f"{sha}.npz")
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        vxz = os.path.join(voxel_dir, f"{sha}.vxz")
        if not os.path.exists(vxz):
            missing += 1
            continue

        coords, attr = o_voxel.io.read_vxz(vxz, num_threads=4)
        feats = torch.cat(
            [attr[a].float() / 255.0 * 2.0 - 1.0 for a in PBR_ATTRS], dim=-1
        )
        bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        x = sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1)).to(args.device)
        z = encoder(x)
        torch.cuda.synchronize()

        z_feats = z.feats.cpu().numpy().astype(np.float32)
        z_coords = z.coords[:, 1:].cpu().numpy().astype(np.uint8)
        np.savez_compressed(out_path, feats=z_feats, coords=z_coords)
        ok += 1

        if (i + 1) % 50 == 0 or i == len(shas) - 1:
            print(f"  [{i+1}/{len(shas)}] ok={ok} skipped={skipped} missing={missing}", flush=True)

    print(f"\nDone: ok={ok} skipped={skipped} missing={missing}")
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
