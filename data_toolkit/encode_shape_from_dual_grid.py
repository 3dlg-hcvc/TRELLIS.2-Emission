#!/usr/bin/env python3
"""Encode dual_grid_<res>/<sha>.vxz into shape latents using the pretrained
shape encoder from TRELLIS.2-4B. Parquet-driven (no metadata.csv needed),
skips already-encoded outputs.

Writes to `<data_root>/shape_latents/<enc_name>_<res>/<sha>.npz`.

Usage:
    python data_toolkit/encode_shape_from_dual_grid.py \
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--enc_pretrained", default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    dual_grid_dir = os.path.join(args.data_root, f"dual_grid_{args.resolution}")
    if not os.path.isdir(dual_grid_dir):
        raise SystemExit(f"missing dual_grid_{args.resolution}/ at {dual_grid_dir}")

    latent_name = f"{args.enc_pretrained.split('/')[-1]}_{args.resolution}"
    save_dir = os.path.join(args.data_root, "shape_latents", latent_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading shape encoder: {args.enc_pretrained}")
    # Force fp32 so it matches fp32 inputs from .vxz → float() (pretrained
    # weights are fp16 per the ckpt name; Triton tl.dot asserts same dtype).
    encoder = models.from_pretrained(args.enc_pretrained).eval().float().to(args.device)

    df = pd.read_parquet(args.parquet)
    shas_all = list(df.index.astype(str))
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

        vxz = os.path.join(dual_grid_dir, f"{sha}.vxz")
        if not os.path.exists(vxz):
            missing += 1
            continue

        coords, attr = o_voxel.io.read_vxz(vxz, num_threads=4)
        # Match encode_shape_latent.py loader exactly: vertices + intersected
        # go in as two separate SparseTensors; encoder signature is (v, inter).
        bc = torch.zeros_like(coords[:, 0:1])
        coords4 = torch.cat([bc, coords], dim=-1)
        vertices = sp.SparseTensor(
            (attr["vertices"] / 255.0).float(),
            coords4,
        )
        intersected = vertices.replace(
            torch.cat(
                [
                    attr["intersected"] % 2,
                    attr["intersected"] // 2 % 2,
                    attr["intersected"] // 4 % 2,
                ],
                dim=-1,
            ).bool()
        )
        z = encoder(vertices.to(args.device), intersected.to(args.device))
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
