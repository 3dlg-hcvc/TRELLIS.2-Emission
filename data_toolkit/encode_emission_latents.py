#!/usr/bin/env python3
"""Encode emission voxels to 32-ch latents using the TWO-STREAM emission VAE.

The two-stream VAE (EmissionVaeTrainer) has a dedicated emission encoder with
in_channels=3, so the raw 3-ch emission SparseTensor goes in directly (no pack-as-PBR
step; that was the deleted albedo->emission VAE's, whose encoder took 6 PBR channels).

The encoder weights are the `models.encoder.*` entries of the two-stream VAE's
Lightning checkpoint (the trainable z_emission encoder; the frozen PBR encoder
and the decoder are not needed to produce z_emission, which is what the DiT
learns to generate). At inference the DiT's z_emission is decoded with
concat(z_emission, z_pbr) through the trained decoder, so the latents MUST come
from this exact encoder.

Emission normalization stats are written to
`<data_root>/emission_latents/<latent_name>/normalization_stats.json` and printed
for pasting into the DiT config.

Usage:
    python data_toolkit/encode_emission_latents.py \
        --data_root <root> \
        --parquet <root>/df_lightgenbench.parquet \
        --vae_ckpt outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/step0035400-0.0036.ckpt \
        --vae_cfg  outputs/emission_vae_pbr2emission_lightgenbench_v1/config.yaml \
        --latent_name emission_pbr2emission_lightgenbench_v1 \
        --resolution 256
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import yaml

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import o_voxel  # noqa: E402

import trellis2.modules.sparse as sp  # noqa: E402
from trellis2.trainers.builders import build_models  # noqa: E402

torch.set_grad_enabled(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="dataset root (expects emission_voxels_<res>/ and will write emission_latents/<name>/)")
    ap.add_argument("--parquet", required=True, help="parquet with sha256 index")
    ap.add_argument("--vae_ckpt", required=True, help="Lightning .ckpt path from the two-stream VAE training")
    ap.add_argument("--vae_cfg", required=True, help="the config.yaml saved next to the ckpt")
    ap.add_argument("--latent_name", required=True,
                    help="subdir name under emission_latents/ (also used in DiT config)")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--aggregate_stats", action="store_true", default=False,
                    help="After encoding my shard, re-read ALL shards' .npz and write normalization_stats.json; only set on rank 0")
    args = ap.parse_args()

    voxel_dir = os.path.join(args.data_root, f"emission_voxels_{args.resolution}")
    save_dir = os.path.join(args.data_root, "emission_latents", args.latent_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading two-stream VAE encoder from {args.vae_ckpt}")
    with open(args.vae_cfg) as f:
        vae_cfg = yaml.safe_load(f)
    models = build_models(vae_cfg["models"])
    encoder = models["encoder"]
    in_ch = vae_cfg["models"]["encoder"]["args"].get("in_channels")
    assert in_ch == 3, (
        f"two-stream emission encoder must have in_channels=3 (raw emission); got {in_ch}. "
        f"(The 6-ch albedo->emission VAE and its encoder script were deleted in the refactor.)"
    )
    ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    enc_state = {
        k.removeprefix("models.encoder."): v
        for k, v in ckpt["state_dict"].items() if k.startswith("models.encoder.")
    }
    encoder.load_state_dict(enc_state)
    encoder.to(args.device).eval()
    print(f"  encoder params: {sum(p.numel() for p in encoder.parameters()):,}  "
          f"(global_step={ckpt.get('global_step')})")

    df = pd.read_parquet(args.parquet)
    shas_all = list(df.index.astype(str))
    start = len(shas_all) * args.rank // args.world_size
    end = len(shas_all) * (args.rank + 1) // args.world_size
    shas = shas_all[start:end]
    print(f"Samples in parquet: {len(shas_all)}  rank {args.rank}/{args.world_size} → {len(shas)}")

    # NOTE: the encode loop deliberately accumulates NOTHING. An earlier version
    # appended every shape's feats to an `all_feats` list here, but that list is
    # provably dead: a non-aggregating rank returns and discards it, and the
    # --aggregate_stats path resets it to [] and re-reads every .npz from disk.
    # It was pure leak -- ~16 GB resident per rank after only ~5k shapes, which
    # fir's 512 GB nodes absorbed but which the 62 GB workstation could not: the
    # kernel OOM-killed rank 0 during startup on 2026-08-24. Removing it changes
    # no output, because the stats come from the disk re-read below.
    processed, skipped, missing = 0, 0, 0
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
        # Raw 3-ch emission in [-1, 1] — fed directly to the in_channels=3 encoder.
        emit_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0
        bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        x = sp.SparseTensor(emit_feats, torch.cat([bc, coords], dim=-1)).to(args.device)
        z = encoder(x)
        torch.cuda.synchronize()

        z_feats = z.feats.cpu().numpy().astype(np.float32)
        z_coords = z.coords[:, 1:].cpu().numpy().astype(np.uint8)
        np.savez_compressed(out_path, feats=z_feats, coords=z_coords)
        processed += 1

        if (i + 1) % 50 == 0 or i == len(shas) - 1:
            print(f"  [{i+1}/{len(shas)}] processed={processed} skipped={skipped} missing={missing}")

    print(f"\nRank {args.rank} done: processed={processed} skipped={skipped} missing={missing}")

    # Stats computed once (on --aggregate_stats worker) by re-reading every .npz
    # in save_dir — handles sharded runs where no single worker saw all samples.
    if not args.aggregate_stats:
        return
    print("Aggregating stats across all shards...")
    all_feats = []
    for p in sorted(glob.glob(os.path.join(save_dir, "*.npz"))):
        d = np.load(p)
        all_feats.append(torch.from_numpy(d["feats"]).float())
    if not all_feats:
        print("No latents to aggregate; aborting.")
        sys.exit(1)
    all_f = torch.cat(all_feats, dim=0)
    mean, std = all_f.mean(dim=0), all_f.std(dim=0)
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "total_tokens": int(all_f.shape[0]),
        "channels": int(all_f.shape[1]),
        "feat_min": float(all_f.min()),
        "feat_max": float(all_f.max()),
        "n_samples": len(all_feats),
    }
    stats_path = os.path.join(save_dir, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote stats to {stats_path}")
    print("\n=== Emission normalization stats (paste into DiT config) ===")
    print("    emission_normalization:")
    print(f"      mean: [{', '.join(f'{v:.6f}' for v in mean.tolist())}]")
    print(f"      std: [{', '.join(f'{v:.6f}' for v in std.tolist())}]")


if __name__ == "__main__":
    main()
