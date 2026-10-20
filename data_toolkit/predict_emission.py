#!/usr/bin/env python3
"""Emission DiT inference: the pbr->emission chain (the paper's and the release's).

Runs the emission DiT (in=96, skip_pbr_cond=false) and saves per-shape emission voxels
(coords, pred, and gt_recon, gt_emission unless --pred_only) as npz under <out_dir>/pred_voxels/,
the layout the parent repo's point-sampled evaluator reads (--pred_root, --pred_format voxels).

  * the emission encoder takes the raw 3-channel emission;
  * conditioning is 96 channels: [emission_noise(32), shape(32), pbr(32)], so
    concat_cond = cat([shape, pbr]) = 64 channels;
  * the decoder takes cat([z_emission, z_pbr]) = 64 channels -> 3-channel emission
    (the decoder's concat order is [emission, pbr]; the DiT's cond order is [shape, pbr]:
    different on purpose);
  * the PBR and shape latents are the precomputed ones (tex_enc_ / shape_enc_next_dc_f16c32_fp16_256).
Recipe as trellis2/trainers/vae/emission_vae.py and trellis2/datasets/structured_latent_emission.py
train it.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml
from PIL import Image

for _m in ("o_voxel.postprocess", "o_voxel.rasterize"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
import o_voxel

import trellis2.modules.sparse as sp
from trellis2.trainers.builders import build_models
from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
from torchvision import transforms

torch.set_grad_enabled(False)
DEVICE = "cuda"


def p(*a): print(*a, flush=True)
def normalize(f, m, s):   return (f - m.to(f.device)) / s.to(f.device)
def denormalize(f, m, s): return f * s.to(f.device) + m.to(f.device)


def load_latent(path):
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))


def sample_seed(sha: str, seed: int) -> int:
    """Per-shape RNG seed for `(sha, seed)`. Order-, batch- and skip-independent.

    VERBATIM COPY of evaluation/newdata_eval/seedutil.py in the parent lightgen repo. Duplicated
    because this submodule runs on cluster nodes where the parent is not checked out. Keep the
    copies byte-identical. The value is the FULL unsigned 64-bit prefix, unmasked -- masking to
    int63 gives a different RNG stream for the same nominal seed.
    """
    return int.from_bytes(hashlib.sha256(f"{sha}:{seed}".encode()).digest()[:8], "big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit_ckpt", required=True)
    ap.add_argument("--dit_cfg", default=None)
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--vae_cfg", default=None)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pbr_latent_name", default="tex_enc_next_dc_f16c32_fp16_256")
    ap.add_argument("--shape_latent_name", default="shape_enc_next_dc_f16c32_fp16_256")
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--thumbnail_dir", default=None)
    ap.add_argument("--shas", default=None, help="comma-separated shas (overrides split/parquet)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=str, default="all")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed. Per-shape noise is derived from sha256(sha:seed), so "
                         "it is independent of --shas order, batching and skipped shapes.")
    ap.add_argument("--pred_only", action="store_true",
                    help="write only coords+pred, skipping the gt_recon decode. Use for seeds "
                         "01..04: gt_recon/gt_emission are noise-independent, so seed00's copy "
                         "is the one the ceiling rows are scored from. Do NOT use for seed00 -- "
                         "build_model_gallery.py and the UV viz tools index those keys "
                         "with no guard and raise KeyError.")
    args = ap.parse_args()

    dit_cfg_path = args.dit_cfg or os.path.join(os.path.dirname(args.dit_ckpt), "..", "config.yaml")
    vae_cfg_path = args.vae_cfg or os.path.join(os.path.dirname(args.vae_ckpt), "..", "config.yaml")
    split_json = args.split_json or os.path.join(args.data_root, "data_splits_emissive_74k_stratified_voxonly.json")
    parquet = args.parquet or os.path.join(args.data_root, "df_SomgProc_final.parquet")
    thumb_dir = args.thumbnail_dir or os.path.join(args.data_root, "thumbnails")
    vox_dir = os.path.join(args.data_root, f"emission_voxels_{args.resolution}")
    pbr_dir = os.path.join(args.data_root, "pbr_latents", args.pbr_latent_name)
    shape_dir = os.path.join(args.data_root, "shape_latents", args.shape_latent_name)

    # ---- pick samples ----
    if args.shas:
        shas_list = [s for s in args.shas.split(",") if s]
        n_sel = len(shas_list) if str(args.n) == "all" else int(args.n)
        sel = shas_list[:n_sel]
    else:
        import pandas as pd
        df = pd.read_parquet(parquet, columns=["success"])
        df = df[df["success"] == True] if "success" in df.columns else df
        shas = [str(x) for x in df.index]
        sp_ = json.load(open(split_json))[args.split]
        idx = sp_["indices"] if isinstance(sp_, dict) else sp_
        n_sel = len(idx) if str(args.n) == "all" else int(args.n)
        sel = [shas[i] for i in idx[:n_sel]]
    p(f"Samples: {len(sel)}")

    # ---- configs / norm (THREE streams: emission, shape, pbr) ----
    dit_cfg = yaml.safe_load(open(dit_cfg_path))
    ds = dit_cfg["dataset"]["args"]
    emit_mean = torch.tensor(ds["emission_normalization"]["mean"]); emit_std = torch.tensor(ds["emission_normalization"]["std"])
    shape_mean = torch.tensor(ds["shape_normalization"]["mean"]);   shape_std = torch.tensor(ds["shape_normalization"]["std"])
    pbr_mean = torch.tensor(ds["pbr_normalization"]["mean"]);       pbr_std = torch.tensor(ds["pbr_normalization"]["std"])

    # ---- DINOv3 ----
    p("Loading DINOv3...")
    dino_cfg = dit_cfg["trainer"]["args"]["image_cond_model"]["args"]
    dino = DinoV3FeatureExtractor(**dino_cfg); dino.cuda()
    dino_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def encode_image(sha):
        img = Image.open(os.path.join(thumb_dir, f"{sha}.png")).convert("RGB")
        t = torch.from_numpy(np.array(img.resize((dino_cfg["image_size"],) * 2, Image.LANCZOS)).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        t = dino_tr(t).to(dino.model.embeddings.patch_embeddings.weight.dtype)
        return img, dino.extract_features(t)

    # ---- DiT (in=96) ----
    p(f"Loading DiT: {args.dit_ckpt}")
    dit = build_models(dit_cfg["models"])["denoiser"]
    # mmap=True: these checkpoints are ~20.7 GB and a plain load was OOM-killed on a busy
    # workstation.
    ck = torch.load(args.dit_ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    dit.load_state_dict({k.removeprefix("models.denoiser."): v for k, v in ck.items() if k.startswith("models.denoiser.")})
    dit.to(DEVICE).eval()
    sampler = FlowEulerCfgSampler(sigma_min=dit_cfg["trainer"]["args"]["sigma_min"])

    # ---- pbr→emission VAE encoder + decoder (no separate pbr_encoder model) ----
    p(f"Loading VAE: {args.vae_ckpt}")
    vae = build_models(yaml.safe_load(open(vae_cfg_path))["models"])
    enc, dec = vae["encoder"], vae["decoder"]
    vck = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    enc.load_state_dict({k.removeprefix("models.encoder."): v for k, v in vck.items() if k.startswith("models.encoder.")})
    dec.load_state_dict({k.removeprefix("models.decoder."): v for k, v in vck.items() if k.startswith("models.decoder.")})
    enc.to(DEVICE).eval(); dec.to(DEVICE).eval()

    out_vox_dir = os.path.join(args.out_dir, "pred_voxels")
    os.makedirs(out_vox_dir, exist_ok=True)

    def to01(feats3):
        return (feats3[:, :3] * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()

    n_done = 0
    for k, sha in enumerate(sel):
        vxz = os.path.join(vox_dir, f"{sha}.vxz")
        if not (os.path.exists(vxz) and os.path.exists(os.path.join(pbr_dir, f"{sha}.npz"))
                and os.path.exists(os.path.join(shape_dir, f"{sha}.npz"))
                and os.path.exists(os.path.join(thumb_dir, f"{sha}.png"))):
            p(f"[{k+1}/{len(sel)}] {sha[:16]} SKIP (missing input)"); continue
        p(f"[{k+1}/{len(sel)}] {sha[:16]}...")
        coords, attr = o_voxel.io.read_vxz(vxz, num_threads=4)
        coords_np = coords.int().cpu().numpy()
        gt_emission01 = (attr["emissive"].float() / 255.0).clamp(0, 1).cpu().numpy()  # [N,3] in [0,1]

        bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        gt_feats = attr["emissive"].float() / 255.0 * 2.0 - 1.0  # raw 3-ch emission in [-1,1]
        x_emit = sp.SparseTensor(gt_feats, torch.cat([bc, coords], dim=-1)).to(DEVICE)

        # (A) encode raw 3-ch emission
        z_emit = enc(x_emit)

        # (B) load precomputed pbr + shape latents
        z_pbr = load_latent(os.path.join(pbr_dir, f"{sha}.npz")).to(DEVICE)
        z_shape = load_latent(os.path.join(shape_dir, f"{sha}.npz")).to(DEVICE)
        assert torch.equal(z_emit.coords, z_pbr.coords) and torch.equal(z_emit.coords, z_shape.coords), \
            f"latent coord mismatch {sha}: emit={tuple(z_emit.coords.shape)} pbr={tuple(z_pbr.coords.shape)} shape={tuple(z_shape.coords.shape)}"

        # (C) normalize each stream; concat_cond = [shape, pbr] (64ch)
        z_emit_n = z_emit.replace(normalize(z_emit.feats, emit_mean, emit_std))
        z_shape_n = z_emit.replace(normalize(z_shape.feats, shape_mean, shape_std))
        z_pbr_n = z_emit.replace(normalize(z_pbr.feats, pbr_mean, pbr_std))
        concat_cond = z_emit.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))

        # (D) sample DiT: noise(32) + concat_cond(64) -> [emission_noise, shape, pbr]=96
        _, img_feat = encode_image(sha); neg = torch.zeros_like(img_feat)
        # GLOBAL seed, not a torch.Generator: the parent repo's
        # docs/superpowers/specs/2026-08-15-multi-seed-evaluation-design.md §2.2.
        torch.manual_seed(sample_seed(sha, args.seed))
        noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            res = sampler.sample(dit, noise, cond=img_feat, neg_cond=neg, concat_cond=concat_cond,
                                 steps=args.steps, guidance_strength=args.guidance, verbose=False)

        # (E) denormalize DiT output back to raw emission-latent space
        z_dit = z_emit.replace(denormalize(res.samples.feats, emit_mean.to(DEVICE), emit_std.to(DEVICE)))

        # (F) decode: cat([emission, pbr]) = 64ch -> 3ch, using RAW latents
        dit_dec = dec(z_emit.replace(torch.cat([z_dit.feats, z_pbr.feats], dim=-1)))
        if args.pred_only:
            # Keep the coord assert: it is the ONLY gate that the decoder output aligns
            # element-wise with coords_np.
            assert torch.equal(dit_dec.coords[:, 1:], coords.to(dit_dec.coords.device)), \
                f"decode coord order mismatch {sha}: dit={tuple(dit_dec.coords.shape)} raw={tuple(coords.shape)}"
            np.savez(os.path.join(out_vox_dir, f"{sha}.npz"),
                     coords=coords_np,
                     pred=to01(dit_dec.feats))
        else:
            gt_dec = dec(z_emit.replace(torch.cat([z_emit.feats, z_pbr.feats], dim=-1)))
            assert torch.equal(gt_dec.coords, dit_dec.coords) and \
                   torch.equal(gt_dec.coords[:, 1:], coords.to(gt_dec.coords.device)), \
                f"decode coord order mismatch {sha}: gt={tuple(gt_dec.coords.shape)} dit={tuple(dit_dec.coords.shape)} raw={tuple(coords.shape)}"

            np.savez(os.path.join(out_vox_dir, f"{sha}.npz"),
                     coords=coords_np,
                     pred=to01(dit_dec.feats),
                     gt_recon=to01(gt_dec.feats),
                     gt_emission=gt_emission01)
        n_done += 1
        torch.cuda.empty_cache()

    p(f"\nDONE -> {out_vox_dir}  ({n_done}/{len(sel)} shapes written)")


if __name__ == "__main__":
    main()
