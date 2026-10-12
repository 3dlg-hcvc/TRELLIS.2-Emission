#!/usr/bin/env python3
"""Unified 3D interactive visualization for DiT → VAE-decode emission.

Usage:
  python vis/vis_3d_dit.py \\
      --dataset <root>/<index>.parquet \\
      --samples all \\
      --glb_root <dir with the GLBs> \\
      --experiment "pbr→emission=outputs/<dit run>/ckpts/<file>.ckpt:outputs/<vae run>/ckpts/<file>.ckpt" \\
      --out visualizations/dit.html
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np   # noqa: E402
import torch         # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

import o_voxel  # noqa: E402

from vis.vis_utils import (  # noqa: E402
    ExperimentHandle, compute_emission_l1, load_samples, make_glb_panel_data,
    make_voxel_panel_data, read_emission_voxel, render_html, resolve_experiment,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_latent(path: str):
    from trellis2.modules import sparse as sp
    d = np.load(path)
    feats = torch.from_numpy(d["feats"]).float()
    coords = torch.from_numpy(d["coords"].astype(np.int32)).int()
    bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return sp.SparseTensor(feats, torch.cat([bc, coords], dim=-1))


def _parse_experiments(specs: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for s in specs:
        if "=" not in s or ":" not in s.split("=", 1)[1]:
            raise ValueError(f"--experiment expects name=dit_path:vae_path, got {s!r}")
        name, pair = s.split("=", 1)
        dit_path, vae_path = pair.split(":", 1)
        out.append((name.strip(), dit_path.strip(), vae_path.strip()))
    return out


def _resolve_data_root(dataset_arg: str, experiments: list[ExperimentHandle], override: str | None) -> str:
    """Same logic as vis_3d_vae._resolve_data_root. See that file for the rationale."""
    if override:
        return os.path.abspath(override)
    ds_parent = os.path.dirname(os.path.abspath(dataset_arg))
    if experiments:
        res = experiments[0].resolution
        if os.path.isdir(os.path.join(ds_parent, f"emission_voxels_{res}")):
            return ds_parent
    elif os.path.isdir(os.path.join(ds_parent, "emission_voxels_256")):
        return ds_parent
    if experiments and os.path.isdir(experiments[0].data_root):
        return experiments[0].data_root
    return ds_parent


def _build_dit(dit_handle, sigma_min: float):
    from trellis2.trainers.builders import build_models
    models = build_models(dit_handle.config["models"])
    dit = models["denoiser"]
    state = torch.load(dit_handle.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    dit.load_state_dict({
        k.removeprefix("models.denoiser."): v for k, v in state.items()
        if k.startswith("models.denoiser.")
    })
    dit.to(DEVICE).eval()
    from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
    return dit, FlowEulerCfgSampler(sigma_min=sigma_min)


def _build_dino(dino_args):
    from trellis2.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor
    dino = DinoV3FeatureExtractor(**dino_args)
    dino.cuda()
    return dino


def _build_vae_decoder_from_latent(vae_handle):
    """Build the VAE encoder+decoder and return a callable
    ``decode_from_latent(x_gt, z_dit_raw, z_pbr_raw) -> [N, 3] cpu tensor``.

    The encoder is needed to prime the decoder's subdivision cache (the sparse
    spatial2channel/channel2spatial ops pair encode→decode state). Legacy:
    archive/vis/legacy/vis_3d_emission_dit.py:191 in tag pre-refactor (``z_enc = enc(x_gt)``).
    """
    from trellis2.trainers.builders import build_models
    models = build_models(vae_handle.config["models"])
    enc = models["encoder"]
    dec = models["decoder"]
    state = torch.load(vae_handle.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    enc.load_state_dict({
        k.removeprefix("models.encoder."): v for k, v in state.items()
        if k.startswith("models.encoder.")
    })
    dec.load_state_dict({
        k.removeprefix("models.decoder."): v for k, v in state.items()
        if k.startswith("models.decoder.")
    })
    enc.to(DEVICE).eval()
    dec.to(DEVICE).eval()

    trainer = vae_handle.trainer_name
    if trainer == "EmissionVaeTrainer":
        @torch.no_grad()
        def decode_from_latent(x_gt, z_dit_raw, z_pbr_raw):
            z_enc = enc(x_gt)  # primes subdivision cache
            z_joint = z_enc.replace(
                torch.cat([z_dit_raw.feats, z_pbr_raw.feats.to(DEVICE)], dim=-1)
            )
            return dec(z_joint).to("cpu").feats
    elif trainer == "EmissionPbrFinetuneTrainer":
        @torch.no_grad()
        def decode_from_latent(x_gt, z_dit_raw, z_pbr_raw):
            # The albedo→emission trainer encodes emission-as-PBR; prime with the
            # same packed input the trainer uses.
            from trellis2.modules import sparse as sp
            packed = _pack_emission_as_pbr(x_gt.feats)
            bc = torch.zeros(x_gt.coords.shape[0], 1, dtype=x_gt.coords.dtype, device=x_gt.coords.device)
            x_packed = sp.SparseTensor(packed, x_gt.coords).to(DEVICE) if x_gt.coords.shape[1] == 4 else \
                       sp.SparseTensor(packed, torch.cat([bc, x_gt.coords], dim=-1)).to(DEVICE)
            z_enc = enc(x_packed)  # primes cache
            z_sub = z_enc.replace(z_dit_raw.feats)
            return dec(z_sub).to("cpu").feats[:, :3]
    else:
        raise RuntimeError(f"unsupported VAE trainer for DiT decode: {trainer!r}")
    return decode_from_latent


_PBR_PAD_METALLIC = -1.0
_PBR_PAD_ROUGHNESS = 1.0
_PBR_PAD_ALPHA = 1.0


def _pack_emission_as_pbr(feats: torch.Tensor) -> torch.Tensor:
    n = feats.shape[0]
    pad = torch.tensor(
        [[_PBR_PAD_METALLIC, _PBR_PAD_ROUGHNESS, _PBR_PAD_ALPHA]],
        device=feats.device, dtype=feats.dtype,
    ).expand(n, -1)
    return torch.cat([feats, pad], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--samples", default="all")
    ap.add_argument("--data_root", default=None,
                    help="override data root; default = parent(--dataset) if it has emission_voxels_<res>/")
    ap.add_argument("--glb_root", default=None)
    ap.add_argument("--experiment", action="append", required=True,
                    help="repeatable name=dit_path:vae_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.001,
                    help="min emission magnitude to render (filters near-black voxels)")
    ap.add_argument("--max_points", type=int, default=60000)
    ap.add_argument("--point_size", type=float, default=0.012)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance_strength", type=float, default=3.0)
    args = ap.parse_args()

    specs = _parse_experiments(args.experiment)
    dit_handles = [(n, resolve_experiment(n + "-dit", dp), resolve_experiment(n + "-vae", vp))
                   for n, dp, vp in specs]

    dit_handle_list: list[ExperimentHandle] = [h[1] for h in dit_handles]
    data_root = _resolve_data_root(args.dataset, dit_handle_list, args.data_root)
    resolution = dit_handle_list[0].resolution if dit_handle_list else 256

    records = load_samples(
        data_root=data_root,
        resolution=resolution,
        source=args.dataset,
        select=args.samples,
        glb_root=args.glb_root,
        require=("emission_voxel_path",),
    )
    if not records:
        raise SystemExit(f"no samples matched --samples={args.samples!r}")

    built = []
    for name, dit_h, vae_h in dit_handles:
        dit, sampler = _build_dit(dit_h, sigma_min=dit_h.config["trainer"]["args"]["sigma_min"])
        decode_from_latent = _build_vae_decoder_from_latent(vae_h)
        dino_args = dit_h.config["trainer"]["args"]["image_cond_model"]["args"]
        dino = _build_dino(dino_args)
        built.append({
            "name": name, "dit": dit, "sampler": sampler,
            "decode_from_latent": decode_from_latent,
            "dino": dino, "dit_cfg": dit_h.config,
        })

    dino_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    out_abs = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_abs)
    glb_assets_dir = os.path.join(out_dir, "assets", "glbs")
    os.makedirs(glb_assets_dir, exist_ok=True)

    def _stage_glb_copy(glb_path: str) -> str:
        """Copy GLB into <out_dir>/assets/glbs/ and return a relative URL.

        Real file copies (not symlinks) so the HTML+assets bundle is portable
        to web servers like aspis where /local-scratch isn't visible.
        """
        import shutil as _sh
        basename = os.path.basename(glb_path)
        dst = os.path.join(glb_assets_dir, basename)
        if not os.path.exists(dst):
            _sh.copy2(glb_path, dst)
        return f"assets/glbs/{basename}"

    samples_out = []
    for rec in records:
        coords, gt_emit, res = read_emission_voxel(rec.emission_voxel_path)

        panels = []
        glb_rel_url = _stage_glb_copy(rec.glb_path) if rec.glb_path else None
        panels.append({"kind": "glb", "data": make_glb_panel_data(glb_rel_url, False),
                       "caption": "", "label": "GLB"})
        panels.append({"kind": "voxel",
                       "data": make_voxel_panel_data(coords, gt_emit, res, args.threshold, args.max_points),
                       "caption": f"{coords.shape[0]:,} voxels", "label": "GT emission"})

        for exp in built:
            ds = exp["dit_cfg"]["dataset"]["args"]
            skip_pbr_cond = bool(ds.get("skip_pbr_cond", False))
            emit_mean = torch.tensor(ds["emission_normalization"]["mean"])
            emit_std  = torch.tensor(ds["emission_normalization"]["std"])
            # pbr_normalization is absent in albedo→emission (skip_pbr_cond) configs.
            pbr_mean = pbr_std = None
            if not skip_pbr_cond:
                pbr_mean = torch.tensor(ds["pbr_normalization"]["mean"])
                pbr_std  = torch.tensor(ds["pbr_normalization"]["std"])
            shape_mean = torch.tensor(ds["shape_normalization"]["mean"])
            shape_std  = torch.tensor(ds["shape_normalization"]["std"])

            emit_latent_name = ds["emission_latent_name"]
            pbr_latent_name = ds.get("pbr_latent_name")
            shape_latent_name = ds["shape_latent_name"]
            z_emit_raw  = _load_latent(f"{data_root}/emission_latents/{emit_latent_name}/{rec.sha}.npz")
            z_shape_raw = _load_latent(f"{data_root}/shape_latents/{shape_latent_name}/{rec.sha}.npz")
            z_pbr_raw   = (None if skip_pbr_cond
                           else _load_latent(f"{data_root}/pbr_latents/{pbr_latent_name}/{rec.sha}.npz"))

            def _norm(st, mu, sd): return st.replace((st.feats - mu.to(st.feats.device)) / sd.to(st.feats.device))

            z_emit_n  = _norm(z_emit_raw,  emit_mean,  emit_std).to(DEVICE)
            z_shape_n = _norm(z_shape_raw, shape_mean, shape_std).to(DEVICE)
            if skip_pbr_cond:
                # albedo→emission DiT: concat_cond = shape only (32ch); DiT input = 32 (x_t) + 32 (shape) = 64.
                concat_cond = z_emit_n.replace(z_shape_n.feats)
                z_pbr_n = None
            else:
                z_pbr_n = _norm(z_pbr_raw, pbr_mean, pbr_std).to(DEVICE)
                concat_cond = z_emit_n.replace(torch.cat([z_shape_n.feats, z_pbr_n.feats], dim=-1))

            thumb = os.path.join(data_root, "thumbnails", f"{rec.sha}.png")
            if not os.path.exists(thumb):
                raise FileNotFoundError(f"thumbnail not found: {thumb}")
            img = Image.open(thumb).convert("RGB")
            sz = exp["dit_cfg"]["trainer"]["args"]["image_cond_model"]["args"]["image_size"]
            arr = np.array(img.resize((sz, sz), Image.LANCZOS)).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            t = dino_tr(t).to(exp["dino"].model.embeddings.patch_embeddings.weight.dtype)
            img_feat = exp["dino"].extract_features(t)
            neg_cond = torch.zeros_like(img_feat)

            noise = z_emit_n.replace(torch.randn_like(z_emit_n.feats))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                res_s = exp["sampler"].sample(
                    exp["dit"], noise, cond=img_feat, neg_cond=neg_cond,
                    concat_cond=concat_cond, steps=args.steps,
                    guidance_strength=args.guidance_strength, verbose=False,
                )
            z_dit_n = res_s.samples
            z_dit_raw = z_dit_n.replace(z_dit_n.feats * emit_std.to(DEVICE) + emit_mean.to(DEVICE))

            # Build a GT SparseTensor to prime the decoder's subdivision cache.
            from trellis2.modules import sparse as sp
            bc = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
            x_gt = sp.SparseTensor(gt_emit, torch.cat([bc, coords], dim=-1)).to(DEVICE)
            # In skip_pbr_cond mode the trainer is EmissionPbrFinetuneTrainer and
            # the decoder ignores z_pbr_raw — pass z_dit_raw as a stand-in to
            # satisfy the signature.
            _z_pbr_for_decode = z_dit_raw if skip_pbr_cond else z_pbr_raw
            pred = exp["decode_from_latent"](x_gt, z_dit_raw, _z_pbr_for_decode)
            l1 = compute_emission_l1(coords, pred, coords, gt_emit)
            panels.append({
                "kind": "voxel",
                "data": make_voxel_panel_data(coords, pred, res, args.threshold, args.max_points),
                "caption": f"emission L1: {l1:.4f}",
                "label": exp["name"],
            })
            torch.cuda.empty_cache()

        samples_out.append({"sha": rec.sha, "info": f"{coords.shape[0]:,} voxels", "panels": panels})

    render_html(
        samples=samples_out,
        title=f"Emission DiT 3D — {len(built)} experiment(s)",
        out_path=args.out,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
