#!/usr/bin/env python3
"""Unified 3D interactive visualization for emission-VAE reconstruction.

Renders GLB + GT emission + N experiment predictions per sample. All panels
share a Three.js camera.

Usage:
  python vis/vis_3d_vae.py \\
      --dataset <root>/<index>.parquet \\
      --samples all \\
      --glb_root <dir with the GLBs> \\
      --experiment "pbr→emission=outputs/<vae run>/ckpts/<file>.ckpt" \\
      --out visualizations/vae.html
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import o_voxel  # noqa: E402
import torch    # noqa: E402

from vis.vis_utils import (  # noqa: E402
    ExperimentHandle, compute_emission_l1, load_samples,
    load_vae_from_handle, make_glb_panel_data, make_voxel_panel_data,
    read_emission_voxel, render_html, resolve_experiment,
)


def _load_pbr_feats(path: str) -> torch.Tensor | None:
    if not os.path.exists(path):
        return None
    attrs = ["base_color", "metallic", "roughness", "alpha"]
    _, pbr_attr = o_voxel.io.read_vxz(path, num_threads=4)
    return torch.cat([pbr_attr[a].float() / 255.0 * 2.0 - 1.0 for a in attrs], dim=-1)


def _parse_experiments(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--experiment expects name=path, got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def _stage_glb(glb_path: str, assets_dir: str) -> str:
    """Symlink ``glb_path`` into ``assets_dir`` and return the relative URL
    (``assets/glbs/<basename>``) for use in the HTML. Using a subdirectory of
    the HTML's output dir lets browsers load GLBs under file:// (they block
    cross-directory ../ traversal but allow same-dir and subdir paths).
    """
    os.makedirs(assets_dir, exist_ok=True)
    basename = os.path.basename(glb_path)
    link_path = os.path.join(assets_dir, basename)
    if not os.path.lexists(link_path):
        os.symlink(os.path.abspath(glb_path), link_path)
    # Relative URL from the HTML (assets_dir = <out_dir>/assets/glbs).
    parent = os.path.dirname(assets_dir)  # <out_dir>/assets
    assets_root = os.path.basename(parent)  # "assets"
    return f"{assets_root}/{os.path.basename(assets_dir)}/{basename}"


def _resolve_data_root(dataset_arg: str, experiments: list[ExperimentHandle], override: str | None) -> str:
    """Pick the data_root to use, in priority order:
    1. --data_root CLI override.
    2. parent directory of --dataset (if it contains emission_voxels_<res>/).
    3. experiments[0].data_root from config (final fallback; may be stale).
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="parquet or splits JSON path")
    ap.add_argument("--samples", default="all",
                    help="all | first:N | split:<name> | sha:<csv>")
    ap.add_argument("--data_root", default=None,
                    help="override data root; default = parent(--dataset) if it has emission_voxels_<res>/")
    ap.add_argument("--glb_root", default=None,
                    help="root for parquet's glb_1k_path; omit → GLB panel missing")
    ap.add_argument("--embed_glb", action="store_true",
                    help="inline each GLB as a base64 data URL (makes HTML self-contained but huge; "
                         "useful for small datasets or when file:// serving is needed)")
    ap.add_argument("--experiment", action="append", default=[],
                    help="repeatable name=run_dir_or_ckpt")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument("--threshold", type=float, default=0.001,
                    help="min emission magnitude to render (filters near-black voxels)")
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--point_size", type=float, default=0.012)
    args = ap.parse_args()

    experiments: list[ExperimentHandle] = [
        resolve_experiment(name, path) for name, path in _parse_experiments(args.experiment)
    ]
    data_root = _resolve_data_root(args.dataset, experiments, args.data_root)
    resolution = experiments[0].resolution if experiments else 256

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

    # Load VAEs once; reuse across samples.
    decoders = [(h, load_vae_from_handle(h)) for h in experiments]

    out_abs = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_abs)
    glb_assets_dir = os.path.join(out_dir, "assets", "glbs")

    samples_out: list[dict] = []
    for rec in records:
        coords, emit, res = read_emission_voxel(rec.emission_voxel_path)
        pbr = _load_pbr_feats(rec.pbr_voxel_path) if any(
            h.trainer_name in ("EmissionVaeTrainer", "EmissionPbrConditionalVaeTrainer")
            for h in experiments
        ) else None

        panels: list[dict] = []
        # GLB panel (or placeholder). Symlink into <out_dir>/assets/glbs/ so
        # browsers can load it under file:// without cross-directory issues.
        if rec.glb_path and os.path.exists(rec.glb_path):
            glb_url = _stage_glb(rec.glb_path, glb_assets_dir)
        else:
            glb_url = None
        panels.append({
            "kind": "glb",
            "data": make_glb_panel_data(
                glb_url,
                z_up_to_y_up=False,
                embed_path=rec.glb_path if (args.embed_glb and rec.glb_path) else None,
            ),
            "caption": "",
            "label": "GLB",
        })
        # GT
        panels.append({
            "kind": "voxel",
            "data": make_voxel_panel_data(coords, emit, res, args.threshold, args.max_points),
            "caption": f"{coords.shape[0]:,} voxels",
            "label": "GT emission",
        })
        # Experiments
        for h, decode in decoders:
            pred = decode(coords, emit, pbr)
            l1 = compute_emission_l1(coords, pred, coords, emit)
            panels.append({
                "kind": "voxel",
                "data": make_voxel_panel_data(coords, pred, res, args.threshold, args.max_points),
                "caption": f"emission L1: {l1:.4f}",
                "label": h.label,
            })
            torch.cuda.empty_cache()
        samples_out.append({
            "sha": rec.sha,
            "info": f"{coords.shape[0]:,} voxels",
            "panels": panels,
        })

    render_html(
        samples=samples_out,
        title=f"Emission VAE 3D — {len(experiments)} experiment(s)",
        out_path=args.out,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
