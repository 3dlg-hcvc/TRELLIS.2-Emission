#!/usr/bin/env python3
"""Build dual_grid_<res>/<sha>.vxz directly from GLB files, bypassing the
dump_mesh.py (Blender) step. Replicates the single-pass flexible-dual-grid
path from data_toolkit/dual_grid.py but loads vertices+faces via trimesh
instead of mesh_dumps pickles.

This is a shortcut for datasets where we have GLBs but not mesh_dumps.
Skips segmentation labels (we don't have them from GLB).

Usage:
    python data_toolkit/dual_grid_from_glb.py \
        --parquet <root>/<index>.parquet \
        --glb_root <dir with the GLBs> \
        --out_root <root> \
        --resolution 256
"""
from __future__ import annotations
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import trimesh

import o_voxel


def load_mesh_from_glb(glb_path: str):
    """Load a GLB into a single (vertices, faces) tensor pair, normalized to
    the unit cube [-0.5, 0.5]^3 (same convention as dual_grid.py)."""
    scene = trimesh.load(glb_path, process=False, force="scene")
    if isinstance(scene, trimesh.Scene):
        verts_list, faces_list, start = [], [], 0
        for name, geom in scene.geometry.items():
            if not hasattr(geom, "vertices") or not hasattr(geom, "faces"):
                continue
            if len(geom.vertices) == 0 or len(geom.faces) == 0:
                continue
            # Apply node transforms to vertices.
            matrix = None
            for node_name in scene.graph.nodes_geometry:
                transform, geometry_name = scene.graph[node_name]
                if geometry_name == name:
                    matrix = transform
                    break
            v = np.asarray(geom.vertices, dtype=np.float32)
            if matrix is not None:
                v = trimesh.transformations.transform_points(v, matrix)
            verts_list.append(v)
            faces_list.append(np.asarray(geom.faces, dtype=np.int64) + start)
            start += len(v)
        if not verts_list:
            raise ValueError(f"no geometry in {glb_path}")
        vertices_np = np.concatenate(verts_list, axis=0)
        faces_np = np.concatenate(faces_list, axis=0)
    else:
        vertices_np = np.asarray(scene.vertices, dtype=np.float32)
        faces_np = np.asarray(scene.faces, dtype=np.int64)

    # Normalize to [-0.5, 0.5]^3 (match dual_grid.py convention).
    vmin = vertices_np.min(axis=0)
    vmax = vertices_np.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    vertices_np = (vertices_np - center) * scale
    assert vertices_np.min() >= -0.5 - 1e-5 and vertices_np.max() <= 0.5 + 1e-5
    return torch.from_numpy(vertices_np).float(), torch.from_numpy(faces_np).long()


def process_one(sha: str, glb_path: str, out_dir: str, res: int) -> dict:
    out_path = os.path.join(out_dir, f"{sha}.vxz")
    if os.path.exists(out_path):
        return {"sha256": sha, "status": "skipped"}
    try:
        vertices, faces = load_mesh_from_glb(glb_path)
        data = {"vertices": vertices, "faces": faces}
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            **data,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=False,
        )
        dual_vertices = dual_vertices * res - voxel_indices
        dual_vertices = torch.clamp(dual_vertices, 0, 1)
        dual_vertices = (dual_vertices * 255).type(torch.uint8)
        intersected = (
            intersected[:, 0:1]
            + 2 * intersected[:, 1:2]
            + 4 * intersected[:, 2:3]
        ).type(torch.uint8)
        attr = {"vertices": dual_vertices, "intersected": intersected}
        o_voxel.io.write_vxz(out_path, voxel_indices, attr)
        return {"sha256": sha, "status": "ok", "n_voxels": int(voxel_indices.shape[0])}
    except Exception as e:
        return {"sha256": sha, "status": "error", "error": repr(e)[:200], "trace": traceback.format_exc()[:400]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--glb_root", required=True,
                    help="root containing glbs/ (matches parquet's glb_1k_path column relative paths)")
    ap.add_argument("--out_root", required=True,
                    help="dataset root where dual_grid_<res>/ will be created")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--max_workers", type=int, default=1,
                    help="o_voxel uses threads internally; default serial to avoid over-subscription")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, f"dual_grid_{args.resolution}")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    shas_all = list(df.index.astype(str))
    start = len(shas_all) * args.rank // args.world_size
    end = len(shas_all) * (args.rank + 1) // args.world_size
    shas = shas_all[start:end]
    print(f"Samples total: {len(shas_all)}  rank {args.rank}/{args.world_size} → {len(shas)}")

    ok, skip, err = 0, 0, 0
    err_log = []
    for i, sha in enumerate(shas):
        glb_rel = df.loc[sha, "glb_1k_path"]
        glb = os.path.join(args.glb_root, glb_rel)
        if not os.path.exists(glb):
            err += 1
            err_log.append({"sha256": sha, "status": "missing_glb", "path": glb})
            continue
        r = process_one(sha, glb, out_dir, args.resolution)
        if r["status"] == "ok":
            ok += 1
        elif r["status"] == "skipped":
            skip += 1
        else:
            err += 1
            err_log.append(r)

        if (i + 1) % 20 == 0 or i == len(shas) - 1:
            print(f"[{i+1}/{len(shas)}] ok={ok} skipped={skip} err={err}", flush=True)

    # Summary
    print(f"\nDone: ok={ok} skipped={skip} err={err}")
    if err_log:
        err_path = os.path.join(out_dir, "errors.txt")
        with open(err_path, "w") as f:
            for e in err_log[:200]:
                f.write(str(e) + "\n")
        print(f"First errors written to {err_path}")


if __name__ == "__main__":
    main()
