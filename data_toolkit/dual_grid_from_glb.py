#!/usr/bin/env python3
"""Build dual_grid_<res>/<sha>.vxz (the shape encoder's input) directly from GLB files, without the
dump_mesh.py (Blender) step of data_toolkit/dual_grid.py: the same single-pass flexible dual grid, from
vertices and faces loaded here. No segmentation labels (a GLB has none).

    python data_toolkit/dual_grid_from_glb.py --parquet <root>/<index>.parquet --glb_root <dir> \
        --out_root <root> [--loader trimesh|lightgenbench] [--lightgen_repo <dir>] [--check_cells] \
        [--resolution 256] [--rank r --world_size w]

--parquet      the shapes: its index, and its `glb_1k_path` column if it has one.
--glb_root     where the GLBs are: each `glb_1k_path` is relative to it; without that column, a TexVerse-1K
               download (Hugging Face YiboZhang2001/TexVerse-1K), whose GLBs are glbs/glbs_1k/<group>/<sha>_1024.glb.
--loader       how a GLB becomes one mesh. `trimesh` (default): load_mesh_from_glb below. `lightgenbench`:
               the loader LightGenBench's data preparation uses (data_processing/glb2representations/utils.py of
               the lightgen repository, --lightgen_repo, default the directory that holds this repository);
               LightGenBench's shape latents were encoded from its grids. Both give the same cells, but the
               `vertices` attribute differs on hundreds of cells per shape.
--check_cells  keep a grid only if its cells equal those of <out_root>/pbr_voxels_<res>/<sha>.vxz.
A grid already in dual_grid_<res>/ is kept as it is; each is written whole (through a temporary file), so
delete one to rebuild it. Failures go to dual_grid_<res>/failed_rank<r>.txt, and the exit code is non-zero
unless every shape of the rank is done. One process per rank (o_voxel runs threads itself); start several
ranks side by side.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import trimesh

import o_voxel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def lightgenbench_loader(lightgen_repo: str):
    """LightGenBench's mesh loader: load_and_merge, then the tight normalize to [-0.5, 0.5]^3 in float64."""
    if not os.path.isfile(os.path.join(lightgen_repo, "data_processing", "glb2representations", "utils.py")):
        sys.exit(f"no data_processing/glb2representations/utils.py under {lightgen_repo}: pass --lightgen_repo")
    sys.path.insert(0, lightgen_repo)
    from data_processing.glb2representations.utils import TIGHT_SCALE_NUM, load_and_merge

    def load(glb_path: str):
        b = load_and_merge(glb_path)
        v = np.asarray(b.vertices, dtype=np.float64)
        center = (v.min(0) + v.max(0)) / 2.0
        ext = float((v.max(0) - v.min(0)).max())
        v = (v - center) * (TIGHT_SCALE_NUM / ext if ext > 0 else 1.0)
        return torch.from_numpy(v).float(), torch.from_numpy(np.asarray(b.faces)).long()

    return load


def find_glbs(df: pd.DataFrame, glb_root: str, shas: list) -> dict:
    """sha -> GLB path: the `glb_1k_path` column under glb_root, else a TexVerse-1K folder layout."""
    if "glb_1k_path" in df.columns:
        return {s: os.path.join(glb_root, df.loc[s, "glb_1k_path"]) for s in shas}
    want, found = set(shas), {}
    top = os.path.join(glb_root, "glbs", "glbs_1k")
    for group in sorted(os.listdir(top)):
        with os.scandir(os.path.join(top, group)) as it:
            for e in it:
                s = e.name[:-len("_1024.glb")] if e.name.endswith("_1024.glb") else None
                if s in want:
                    found[s] = e.path
    return found


def cells(c):
    return np.unique(np.ascontiguousarray(c.cpu().numpy().astype(np.int64)), axis=0)


def main() -> None:
    global load_mesh_from_glb
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--glb_root", required=True)
    ap.add_argument("--out_root", required=True, help="dataset root where dual_grid_<res>/ is written")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--loader", choices=["trimesh", "lightgenbench"], default="trimesh")
    ap.add_argument("--lightgen_repo", default=os.path.dirname(REPO))
    ap.add_argument("--check_cells", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()
    if args.loader == "lightgenbench":
        load_mesh_from_glb = lightgenbench_loader(os.path.abspath(args.lightgen_repo))

    res = args.resolution
    df = pd.read_parquet(args.parquet)
    shas_all = [str(s) for s in df.index]
    shas = shas_all[len(shas_all) * args.rank // args.world_size:len(shas_all) * (args.rank + 1) // args.world_size]
    out_dir = os.path.join(args.out_root, f"dual_grid_{res}")
    tmp_dir = os.path.join(out_dir, f".tmp_rank{args.rank}")
    os.makedirs(tmp_dir, exist_ok=True)
    failed = os.path.join(out_dir, f"failed_rank{args.rank}.txt")
    todo = [s for s in shas if not os.path.exists(os.path.join(out_dir, f"{s}.vxz"))]
    glbs = find_glbs(df, args.glb_root, todo) if todo else {}
    print(f"[dual_grid] rank {args.rank}/{args.world_size}: {len(shas)} shapes, {len(todo)} to build, "
          f"loader {args.loader}", flush=True)

    n = {"ok": 0, "skipped": len(shas) - len(todo), "missing_glb": 0, "mismatch": 0, "error": 0}
    t0 = time.time()
    for i, s in enumerate(todo):
        out_path, tmp_path = os.path.join(out_dir, f"{s}.vxz"), os.path.join(tmp_dir, f"{s}.vxz")
        if s not in glbs or not os.path.exists(glbs[s]):
            n["missing_glb"] += 1
            with open(failed, "a") as f:
                f.write(f"{s}\tmissing_glb\t{glbs.get(s, '')}\n")
            continue
        try:
            if os.path.exists(tmp_path):  # left by an interrupted run; process_one would skip it
                os.remove(tmp_path)
            r = process_one(s, glbs[s], tmp_dir, res)
            if r["status"] != "ok":
                raise RuntimeError(r.get("error", r["status"]))
            if args.check_cells:
                c_dg, _ = o_voxel.io.read_vxz(tmp_path)
                c_pbr, _ = o_voxel.io.read_vxz(os.path.join(args.out_root, f"pbr_voxels_{res}", f"{s}.vxz"))
                a, b = cells(c_dg), cells(c_pbr)
                if a.shape != b.shape or not np.array_equal(a, b):
                    os.remove(tmp_path)
                    n["mismatch"] += 1
                    with open(failed, "a") as f:
                        f.write(f"{s}\tmismatch\tdual_grid={len(a)} pbr={len(b)}\n")
                    continue
            os.replace(tmp_path, out_path)
            n["ok"] += 1
        except Exception as e:  # noqa: BLE001
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            n["error"] += 1
            with open(failed, "a") as f:
                f.write(f"{s}\terror\t{e!r}\n")
            print(f"[dual_grid] ERROR {s}: {e!r}\n{traceback.format_exc()[-400:]}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"[dual_grid] rank {args.rank}: {i + 1}/{len(todo)} {n} {time.time() - t0:.0f}s", flush=True)
    os.rmdir(tmp_dir)
    print(f"[dual_grid] rank {args.rank} done: {n} in {time.time() - t0:.0f}s", flush=True)
    # os._exit, not sys.exit: the interpreter's teardown of the loaded extensions can abort after all
    # the work is done.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if n["ok"] + n["skipped"] == len(shas) else 1)


if __name__ == "__main__":
    main()
