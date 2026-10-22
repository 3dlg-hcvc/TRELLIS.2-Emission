"""
Helpers shared by the two LightGen emission datasets, `sparse_voxel_emission.LightGenSLatEmission`
(the emission VAE's data) and `structured_latent_emission.LightGenSLatEmissionLatent` (the emission
DiT's data): the voxel and latent readers, the sparse collate, and the parquet / split-JSON sample
list.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..modules import sparse as sp
from ..modules.sparse.basic import SparseTensor, sparse_cat
from ..utils.data_utils import load_balanced_group_indices

# o_voxel is a compiled C extension — import lazily so the module loads
# even when the extension hasn't been built yet.
try:
    import o_voxel as _o_voxel
except ImportError:
    _o_voxel = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_vxz_as_sparse(path: str, attr_keys: List[str]) -> SparseTensor:
    """Read a .vxz file and return a batch-0 SparseTensor."""
    if _o_voxel is None:
        raise ImportError(
            "o_voxel is required to load .vxz voxel files. "
            "Build it with: cd o-voxel && pip install -e ."
        )
    coords, attr = _o_voxel.io.read_vxz(path, num_threads=4)
    feats = torch.cat([attr[k] for k in attr_keys], dim=-1).float() / 255.0 * 2.0 - 1.0
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return SparseTensor(feats, torch.cat([batch_col, coords], dim=-1))


def _read_npz_as_sparse(path: str, normalization: Optional[Dict] = None) -> SparseTensor:
    """Load a pre-encoded latent .npz and return a batch-0 SparseTensor."""
    data = np.load(path)
    feats  = torch.from_numpy(data['feats'].copy()).float()
    coords = torch.from_numpy(data['coords'].copy()).int()
    if normalization is not None:
        mean = torch.tensor(normalization['mean']).reshape(1, -1)
        std  = torch.tensor(normalization['std']).reshape(1, -1)
        feats = (feats - mean) / std
    batch_col = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
    return SparseTensor(feats, torch.cat([batch_col, coords], dim=-1))


# ---------------------------------------------------------------------------
# Shared collate function (works for any dict of SparseTensors / Tensors)
# ---------------------------------------------------------------------------

def _sparse_collate(batch: List[Dict], split_size: Optional[int] = None):
    """Load-balance and collate a list of sample dicts containing SparseTensors."""
    if split_size is None:
        group_idx = [list(range(len(batch)))]
    else:
        # support both "x" (VAE) and "x_0" (DiT) as the primary sparse tensor key
        primary_key = "x" if "x" in batch[0] else "x_0"
        group_idx = load_balanced_group_indices(
            [b[primary_key].feats.shape[0] for b in batch], split_size
        )

    packs = []
    for group in group_idx:
        sub = [batch[i] for i in group]
        pack: Dict[str, Any] = {}
        for k in sub[0]:
            v0 = sub[0][k]
            if isinstance(v0, SparseTensor):
                pack[k] = sparse_cat([b[k] for b in sub], dim=0)
            elif isinstance(v0, torch.Tensor):
                pack[k] = torch.stack([b[k] for b in sub])
            elif isinstance(v0, list):
                pack[k] = sum([b[k] for b in sub], [])
            else:
                pack[k] = [b[k] for b in sub]
        packs.append(pack)

    return packs[0] if split_size is None else packs


# ---------------------------------------------------------------------------
# Parquet-based sample list builder
# ---------------------------------------------------------------------------

_DEFAULT_SPLIT_JSONS = [
    "data_splits_emission_filtered.json",
    "data_splits_filtered.json",
]

_VXZ_DIR    = "vxz_root"   # placeholder key in roots dict
_LATENT_DIR = "latent_root"


def _build_parquet_samples(
    data_root: str,
    parquet_file: str,
    split: Optional[str],
    min_emission: float,
    split_indices: Optional[Any],
    split_json: Optional[str],
) -> List[Dict[str, str]]:
    """Build a list of {sha256, ...path...} dicts from a parquet index file."""
    df = pd.read_parquet(parquet_file)
    if "success" in df.columns:
        df = df[df["success"] == True]
    emit_col = "#stat_emission_color_max"
    if min_emission > 0.0 and emit_col in df.columns:
        df = df[df[emit_col] >= min_emission]

    # Auto-detect split JSON
    if split is not None and split_indices is None and split_json is None:
        for name in _DEFAULT_SPLIT_JSONS:
            candidate = os.path.join(data_root, name)
            if os.path.exists(candidate):
                split_json = candidate
                break

    # Build flat sample list (before any split filtering)
    samples: List[Dict[str, str]] = []
    for sample_id, row in df.iterrows():
        ditem_dir = row.get("ditem_dir", None)
        if ditem_dir is None:
            sid = str(sample_id)
            ditem_dir = f"{sid[:3]}-{sid[3:6]}/{sid}"
        samples.append({"sha256": str(sample_id), "ditem_dir": ditem_dir})

    # Apply split
    if split is not None and split_json is not None and split_indices is None:
        split_indices = split_json
    if split_indices is not None:
        samples = _apply_split_indices(samples, split_indices, split)

    return samples


def _apply_split_indices(
    samples: List[Dict],
    indices: Any,
    split: Optional[str],
) -> List[Dict]:
    if isinstance(indices, tuple):
        return samples[indices[0] : indices[1]]
    if isinstance(indices, list):
        return [samples[i] for i in indices]
    if isinstance(indices, str):
        with open(indices) as f:
            split_data = json.load(f)
        if split is None:
            idx_list = []
            for v in split_data.values():
                if isinstance(v, list):
                    idx_list.extend(v)
                elif isinstance(v, dict):
                    idx_list.extend(v.get("indices", []))
        else:
            entry = split_data.get(split, [])
            idx_list = entry.get("indices", entry) if isinstance(entry, dict) else entry
        return [samples[i] for i in idx_list]
    return samples


# ---------------------------------------------------------------------------
# Dataset 1 — raw emission voxels for VAE training
# ---------------------------------------------------------------------------
