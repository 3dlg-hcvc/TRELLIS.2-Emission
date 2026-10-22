"""
LightGenSLatEmission: the emission VAE's dataset (LightGen, emission generation in voxel space).

Returns the raw emission voxels (x_0) and, optionally, the PBR voxels (cond). It mirrors upstream's
SparseVoxelPbrDataset (`sparse_voxel_pbr.py`) but loads the emission channel instead; its index is a
parquet + split JSON (`emission_helpers._build_parquet_samples`), the layout
`data_toolkit/build_lightgenbench_index.py` writes:

  emission_voxels_{res}/{sha256}.vxz   emissive(3)
  pbr_voxels_{res}/{sha256}.vxz        base_color(3) + metallic(1) + roughness(1) + alpha(1), same coordinates
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from .emission_helpers import _build_parquet_samples, _o_voxel, _read_npz_as_sparse, _sparse_collate


class LightGenSLatEmission(Dataset):
    """
    Raw emission + PBR voxels for training the emission VAE.

    Mirrors SparseVoxelPbrDataset: each sample returns a single SparseTensor ``x``
    whose features are ``cat(emission[3], pbr_attrs[C_pbr])`` in [-1, 1].
    The first ``emission_channels`` features are the reconstruction target; the
    remaining PBR features serve as geometry+material context for the encoder.

    Channel layout (default pbr_attrs):
        x[:, 0:3]  — emissive RGB
        x[:, 3:6]  — base_color
        x[:, 6:7]  — metallic
        x[:, 7:8]  — roughness
        x[:, 8:9]  — alpha
        total = 9

    Args:
        data_root      Root of the processed data tree, containing
                       ``emission_voxels_{res}/`` and ``pbr_voxels_{res}/``.
        parquet_file   Path to the parquet index.
        resolution     Voxel grid resolution used during voxelization.
        split          "train" / "val" / "test" / None.
        min_emission   Keep only samples whose max emission stat >= this value.
        split_indices  Override split (tuple / list / JSON path).
        split_json     Explicit path to a split JSON file.
        pbr_attrs      Which PBR attributes to stack after emission.
                       Set to [] to get emission-only (3-channel) input.
    """

    _ATTR_CHANNELS: Dict[str, int] = {
        "base_color": 3, "metallic": 1, "roughness": 1, "alpha": 1
    }
    EMISSION_CHANNELS: int = 3

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        resolution: int = 256,
        split: Optional[str] = "train",
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
        pbr_attrs: List[str] = ("base_color", "metallic", "roughness", "alpha"),
        pbr_latent_name: Optional[str] = None,
        pbr_latent_root: Optional[str] = None,
        roots: Optional[Any] = None,  # accepted but ignored (TrellisDataModule injects this)
    ) -> None:
        super().__init__()
        self.data_root   = data_root
        self.resolution  = resolution
        self.pbr_attrs   = list(pbr_attrs)
        self.split       = split
        self.value_range = (-1, 1)

        # channel layout: emission first, then pbr_attrs
        self.emission_channels = self.EMISSION_CHANNELS
        pbr_ch = sum(self._ATTR_CHANNELS[a] for a in self.pbr_attrs)
        self.in_channels = self.emission_channels + pbr_ch

        # layout dict for visualisation / downstream use
        self.layout: Dict[str, slice] = {"emissive": slice(0, self.emission_channels)}
        cur = self.emission_channels
        for a in self.pbr_attrs:
            c = self._ATTR_CHANNELS[a]
            self.layout[a] = slice(cur, cur + c)
            cur += c

        self.samples = _build_parquet_samples(
            data_root, parquet_file, split, min_emission, split_indices, split_json
        )
        # Per-sample "load" for BalancedResumableSampler (and collate split_size).
        # Emission .vxz file size is proportional to voxel count (fixed-byte-per-
        # voxel attrs), so it's a cheap, accurate proxy for per-sample memory.
        # 112x variance in practice — unbalanced assignment to ranks causes OOM.
        self.loads = [
            os.path.getsize(self._emission_vxz(s["sha256"])) for s in self.samples
        ]
        print(
            f"[LightGenSLatEmission] split={split!r}: {len(self.samples)} samples  "
            f"res={resolution}  in_channels={self.in_channels}  "
            f"({self.emission_channels} emission + {pbr_ch} PBR)"
        )

        # Optional pre-encoded PBR conditioning latent (for the conditional VAE).
        self.pbr_latent_dir = (
            os.path.join(pbr_latent_root or data_root, "pbr_latents", pbr_latent_name)
            if pbr_latent_name else None
        )
        if self.pbr_latent_dir is not None:
            assert os.path.isdir(self.pbr_latent_dir), (
                f"pbr_latent dir does not exist: {self.pbr_latent_dir}"
            )

    def _emission_vxz(self, sha256: str) -> str:
        return os.path.join(
            self.data_root, f"emission_voxels_{self.resolution}", f"{sha256}.vxz"
        )

    def _pbr_vxz(self, sha256: str) -> str:
        return os.path.join(
            self.data_root, f"pbr_voxels_{self.resolution}", f"{sha256}.vxz"
        )

    def get_instance(self, sha256: str) -> Dict[str, Any]:
        # emission — always loaded
        coords_e, attr_e = _o_voxel.io.read_vxz(self._emission_vxz(sha256), num_threads=4)
        emit_feats = attr_e["emissive"].float() / 255.0 * 2.0 - 1.0  # [N, 3]

        if self.pbr_attrs:
            # pbr — same coordinates, verified at voxelisation time
            _, attr_p = _o_voxel.io.read_vxz(self._pbr_vxz(sha256), num_threads=4)
            pbr_feats = torch.cat(
                [attr_p[a].float() / 255.0 * 2.0 - 1.0 for a in self.pbr_attrs], dim=-1
            )  # [N, C_pbr]
            feats = torch.cat([emit_feats, pbr_feats], dim=-1)  # [N, 3+C_pbr]
        else:
            feats = emit_feats

        batch_col = torch.zeros(coords_e.shape[0], 1, dtype=coords_e.dtype)
        x = SparseTensor(feats, torch.cat([batch_col, coords_e], dim=-1))

        out: Dict[str, Any] = {"x": x, "sha256": sha256}
        if self.pbr_latent_dir is not None:
            out["pbr_latent"] = _read_npz_as_sparse(
                os.path.join(self.pbr_latent_dir, f"{sha256}.npz"),
                normalization=None,
            )
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sha256 = self.samples[index]["sha256"]
        try:
            return self.get_instance(sha256)
        except Exception as exc:
            print(f"[LightGenSLatEmission] Error loading {sha256}: {exc}")
            raise

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int] = None):
        return _sparse_collate(batch, split_size)
