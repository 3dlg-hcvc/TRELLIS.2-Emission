"""
LightGenSLatEmissionLatent: the emission DiT's dataset (LightGen, emission generation in voxel space).

For training the emission DiT with flow matching. Returns the pre-encoded emission latent (x_0), the
pre-encoded PBR material latent and the shape latent (concat_cond, concatenated per token), and the
thumbnail for image conditioning. It mirrors upstream's SLatPbr (`structured_latent_svpbr.py`) with
the roles swapped; its index is a parquet + split JSON (`emission_helpers._build_parquet_samples`):

  emission_latents/{name}/{sha256}.npz   the emission VAE's encoder output (+ normalization_stats.json)
  pbr_latents/{name}/{sha256}.npz        the stock PBR encoder's latent
  shape_latents/{name}/{sha256}.npz      the stock shape encoder's latent
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from .emission_helpers import _build_parquet_samples, _read_npz_as_sparse, _sparse_collate


class LightGenSLatEmissionLatent(Dataset):
    """
    Returns pre-encoded latents for training the emission DiT.

    Each sample contains:
        x_0           SparseTensor  [N, C_emit_lat]   emission latent (noised by flow matching)
        concat_cond   SparseTensor  [N, C_pbr_lat]    PBR material latent (concatenated per-token
                                                       as DiT input; same coordinates as x_0)
        cond          SparseTensor  [M, C_shape_lat]  shape latent (geometry cross-attention),
                                                       only if ``shape_latent_root`` is provided
        sha256        str

    The DiT receives ``cat(x_t, concat_cond)`` as its sequence input and cross-attends to
    ``cond`` (shape) — mirroring the SLatPbr → shape_latent conditioning in TRELLIS.2.

    Args:
        data_root              Root of the processed data tree.
        parquet_file           Path to the parquet index.
        emission_latent_root   Directory containing ``emission_latents/{latent_name}/``.
                               Defaults to ``data_root``.
        pbr_latent_root        Directory containing ``pbr_latents/{latent_name}/``.
                               Defaults to ``data_root``.
        emission_latent_name   Subfolder name for emission latents.
        pbr_latent_name        Subfolder name for PBR latents.
        shape_latent_root      Optional directory for shape latents (geometry conditioning).
        shape_latent_name      Subfolder name for shape latents.
        resolution             Voxel grid resolution (used to look up latent tokens count).
        split                  "train" / "val" / "test" / None.
        min_emission           Keep only samples with max-emission ≥ this value.
        split_indices          Override split selection.
        split_json             Explicit path to split JSON.
        emission_normalization Optional {"mean": [...], "std": [...]} to normalise emission latent.
        pbr_normalization      Optional normalisation for the PBR latent.
        shape_normalization    Optional normalisation for the shape latent.
    """

    def __init__(
        self,
        data_root: str,
        parquet_file: str,
        emission_latent_name: str,
        pbr_latent_name: str,
        emission_latent_root: Optional[str] = None,
        pbr_latent_root: Optional[str] = None,
        shape_latent_root: Optional[str] = None,
        shape_latent_name: Optional[str] = None,
        thumbnail_dir: Optional[str] = None,
        image_size: int = 512,
        resolution: int = 256,
        split: Optional[str] = "train",
        min_emission: float = 0.0,
        split_indices: Optional[Any] = None,
        split_json: Optional[str] = None,
        emission_normalization: Optional[Dict] = None,
        pbr_normalization: Optional[Dict] = None,
        shape_normalization: Optional[Dict] = None,
        dummy_cond_channels: int = 32,
        roots: Optional[Any] = None,  # accepted but ignored
        skip_pbr_cond: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.resolution = resolution
        self.split = split
        self.dummy_cond_channels = dummy_cond_channels
        self.thumbnail_dir = thumbnail_dir
        self.image_size = image_size
        # When skip_pbr_cond=True, we run in the channel-hijack regime: the
        # emission latent IS the PBR-slot latent (VAE trained with emission
        # packed into base_color + default metallic/roughness/alpha), so
        # concat_cond = shape only and DiT input = 32 (x_0) + 32 (shape) = 64,
        # architecturally identical to the pretrained PBR DiT.
        self.skip_pbr_cond = skip_pbr_cond

        self.emit_dir   = os.path.join(emission_latent_root or data_root,
                                       "emission_latents", emission_latent_name)
        self.pbr_dir    = (
            os.path.join(pbr_latent_root or data_root,
                         "pbr_latents",      pbr_latent_name)
            if not skip_pbr_cond else None
        )
        self.shape_dir  = (
            os.path.join(shape_latent_root or data_root, "shape_latents", shape_latent_name)
            if shape_latent_name
            else None
        )

        self.emit_norm  = emission_normalization
        self.pbr_norm   = pbr_normalization
        self.shape_norm = shape_normalization

        self.samples = _build_parquet_samples(
            data_root, parquet_file, split, min_emission, split_indices, split_json
        )

        if self.samples:
            self._validate_first_sample()

        # Populate self.loads (per-sample token count) so BalancedResumableSampler
        # can distribute heavy vs light samples evenly across DDP ranks. Uses the
        # emission latent's coord count, which matches the sparse attention cost.
        # Fast: just reads .npz header (no decompression of feats).
        self.loads: List[int] = []
        if self.samples:
            for s in self.samples:
                p = os.path.join(self.emit_dir, f"{s['sha256']}.npz")
                try:
                    with np.load(p) as d:
                        self.loads.append(int(d["coords"].shape[0]))
                except Exception:
                    self.loads.append(0)
            if self.loads:
                lo, hi = min(self.loads), max(self.loads)
                print(
                    f"[LightGenSLatEmissionLatent] loads: min={lo} max={hi} "
                    f"ratio={hi/max(lo,1):.1f}× (for BalancedResumableSampler)"
                )

        print(
            f"[LightGenSLatEmissionLatent] split={split!r}: {len(self.samples)} samples, "
            f"emit_dir={self.emit_dir}, shape_dir={self.shape_dir}, "
            f"thumbnail_dir={self.thumbnail_dir}"
        )

    def _validate_first_sample(self) -> None:
        sha256 = self.samples[0]["sha256"]
        checks = [(self.emit_dir, "emission")]
        if self.pbr_dir is not None:
            checks.append((self.pbr_dir, "PBR material"))
        for d, label in checks:
            p = os.path.join(d, f"{sha256}.npz")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{label} latent not found: {p}\n"
                    f"Run the corresponding encode_*_latent.py script first."
                )

    def get_instance(self, sha256: str) -> Dict[str, Any]:
        x_0 = _read_npz_as_sparse(
            os.path.join(self.emit_dir, f"{sha256}.npz"),
            normalization=self.emit_norm,
        )

        if self.skip_pbr_cond:
            # Channel-hijack regime: concat_cond = shape only (32ch). PBR
            # conditioning is unnecessary because our emission VAE already
            # encoded [emit_rgb, default_metallic, default_roughness, default_alpha]
            # into the 32-ch latent, which IS the PBR-slot latent.
            if self.shape_dir is None:
                raise ValueError("skip_pbr_cond=True requires shape_latent_name to be set")
            shape_path = os.path.join(self.shape_dir, f"{sha256}.npz")
            shape_z = _read_npz_as_sparse(shape_path, normalization=self.shape_norm)
            assert torch.equal(x_0.coords, shape_z.coords), (
                f"Emission and shape latent coords mismatch for {sha256}"
            )
            concat_cond = x_0.replace(shape_z.feats)
        else:
            pbr_z = _read_npz_as_sparse(
                os.path.join(self.pbr_dir, f"{sha256}.npz"),
                normalization=self.pbr_norm,
            )
            assert torch.equal(x_0.coords, pbr_z.coords), (
                f"Emission and PBR latent coords mismatch for {sha256}: "
                f"{x_0.coords.shape} vs {pbr_z.coords.shape}"
            )

            # Build concat_cond: shape(32) + PBR(32) = 64ch, or PBR-only (32ch)
            if self.shape_dir is not None:
                shape_path = os.path.join(self.shape_dir, f"{sha256}.npz")
                if os.path.exists(shape_path):
                    shape_z = _read_npz_as_sparse(shape_path, normalization=self.shape_norm)
                    assert torch.equal(x_0.coords, shape_z.coords), (
                        f"Emission and shape latent coords mismatch for {sha256}"
                    )
                    concat_cond = x_0.replace(
                        torch.cat([shape_z.feats, pbr_z.feats], dim=-1)
                    )
                else:
                    concat_cond = pbr_z
            else:
                concat_cond = pbr_z

        out: Dict[str, Any] = {
            "x_0":         x_0,
            "concat_cond": concat_cond,
            "sha256":      sha256,
        }

        # Image conditioning: load thumbnail if available
        if self.thumbnail_dir is not None:
            img_path = os.path.join(self.thumbnail_dir, f"{sha256}.png")
            if os.path.exists(img_path):
                from PIL import Image
                image = Image.open(img_path).convert("RGBA")
                image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
                alpha = torch.tensor(
                    np.array(image.getchannel(3)), dtype=torch.float32
                ) / 255.0
                rgb = torch.tensor(
                    np.array(image.convert("RGB")), dtype=torch.float32
                ).permute(2, 0, 1) / 255.0
                out["cond"] = rgb * alpha.unsqueeze(0)
                return out

        # Fallback: dummy cond for cross-attention
        out["cond"] = torch.zeros(1, self.dummy_cond_channels)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sha256 = self.samples[index]["sha256"]
        try:
            return self.get_instance(sha256)
        except Exception as exc:
            print(f"[LightGenSLatEmissionLatent] Error loading {sha256}: {exc}")
            raise

    @torch.no_grad()
    def visualize_sample(self, sample: dict) -> dict:
        """Max-projection visualization of emission latent channels."""
        x_0 = sample["x_0"]  # SparseTensor [N, C]
        res = self.resolution // 16  # latent resolution

        def _max_proj(sp_tensor: SparseTensor, channels: slice) -> torch.Tensor:
            """Render a batch of max-projection images (B, 3, H, W)."""
            images = []
            for b in range(sp_tensor.shape[0]):
                mask = sp_tensor.coords[:, 0] == b
                coords = sp_tensor.coords[mask, 1:]  # [M, 3]
                feats = sp_tensor.feats[mask, channels]  # [M, C']
                C = feats.shape[-1]
                if C >= 3:
                    rgb = feats[:, :3]
                else:
                    rgb = torch.zeros(feats.shape[0], 3, device=feats.device)
                    rgb[:, :C] = feats
                lo, hi = rgb.min(), rgb.max()
                if hi - lo > 1e-6:
                    rgb = (rgb - lo) / (hi - lo)
                else:
                    rgb = rgb * 0 + 0.5
                img = torch.zeros(res, res, 3, device=feats.device)
                depth = torch.full((res, res), -1.0, device=feats.device)
                mag = rgb.norm(dim=-1)
                c = coords.long()
                for i in range(c.shape[0]):
                    y, x = c[i, 1].item(), c[i, 0].item()
                    if 0 <= y < res and 0 <= x < res and mag[i] > depth[y, x]:
                        depth[y, x] = mag[i]
                        img[y, x] = rgb[i]
                images.append(img.permute(2, 0, 1))  # [3, H, W]
            return torch.stack(images)  # [B, 3, H, W]

        vis = {}
        vis["emission_latent"] = _max_proj(x_0, slice(0, 3))
        if "concat_cond" in sample:
            vis["cond_latent"] = _max_proj(sample["concat_cond"], slice(0, 3))
        if "cond" in sample and sample["cond"].ndim == 4:
            vis["image_cond"] = sample["cond"]  # [B, 3, H, W]
        return vis

    @staticmethod
    def collate_fn(batch: List[Dict], split_size: Optional[int] = None):
        return _sparse_collate(batch, split_size)
