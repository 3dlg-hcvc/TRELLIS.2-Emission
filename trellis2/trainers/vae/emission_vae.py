from typing import *
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, l2_loss


class EmissionVaeTrainer(BasicTrainer):
    """
    Trainer for the emission VAE with material conditioning:
    material latent concatenated to emission latent before decoding.

    Architecture:
        Emission encoder (trainable):  emission voxels [N,3] → emission latent [M, C_e]
        PBR encoder (frozen, pretrained): PBR voxels [N,6] → material latent [M, C_m]
        Emission decoder (trainable):  concat([emit_lat, mat_lat]) [M, C_e+C_m] → emission [N,3]

    Both encoders must have the same number of downsample stages so that
    their latents share the same spatial resolution and coordinates.

    The decoder uses pred_subdiv=False; during training the spatial cache from
    the emission encoder provides the GT voxel structure automatically.

    Batch dict keys (from LightGenSLatEmission):
        x         SparseTensor  [N, emission_ch + pbr_ch]   combined voxel features

    Args:
        loss_type               : "l1" or "l2"
        lambda_kl               : KL regularisation weight
        emission_channels       : number of emission channels at the start of x.feats (default 3)
        pretrained_pbr_encoder  : path to pretrained PBR encoder (HF or local).
                                  If None, falls back to unconditioned mode.
    """

    def __init__(
        self,
        *args,
        loss_type: str = "l1",
        lambda_kl: float = 1e-6,
        emission_channels: int = 3,
        pretrained_pbr_encoder: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.emission_channels = emission_channels
        self._pretrained_pbr_encoder_path = pretrained_pbr_encoder
        self._pbr_encoder: Optional[torch.nn.Module] = None

        # The frozen PBR encoder is registered lazily on the first training step
        # (see _ensure_pbr_encoder), so it lands in saved checkpoints but is
        # absent from the freshly-built module Lightning loads into on resume.
        # Load non-strict so resuming a checkpoint that still carries
        # `_pbr_encoder.*` keys does not raise; the encoder is frozen and reloaded
        # from `pretrained_pbr_encoder` anyway. New checkpoints drop those keys
        # via on_save_checkpoint below.
        self.strict_loading = False

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Exclude the frozen PBR encoder from saved checkpoints.

        It is reconstructed from `pretrained_pbr_encoder` on load, so persisting
        it only bloats the checkpoint and (combined with lazy registration)
        breaks strict resume. Stripping keeps checkpoints lean and loadable.
        """
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            for key in [k for k in state_dict if k.startswith("_pbr_encoder.")]:
                del state_dict[key]

    def _ensure_pbr_encoder(self) -> Optional[torch.nn.Module]:
        """Lazily load the frozen PBR encoder on first use."""
        if self._pbr_encoder is not None:
            return self._pbr_encoder
        if self._pretrained_pbr_encoder_path is None:
            return None

        import trellis2.models as trellis_models
        print(f"[EmissionVaeTrainer] Loading frozen PBR encoder: {self._pretrained_pbr_encoder_path}")
        enc = trellis_models.from_pretrained(self._pretrained_pbr_encoder_path)
        enc.float().eval().to(self.device)
        for p in enc.parameters():
            p.requires_grad_(False)
        self._pbr_encoder = enc
        return self._pbr_encoder

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / max(1, torch.cuda.device_count()))),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _encode_material_latent(self, x: sp.SparseTensor) -> Optional[sp.SparseTensor]:
        """Encode PBR features into material latent using the frozen pretrained encoder."""
        pbr_enc = self._ensure_pbr_encoder()
        if pbr_enc is None:
            return None
        x_pbr = x.replace(x.feats[:, self.emission_channels:])
        with torch.no_grad():
            z_pbr = pbr_enc(x_pbr)
        return z_pbr

    def training_losses(
        self,
        x: sp.SparseTensor,
        **kwargs,
    ):
        """
        x : SparseTensor  [N, emission_ch + pbr_ch]
            Full combined features — emission first, then PBR material.
        """
        x_emit = x.replace(x.feats[:, : self.emission_channels])

        z, mean, logvar = self.training_models["encoder"](
            x_emit, sample_posterior=True, return_raw=True
        )

        z_pbr = self._encode_material_latent(x)
        if z_pbr is not None:
            assert z.coords.shape == z_pbr.coords.shape, (
                f"Emission/PBR latent coord shape mismatch: {z.coords.shape} vs {z_pbr.coords.shape}"
            )
            z_combined = z.replace(torch.cat([z.feats, z_pbr.feats], dim=-1))
        else:
            z_combined = z

        y = self.training_models["decoder"](z_combined)

        target = x.feats[:, : self.emission_channels]

        terms = edict(loss=0.0)
        if self.loss_type == "l1":
            terms["l1"] = l1_loss(target, y.feats)
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == "l2":
            terms["l2"] = l2_loss(target, y.feats)
            terms["loss"] = terms["loss"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type!r}")

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        return terms, {}

    @torch.no_grad()
    def snapshot_dataset(self, num_samples: int = 100, batch_size: int = 4):
        """Skip image-based dataset snapshot — sparse 3-D data not visualisable as images."""
        pass

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool = False) -> Dict:
        """Report mean-absolute emission reconstruction error on the training set."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=min(batch_size, len(self.dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
        )
        self.models["encoder"].eval()
        self.models["decoder"].eval()
        errors: List[float] = []
        for data in dataloader:
            data = recursive_to_device(data, self.device)
            x = data["x"]
            x_emit = x.replace(x.feats[:, : self.emission_channels])
            z = self.models["encoder"](x_emit)

            z_pbr = self._encode_material_latent(x)
            if z_pbr is not None:
                z = z.replace(torch.cat([z.feats, z_pbr.feats], dim=-1))

            y = self.models["decoder"](z)
            target = x.feats[:, : self.emission_channels]
            errors.append((target - y.feats).abs().mean().item())
            if len(errors) * batch_size >= num_samples:
                break
        self.models["encoder"].train()
        self.models["decoder"].train()
        mean_err = float(np.mean(errors)) if errors else 0.0
        return {"emission_recon_l1": {"value": torch.tensor(mean_err), "type": "scalar"}}
