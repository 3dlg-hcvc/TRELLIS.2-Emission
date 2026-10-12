from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext
import os
import glob
import re
from abc import abstractmethod

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils as tv_utils

from trellis2.utils.data_utils import recursive_to_device
from trellis2.utils.general_utils import save_image_with_notes
from .utils import str_to_dtype
from .builders import build_lr_scheduler, build_models, build_optimizer


def flatten_dict(data: Dict[str, Any], sep: str = "/", prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, sep=sep, prefix=name))
        else:
            flat[name] = value
    return flat


class BasicTrainer(pl.LightningModule):
    def __init__(
        self,
        models: Dict[str, Any],
        dataset: Any = None,
        *,
        optimizer: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Dict[str, Any]] = None,
        grad_clip: Optional[Any] = None,
        fp16_mode: Optional[str] = None,
        mix_precision_mode: Optional[str] = None,
        mix_precision_dtype: Any = "float16",
        batch_split: int = 1,
        snapshot_batch_size: int = 4,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if optimizer is None:
            optimizer = {}

        self.models_cfg = models
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.grad_clip_cfg = grad_clip

        # Legacy compatibility: fp16_mode overrides mix_precision_mode in old configs.
        if fp16_mode is not None and mix_precision_mode is None:
            mix_precision_mode = "amp" if str(fp16_mode).lower() in ("amp", "inflat_all") else fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype) if isinstance(mix_precision_dtype, str) else mix_precision_dtype

        self.batch_split = int(batch_split)
        self.snapshot_batch_size = int(snapshot_batch_size)
        self._dataset = dataset
        self.output_dir = output_dir

        built = build_models(models)
        self.models = nn.ModuleDict(built)
        self.training_models = self.models

        self._grad_clipper = None
        self._pending_grad_clipper_state = None
        self._finetune_applied = False
        self._legacy_resume_applied = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def on_fit_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            self._dataset = getattr(dm, "train_dataset", None)
        resume_mode = getattr(self, "_resume_mode", None)
        legacy_resume_path = getattr(self, "_legacy_resume_path", None)
        if resume_mode == "legacy" and legacy_resume_path and not self._legacy_resume_applied:
            self._load_legacy_resume_checkpoint(legacy_resume_path)
            self._legacy_resume_applied = True
        if resume_mode in ("legacy", "lightning"):
            return
        self._apply_finetune_checkpoint()

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = build_optimizer(self.optimizer_cfg, params)
        scheduler = build_lr_scheduler(self.lr_scheduler_cfg, optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx=None,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        *args,
        **kwargs,
    ):
        if self.grad_clip_cfg is None:
            return

        if isinstance(self.grad_clip_cfg, (float, int)):
            torch.nn.utils.clip_grad_norm_(self.parameters(), float(self.grad_clip_cfg))
            return

        name = self.grad_clip_cfg.get("name")
        cfg_args = self.grad_clip_cfg.get("args", {})
        if name == "AdaptiveGradClipper":
            if self._grad_clipper is None:
                from trellis2.utils.grad_clip_utils import AdaptiveGradClipper

                self._grad_clipper = AdaptiveGradClipper(**cfg_args)
                if self._pending_grad_clipper_state is not None and hasattr(self._grad_clipper, "load_state_dict"):
                    self._grad_clipper.load_state_dict(self._pending_grad_clipper_state)
                    self._pending_grad_clipper_state = None
            self._grad_clipper(self.parameters())
            return

        from trellis2.utils import grad_clip_utils

        if self._grad_clipper is None:
            clipper_cls = getattr(grad_clip_utils, name, None)
            if clipper_cls is None:
                raise ValueError(f"Unknown grad clipper: {name}")
            self._grad_clipper = clipper_cls(**cfg_args)
            if self._pending_grad_clipper_state is not None and hasattr(self._grad_clipper, "load_state_dict"):
                self._grad_clipper.load_state_dict(self._pending_grad_clipper_state)
                self._pending_grad_clipper_state = None
        self._grad_clipper(self.parameters())

    def _autocast_context(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None or self.device.type != "cuda":
            return nullcontext()

        precision = getattr(trainer, "precision", None)
        if precision in ("16-mixed", "16"):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision in ("bf16-mixed", "bf16"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        if self.mix_precision_mode in ("amp", "mixed"):
            return torch.autocast(device_type="cuda", dtype=self.mix_precision_dtype)

        return nullcontext()

    def _extract_terms(self, output: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(output, tuple) and len(output) == 2:
            return output[0], output[1]
        if isinstance(output, dict):
            return output, {}
        raise TypeError(f"Unexpected training_losses output type: {type(output)}")

    def _log_terms(self, terms: Dict[str, Any], prefix: str, *, batch_size: int, on_step: bool, on_epoch: bool):
        flat = flatten_dict(terms)
        for key, value in flat.items():
            if not torch.is_tensor(value) and not isinstance(value, (float, int)):
                continue
            name = f"{prefix}/{key}"
            is_dynamic_bin = any(seg.startswith("bin_") for seg in key.split("/"))
            self.log(
                name,
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=(key == "loss"),
                batch_size=batch_size,
                sync_dist=not is_dynamic_bin,
                rank_zero_only=is_dynamic_bin,
            )

    def _is_global_zero(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return True
        return bool(getattr(trainer, "is_global_zero", True))

    def _warn(self, message: str) -> None:
        if self._is_global_zero():
            print(message)

    @staticmethod
    def _safe_torch_load(path: str, map_location: Any = "cpu"):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path, device=str(map_location))
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _dict_looks_like_state_dict(payload: Any) -> bool:
        if not isinstance(payload, dict) or not payload:
            return False
        if not all(isinstance(k, str) for k in payload.keys()):
            return False
        return any(torch.is_tensor(v) for v in payload.values())

    @staticmethod
    def _strip_prefix_from_state_dict(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        plen = len(prefix)
        return {k[plen:]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(prefix)}

    def _extract_state_dict_for_model(self, payload: Any, model_name: str) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            state_dict = payload.get("state_dict")
            if isinstance(state_dict, dict):
                extracted = self._extract_state_dict_for_model(state_dict, model_name)
                if extracted:
                    return extracted

            models_bucket = payload.get("models")
            if isinstance(models_bucket, dict):
                direct = models_bucket.get(model_name)
                if self._dict_looks_like_state_dict(direct):
                    return dict(direct)

            direct = payload.get(model_name)
            if self._dict_looks_like_state_dict(direct):
                return dict(direct)

            if self._dict_looks_like_state_dict(payload):
                for prefix in (f"models.{model_name}.", f"{model_name}.", f"model.{model_name}."):
                    stripped = self._strip_prefix_from_state_dict(payload, prefix)
                    if stripped:
                        return stripped
                return dict(payload)
        return None

    @staticmethod
    def _slice_value(value: Any, start: int, end: int) -> Any:
        if isinstance(value, dict):
            return {k: BasicTrainer._slice_value(v, start, end) for k, v in value.items()}
        if isinstance(value, list):
            return value[start:end]
        if isinstance(value, tuple):
            return type(value)(BasicTrainer._slice_value(v, start, end) for v in value[start:end])
        try:
            return value[start:end]
        except Exception:
            return value

    @staticmethod
    def _infer_batch_size_from_dict(batch: Dict[str, Any]) -> int:
        x0 = batch.get("x_0")
        if hasattr(x0, "shape"):
            return int(x0.shape[0])
        if hasattr(x0, "layout"):
            return int(len(x0.layout))
        for value in batch.values():
            if hasattr(value, "shape"):
                return int(value.shape[0])
            if hasattr(value, "layout"):
                return int(len(value.layout))
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1

    def _normalize_microbatches(self, batch: Any) -> List[Dict[str, Any]]:
        if isinstance(batch, list):
            out = [b for b in batch if isinstance(b, dict)]
            if out:
                return out
            raise TypeError("Batch list is expected to contain dict microbatches.")
        if not isinstance(batch, dict):
            raise TypeError(f"Unexpected batch type: {type(batch)}")
        if self.batch_split <= 1:
            return [batch]
        batch_size = self._infer_batch_size_from_dict(batch)
        if batch_size <= 1 or batch_size % self.batch_split != 0:
            return [batch]
        shard = batch_size // self.batch_split
        return [{k: self._slice_value(v, i * shard, (i + 1) * shard) for k, v in batch.items()} for i in range(self.batch_split)]

    @staticmethod
    def _aggregate_terms(terms_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        sums: Dict[str, Any] = {}
        counts: Dict[str, Any] = {}

        def merge(sum_dst: Dict[str, Any], cnt_dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            for key, value in src.items():
                if isinstance(value, dict):
                    child_sum = sum_dst.setdefault(key, {})
                    child_cnt = cnt_dst.setdefault(key, {})
                    merge(child_sum, child_cnt, value)
                    continue
                if not torch.is_tensor(value) and not isinstance(value, (float, int)):
                    continue
                if key not in sum_dst:
                    sum_dst[key] = value.detach() if torch.is_tensor(value) else float(value)
                    cnt_dst[key] = 1
                else:
                    sum_dst[key] = sum_dst[key] + (value.detach() if torch.is_tensor(value) else float(value))
                    cnt_dst[key] = int(cnt_dst[key]) + 1

        def finalize(sum_src: Dict[str, Any], cnt_src: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for key, value in sum_src.items():
                if isinstance(value, dict):
                    nested = finalize(value, cnt_src.get(key, {}))
                    if nested:
                        out[key] = nested
                    continue
                denom = max(1, int(cnt_src.get(key, 1)))
                out[key] = value / denom
            return out

        for terms in terms_list:
            merge(sums, counts, terms)
        return finalize(sums, counts)

    def _batch_effective_size(self, batch: Any, microbatches: List[Dict[str, Any]]) -> int:
        if isinstance(batch, dict):
            return self._infer_batch_size_from_dict(batch)
        if isinstance(batch, list):
            return sum(self._infer_batch_size_from_dict(mb) for mb in microbatches)
        return 1

    @staticmethod
    def _resolve_hf_path(path: str) -> str:
        """If path looks like a HuggingFace model path (e.g. microsoft/TRELLIS.2-4B/ckpts/name),
        resolve it to a local file by downloading via hf_hub_download."""
        if os.path.exists(path) or os.path.exists(path + ".safetensors"):
            return path
        parts = path.split("/")
        if len(parts) >= 3 and not path.startswith("/"):
            from huggingface_hub import hf_hub_download
            repo_id = f"{parts[0]}/{parts[1]}"
            model_name = "/".join(parts[2:])
            for ext in [".safetensors", ".pt", ".bin"]:
                try:
                    return hf_hub_download(repo_id, f"{model_name}{ext}")
                except Exception:
                    continue
        raise FileNotFoundError(f"Cannot resolve checkpoint path: {path}")

    def _load_weights_for_model(self, model_name: str, path: str) -> None:
        if not os.path.exists(path):
            path = self._resolve_hf_path(path)
        if os.path.basename(path).startswith("misc_step") and path.endswith(".pt"):
            step = self._extract_step_from_legacy_path(path)
            if step is not None:
                candidate = os.path.join(os.path.dirname(path), f"{model_name}_step{step:07d}.pt")
                if os.path.exists(candidate):
                    path = candidate
        payload = self._safe_torch_load(path, map_location="cpu")
        model_state = self._extract_state_dict_for_model(payload, model_name)
        if not isinstance(model_state, dict):
            raise ValueError(f"Unable to extract state_dict for '{model_name}' from: {path}")
        target_state = self.models[model_name].state_dict()
        filtered = {}
        for key, value in model_state.items():
            if key not in target_state:
                self._warn(f"[finetune] Warning: {model_name}.{key} not in model; skipped.")
                continue
            if hasattr(value, "shape") and hasattr(target_state[key], "shape") and tuple(value.shape) != tuple(target_state[key].shape):
                self._warn(
                    f"[finetune] Warning: {model_name}.{key} shape mismatch "
                    f"{tuple(value.shape)} vs {tuple(target_state[key].shape)}; skipped."
                )
                continue
            filtered[key] = value
        missing, unexpected = self.models[model_name].load_state_dict(filtered, strict=False)
        if missing:
            self._warn(f"[finetune] {model_name}: missing keys: {len(missing)}")
        if unexpected:
            self._warn(f"[finetune] {model_name}: unexpected keys: {len(unexpected)}")

    def _apply_finetune_checkpoint(self) -> None:
        if self._finetune_applied:
            return
        finetune_ckpt = getattr(self, "finetune_ckpt", None)
        if not isinstance(finetune_ckpt, dict) or not finetune_ckpt:
            self._finetune_applied = True
            return
        self._warn("[finetune] Loading finetune checkpoints")
        for model_name in self.models.keys():
            ckpt_path = finetune_ckpt.get(model_name)
            if not ckpt_path:
                self._warn(f"[finetune] Warning: {model_name} not found in finetune_ckpt; skipped.")
                continue
            self._warn(f"[finetune] {model_name}: {ckpt_path}")
            self._load_weights_for_model(model_name, ckpt_path)
        self._finetune_applied = True

    @staticmethod
    def _extract_step_from_legacy_path(path: str) -> Optional[int]:
        match = re.search(r"step(\d+)\.pt$", os.path.basename(path))
        if match is None:
            return None
        return int(match.group(1))

    def _set_trainer_global_step(self, step: int) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        def set_attr_path(root: Any, path: str, value: int) -> None:
            obj = root
            parts = path.split(".")
            for name in parts[:-1]:
                if not hasattr(obj, name):
                    return
                obj = getattr(obj, name)
            last = parts[-1]
            if hasattr(obj, last):
                try:
                    setattr(obj, last, value)
                except Exception:
                    pass

        candidates = [
            "fit_loop.epoch_loop._batches_that_stepped",
            "fit_loop.epoch_loop.batch_progress.total.completed",
            "fit_loop.epoch_loop.batch_progress.current.completed",
            "fit_loop.epoch_progress.current.completed",
        ]
        for path in candidates:
            set_attr_path(trainer, path, step)

    def _load_legacy_resume_checkpoint(self, misc_ckpt_path: str) -> None:
        if not os.path.exists(misc_ckpt_path):
            raise FileNotFoundError(f"Legacy resume checkpoint not found: {misc_ckpt_path}")
        step = self._extract_step_from_legacy_path(misc_ckpt_path)
        misc_ckpt = self._safe_torch_load(misc_ckpt_path, map_location="cpu")
        if step is None:
            step = int(misc_ckpt.get("step", 0))
        ckpt_dir = os.path.dirname(misc_ckpt_path)
        self._warn(f"[legacy-resume] Loading step {step} from {misc_ckpt_path}")

        for model_name in self.models.keys():
            model_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_step{step:07d}.pt")
            if not os.path.exists(model_ckpt_path):
                candidates = sorted(glob.glob(os.path.join(ckpt_dir, f"{model_name}_step*.pt")))
                if not candidates:
                    raise FileNotFoundError(f"Legacy checkpoint missing model weights for '{model_name}' at step {step}")
                model_ckpt_path = candidates[-1]
            self._load_weights_for_model(model_name, model_ckpt_path)

        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            optimizer_state = misc_ckpt.get("optimizer")
            if optimizer_state is not None and getattr(trainer, "optimizers", None):
                trainer.optimizers[0].load_state_dict(optimizer_state)

            scheduler_state = misc_ckpt.get("lr_scheduler")
            if scheduler_state is not None:
                configs = getattr(trainer, "lr_scheduler_configs", [])
                if configs:
                    scheduler = getattr(configs[0], "scheduler", None)
                    if scheduler is not None:
                        scheduler.load_state_dict(scheduler_state)

            sampler_state = misc_ckpt.get("data_sampler")
            if sampler_state is not None:
                datamodule = getattr(trainer, "datamodule", None)
                sampler = None
                if datamodule is not None:
                    sampler = getattr(datamodule, "train_sampler", None)
                if sampler is not None and hasattr(sampler, "load_state_dict"):
                    sampler.load_state_dict(sampler_state)

            elastic_state = misc_ckpt.get("elastic_controller")
            if elastic_state is not None:
                for cb in getattr(trainer, "callbacks", []):
                    if cb.__class__.__name__ != "ElasticControllerCallback":
                        continue
                    if getattr(cb, "controller", None) is not None:
                        cb.controller.load_state_dict(elastic_state)
                    else:
                        cb._pending_state = elastic_state
                    break

            grad_clip_state = misc_ckpt.get("grad_clip")
            if grad_clip_state is not None:
                if self._grad_clipper is not None and hasattr(self._grad_clipper, "load_state_dict"):
                    self._grad_clipper.load_state_dict(grad_clip_state)
                else:
                    self._pending_grad_clipper_state = grad_clip_state

            self._set_trainer_global_step(step)

    def training_step(self, batch: Any, batch_idx: int):
        microbatches = self._normalize_microbatches(batch)
        terms_list: List[Dict[str, Any]] = []
        losses: List[torch.Tensor] = []
        for mb_data in microbatches:
            with self._autocast_context():
                terms, _ = self._extract_terms(self.training_losses(**mb_data))
            terms_list.append(terms)
            losses.append(terms["loss"])
        loss = torch.stack([l if torch.is_tensor(l) else torch.tensor(float(l), device=self.device) for l in losses]).mean()
        terms = self._aggregate_terms(terms_list)
        batch_size = self._batch_effective_size(batch, microbatches)
        self._log_terms(terms, "train", batch_size=batch_size, on_step=True, on_epoch=True)
        # Log current learning rate
        opts = self.optimizers()
        if opts is not None:
            opt = opts[0] if isinstance(opts, (list, tuple)) else opts
            lr = opt.param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False,
                     prog_bar=False, batch_size=batch_size, sync_dist=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        microbatches = self._normalize_microbatches(batch)
        terms_list: List[Dict[str, Any]] = []
        for mb_data in microbatches:
            with self._autocast_context():
                terms, _ = self._extract_terms(self.training_losses(**mb_data))
            terms_list.append(terms)
        terms = self._aggregate_terms(terms_list)
        batch_size = self._batch_effective_size(batch, microbatches)
        self._log_terms(terms, "val", batch_size=batch_size, on_step=False, on_epoch=True)
        return terms

    @torch.no_grad()
    def visualize_sample(self, sample):
        if self.dataset is not None and hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        return sample

    def _get_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "default_root_dir", None):
            return str(trainer.default_root_dir)
        return os.getcwd()

    @torch.no_grad()
    def snapshot_dataset(self, num_samples: int = 100, batch_size: int = 4):
        if self.dataset is None:
            return
        if not hasattr(self.dataset, "visualize_sample"):
            return
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[: min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            vis = self.visualize_sample(data)
            if isinstance(vis, dict):
                for key, val in vis.items():
                    save_cfg.setdefault(f"dataset_{key}", []).append(val)
            else:
                save_cfg.setdefault("dataset", []).append(vis)

        out_dir = self._get_output_dir()
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        value_range = getattr(self.dataset, "value_range", (0, 1))
        for name, image in save_cfg.items():
            tv_utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(out_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix: Optional[str] = None, num_samples: int = 64, batch_size: int = 4, verbose: bool = False):
        if suffix is None:
            step = int(getattr(getattr(self, "trainer", None), "global_step", 0))
            suffix = f"step{step:07d}"

        with self._autocast_context():
            samples = self.run_snapshot(num_samples=num_samples, batch_size=batch_size, verbose=verbose)

        for key in list(samples.keys()):
            if samples[key]["type"] == "sample":
                vis = self.visualize_sample(samples[key]["value"])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f"{key}_{k}"] = {"value": v, "type": "image"}
                    del samples[key]
                else:
                    samples[key] = {"value": vis, "type": "image"}

        out_dir = self._get_output_dir()
        sample_dir = os.path.join(out_dir, "samples", suffix)
        os.makedirs(sample_dir, exist_ok=True)
        value_range = getattr(self.dataset, "value_range", (0, 1)) if self.dataset is not None else (0, 1)

        for key, payload in samples.items():
            if payload["type"] == "image":
                tv_utils.save_image(
                    payload["value"],
                    os.path.join(sample_dir, f"{key}_{suffix}.jpg"),
                    nrow=max(1, int(np.sqrt(num_samples))),
                    normalize=True,
                    value_range=value_range,
                )
            elif payload["type"] == "number":
                min_val = payload["value"].min()
                max_val = payload["value"].max()
                image = (payload["value"] - min_val) / (max_val - min_val + 1e-8)
                image = tv_utils.make_grid(image, nrow=max(1, int(np.sqrt(num_samples))), normalize=False)
                save_image_with_notes(
                    image,
                    os.path.join(sample_dir, f"{key}_{suffix}.jpg"),
                    notes=f"{key} min: {min_val}, max: {max_val}",
                )

    @abstractmethod
    def training_losses(self, **mb_data):
        pass

    @abstractmethod
    def run_snapshot(self, num_samples: int, batch_size: int = 4, verbose: bool = False) -> Dict[str, Any]:
        pass
