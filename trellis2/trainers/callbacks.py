from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch import nn


class ElasticControllerCallback(pl.Callback):
    def __init__(self, elastic_cfg: Optional[Dict[str, Any]]) -> None:
        super().__init__()
        self.elastic_cfg = elastic_cfg
        self.controller = None
        self._ctx = None
        self._pending_state = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.elastic_cfg:
            return
        from trellis2.utils import elastic_utils

        name = self.elastic_cfg.get("name")
        args = self.elastic_cfg.get("args", {})
        controller_cls = getattr(elastic_utils, name, None)
        if controller_cls is None:
            raise ValueError(f"Unknown elastic controller: {name}")
        controller = controller_cls(**args)

        elastic_models = []
        models = getattr(pl_module, "models", None)
        if isinstance(models, (dict, nn.ModuleDict)):
            for model in models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    elastic_models.append(model)
        if not elastic_models:
            if trainer.is_global_zero:
                print("[ElasticControllerCallback] No elastic modules found; disabling controller.")
            return

        for model in elastic_models:
            model.register_memory_controller(controller)
        self.controller = controller
        if self._pending_state is not None:
            self.controller.load_state_dict(self._pending_state)
            self._pending_state = None

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        if self.controller is None:
            return
        self._ctx = self.controller.record()
        self._ctx.__enter__()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        if self.controller is None:
            return
        metrics = self.controller.log()
        if metrics:
            for key, value in metrics.items():
                pl_module.log(f"elastic/{key}", value, on_step=True, on_epoch=False, sync_dist=True)

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        if self.controller is not None:
            checkpoint["elastic_controller"] = self.controller.state_dict()

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        if self.controller is None:
            self._pending_state = checkpoint.get("elastic_controller")
            return
        state = checkpoint.get("elastic_controller")
        if state is not None:
            self.controller.load_state_dict(state)


class EmaCallback(pl.Callback):
    def __init__(self, ema_rates: Optional[list[float]]) -> None:
        super().__init__()
        if ema_rates is None:
            ema_rates = []
        self.ema_rates = [float(r) for r in ema_rates]
        self._ema_state: Dict[str, Dict[str, Any]] = {}
        self._param_names: list[str] = []
        self._params: list[Any] = []

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.ema_rates:
            return
        names = []
        params = []
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                names.append(name)
                params.append(param)
        self._param_names = names
        self._params = params
        if not self._ema_state:
            for rate in self.ema_rates:
                self._ema_state[str(rate)] = {name: p.detach().clone() for name, p in zip(names, params)}
        else:
            for key, bucket in self._ema_state.items():
                self._ema_state[key] = {n: t.to(pl_module.device) for n, t in bucket.items()}

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self._ema_state:
            return
        with torch.no_grad():
            for rate in self.ema_rates:
                bucket = self._ema_state.get(str(rate))
                if bucket is None:
                    continue
                for name, param in zip(self._param_names, self._params):
                    ema_param = bucket[name]
                    ema_param.mul_(rate).add_(param.detach(), alpha=1.0 - rate)

    @contextmanager
    def apply_ema(self, pl_module: "pl.LightningModule", rate: Optional[float] = None):
        if not self._ema_state:
            yield
            return
        if rate is None:
            rate = self.ema_rates[0]
        bucket = self._ema_state.get(str(rate))
        if bucket is None:
            yield
            return
        backup = {name: param.detach().clone() for name, param in zip(self._param_names, self._params)}
        try:
            for name, param in zip(self._param_names, self._params):
                param.data.copy_(bucket[name])
            yield
        finally:
            for name, param in zip(self._param_names, self._params):
                param.data.copy_(backup[name])

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        if self._ema_state:
            checkpoint["ema"] = {
                "rates": self.ema_rates,
                "state": {k: {n: t.cpu() for n, t in v.items()} for k, v in self._ema_state.items()},
            }

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        ema = checkpoint.get("ema")
        if not ema:
            return
        rates = [float(r) for r in ema.get("rates", [])]
        state = ema.get("state", {})
        self.ema_rates = rates
        self._ema_state = {}
        for key, bucket in state.items():
            self._ema_state[str(key)] = {n: t for n, t in bucket.items()}


class SamplerStateCallback(pl.Callback):
    def _get_sampler(self, trainer: "pl.Trainer"):
        datamodule = trainer.datamodule
        if datamodule is not None and getattr(datamodule, "train_sampler", None) is not None:
            return datamodule.train_sampler
        loaders = trainer.train_dataloader
        loader = loaders[0] if isinstance(loaders, list) else loaders
        if loader is not None:
            return getattr(loader, "sampler", None)
        return None

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        sampler = self._get_sampler(trainer)
        if sampler is not None and hasattr(sampler, "state_dict"):
            checkpoint["sampler_state"] = sampler.state_dict()

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        sampler = self._get_sampler(trainer)
        state = checkpoint.get("sampler_state")
        if sampler is not None and state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state)


class SnapshotCallback(pl.Callback):
    def __init__(
        self,
        *,
        snapshot_batch_size: int = 4,
        val_num_samples: int = 16,
        sample_every_n_steps: int = 0,
        skip_initial_samples: bool = False,
    ) -> None:
        super().__init__()
        self.snapshot_batch_size = int(snapshot_batch_size)
        self.val_num_samples = int(val_num_samples)
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.skip_initial_samples = bool(skip_initial_samples)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.is_global_zero:
            return
        if hasattr(pl_module, "snapshot_dataset"):
            pl_module.snapshot_dataset(batch_size=self.snapshot_batch_size)
        if not self.skip_initial_samples and hasattr(pl_module, "snapshot"):
            pl_module.snapshot(suffix="init", num_samples=self.val_num_samples, batch_size=self.snapshot_batch_size)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        if not trainer.is_global_zero:
            return
        if self.sample_every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0:
            return
        if step % self.sample_every_n_steps != 0:
            return
        if hasattr(pl_module, "snapshot"):
            pl_module.snapshot(
                suffix=f"step{step:07d}",
                num_samples=self.val_num_samples,
                batch_size=self.snapshot_batch_size,
            )

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.is_global_zero:
            return
        if hasattr(pl_module, "snapshot"):
            pl_module.snapshot(suffix="final", num_samples=self.val_num_samples, batch_size=self.snapshot_batch_size)

# ----------------------------------------------------------------------------------------------------
# LightGen's training callbacks; train.py calls them in four places:
#   * make_wandb_logger: the WandbLogger, with the run id kept in <output_dir>/wandb_run_id.txt so a
#     requeued job continues the same wandb run;
#   * build_checkpoint_callback: top-k by val/loss plus last.ckpt every ckpt_last_every_n_steps, or
#     upstream's step-interval mode;
#   * CheckpointTimingCallback: the [ckpt-timing] lines; the snapshot watchers parse SAVE_START and
#     RESUME_AFTER_SAVE;
#   * TrainTraceCallback: the per-step [train] and per-validation [val] lines (formerly
#     BasicTrainer.on_train_batch_end / on_validation_epoch_end). It is the last callback, as
#     Lightning calls the module's own hooks after every callback's.


def make_wandb_logger(project, name, entity, output_dir):
    from lightning.pytorch.loggers import WandbLogger

    # Persist the wandb run ID across requeues so resumed jobs continue the
    # same wandb run instead of starting a fresh one each cycle. Delete
    # {output_dir}/wandb_run_id.txt to force a new run (e.g. when starting
    # a new experiment from the same output_dir).
    wandb_id_path = os.path.join(output_dir, "wandb_run_id.txt")
    wandb_id = None
    if os.path.exists(wandb_id_path):
        try:
            with open(wandb_id_path) as f:
                wandb_id = (f.read().strip() or None)
        except Exception:
            wandb_id = None

    logger = WandbLogger(
        project=project,
        name=name,
        entity=entity,
        save_dir=output_dir,
        id=wandb_id,
        resume="allow" if wandb_id else None,
    )

    # First-run only (rank 0): persist the auto-generated run ID for later
    # requeues. Accessing `logger.experiment.id` triggers wandb.init() so
    # the ID is available; non-rank-0 instances of WandbLogger return a
    # dummy experiment whose .id is None, which we ignore.
    if wandb_id is None and int(os.environ.get("SLURM_PROCID", "0")) == 0:
        try:
            new_id = logger.experiment.id
            if new_id:
                os.makedirs(output_dir, exist_ok=True)
                with open(wandb_id_path, "w") as f:
                    f.write(new_id)
        except Exception:
            pass
    return logger


def build_checkpoint_callback(trainer_args: Dict[str, Any], ckpt_dir: str) -> ModelCheckpoint:
    """Construct the run's ModelCheckpoint callback.

    Two modes:
      * Top-k val mode (``save_top_k`` set): keep the K best checkpoints by
        ``monitor_metric`` (written at validation-epoch end) plus a ``last.ckpt``
        for resume. ``last.ckpt`` is refreshed every ``ckpt_last_every_n_steps``
        steps (default 200) so a hard crash loses at most that many steps; on a
        graceful (SIGTERM) requeue the trainer's signal handler still saves exact
        state. NOTE: never set this to 1 — rewriting the multi-GB checkpoint every
        step blocks training on a Lustre write each step (~90% of wall-clock; see
        CheckpointTimingCallback). The top-k save itself is independent of this
        interval — it fires at validation-epoch end via the monitor metric.
      * Legacy step-interval mode (``i_save`` set, no ``save_top_k``): keep all
        checkpoints at the ``i_save`` step interval.
    """
    i_save = trainer_args.get("i_save")
    save_top_k_cfg = trainer_args.get("save_top_k")
    if save_top_k_cfg is not None:
        monitor_metric = str(trainer_args.get("monitor_metric", "val/loss"))
        monitor_mode = str(trainer_args.get("monitor_mode", "min")).lower()
        if monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")
        filename = "step{step:07d}-{" + monitor_metric + ":.4f}"
        last_every_n_steps = int(trainer_args.get("ckpt_last_every_n_steps", 200))
        return ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=filename,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=int(save_top_k_cfg),
            save_last=True,
            every_n_train_steps=last_every_n_steps,
            auto_insert_metric_name=False,
        )
    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:07d}",
        every_n_train_steps=(int(i_save) if i_save is not None else None),
        save_last=True,
        save_top_k=-1,
    )


class CheckpointTimingCallback(pl.Callback):
    """Log checkpoint save/load events with timestamps + per-step batch time
    every N steps. Lets us correlate KL/L1 spikes against ckpt I/O.

    Emits to stdout (rank 0 only) — gets picked up by the tee log:

        [ckpt-timing] LOAD step=2420 t=1747312345.12
        [ckpt-timing] SAVE_START step=4488 t=...
        [ckpt-timing] RESUME_AFTER_SAVE step=4489 gap=132.4s t=...
        [ckpt-timing] BATCH_TIME step=4500 dt=1.42s
    """

    def __init__(self, batch_log_every: int = 100) -> None:
        super().__init__()
        self.batch_log_every = int(batch_log_every)
        self._save_t0: Optional[float] = None
        self._batch_t0: Optional[float] = None

    def _emit(self, msg: str) -> None:
        # Rank-0 only via Lightning's is_global_zero (set via trainer at runtime).
        # We can't easily check rank here without trainer; on_*_checkpoint
        # callbacks already only fire on rank 0 in Lightning's default
        # checkpoint flow. on_train_batch_* fires on every rank, so we gate
        # those manually via the trainer arg.
        print(msg, flush=True)

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        step = checkpoint.get("global_step", "?")
        epoch = checkpoint.get("epoch", "?")
        self._emit(f"[ckpt-timing] LOAD step={step} epoch={epoch} t={time.time():.3f}")

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", 0))
        self._save_t0 = time.time()
        self._emit(f"[ckpt-timing] SAVE_START step={step} epoch={epoch} t={self._save_t0:.3f}")

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        # First train batch after a save → roughly when ckpt write + post-save
        # callback chain finished blocking the trainer.
        if self._save_t0 is not None:
            dt = time.time() - self._save_t0
            step = int(getattr(trainer, "global_step", 0))
            self._emit(
                f"[ckpt-timing] RESUME_AFTER_SAVE step={step} gap={dt:.2f}s t={time.time():.3f}"
            )
            self._save_t0 = None
        self._batch_t0 = time.time()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        if self._batch_t0 is None:
            return
        dt = time.time() - self._batch_t0
        self._batch_t0 = None
        step = int(getattr(trainer, "global_step", 0))
        if self.batch_log_every > 0 and step > 0 and step % self.batch_log_every == 0:
            self._emit(f"[ckpt-timing] BATCH_TIME step={step} dt={dt:.2f}s")


class TrainTraceCallback(pl.Callback):
    """The [train] line per logged step and the [val] line per validation, rank 0 only."""

    @staticmethod
    def _global_zero(trainer, pl_module) -> bool:
        is_zero = getattr(pl_module, "_is_global_zero", None)
        return bool(is_zero()) if callable(is_zero) else bool(getattr(trainer, "is_global_zero", True))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._global_zero(trainer, pl_module):
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        val_items = []
        for key in sorted(metrics.keys()):
            if not str(key).startswith("val/"):
                continue
            value = metrics[key]
            try:
                value = float(value.item()) if torch.is_tensor(value) else float(value)
            except Exception:
                continue
            val_items.append((str(key), value))
        if not val_items:
            return
        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", 0))
        formatted = "  ".join(f"{k}={v:.4f}" for k, v in val_items)
        print(f"[val] step={step}  epoch={epoch}  {formatted}", flush=True)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int) -> None:
        # Per-step text trace symmetric to [val]. Cadence follows
        # Trainer.log_every_n_steps (= i_log) so it stays aligned with the
        # wandb / CSV scalar flush — and survives `tee` piping that mangles
        # Lightning's TTY-only progress bar.
        if not self._global_zero(trainer, pl_module):
            return
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        log_every = int(getattr(trainer, "log_every_n_steps", 1) or 1)
        if step % log_every != 0:
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        train_items = []
        for key in sorted(metrics.keys()):
            if not str(key).startswith("train/"):
                continue
            # Skip the on_epoch aggregates — they only update at epoch boundary
            # and would be redundant with the per-step *_step values.
            if str(key).endswith("_epoch"):
                continue
            value = metrics[key]
            try:
                value = float(value.item()) if torch.is_tensor(value) else float(value)
            except Exception:
                continue
            train_items.append((str(key), value))
        if not train_items:
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        formatted = "  ".join(f"{k}={v:.4f}" for k, v in train_items)
        print(f"[train] step={step}  epoch={epoch}  {formatted}", flush=True)
