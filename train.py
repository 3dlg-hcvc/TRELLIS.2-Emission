import argparse
import glob
import inspect
import json
import os
import re
import sys
from datetime import timedelta
from copy import deepcopy
from typing import Any, Dict, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise ImportError("PyYAML is required for Lightning configs.") from exc

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from trellis2.trainers.callbacks import (
    CheckpointTimingCallback,
    ElasticControllerCallback,
    EmaCallback,
    SamplerStateCallback,
    SnapshotCallback,
)
from trellis2.trainers.data import TrellisDataModule
from trellis2 import trainers as trellis_trainers


TRAINER_MAP = {
    "SparseStructureVaeTrainer": trellis_trainers.SparseStructureVaeTrainer,
    "ShapeVaeTrainer": trellis_trainers.ShapeVaeTrainer,
    "PbrVaeTrainer": trellis_trainers.PbrVaeTrainer,
    "EmissionVaeTrainer": trellis_trainers.EmissionVaeTrainer,
    "FlowMatchingTrainer": trellis_trainers.FlowMatchingTrainer,
    "FlowMatchingCFGTrainer": trellis_trainers.FlowMatchingCFGTrainer,
    "TextConditionedFlowMatchingCFGTrainer": trellis_trainers.TextConditionedFlowMatchingCFGTrainer,
    "ImageConditionedFlowMatchingCFGTrainer": trellis_trainers.ImageConditionedFlowMatchingCFGTrainer,
    "SparseFlowMatchingTrainer": trellis_trainers.SparseFlowMatchingTrainer,
    "SparseFlowMatchingCFGTrainer": trellis_trainers.SparseFlowMatchingCFGTrainer,
    "TextConditionedSparseFlowMatchingCFGTrainer": trellis_trainers.TextConditionedSparseFlowMatchingCFGTrainer,
    "ImageConditionedSparseFlowMatchingCFGTrainer": trellis_trainers.ImageConditionedSparseFlowMatchingCFGTrainer,
    "MultiImageConditionedSparseFlowMatchingCFGTrainer": trellis_trainers.MultiImageConditionedSparseFlowMatchingCFGTrainer,
}


def _maybe_patch_flex_gemm():
    try:
        import flex_gemm.ops.spconv.submanifold_conv3d as subm
    except Exception:
        return

    func = getattr(subm, "SubMConv3dFunction", None)
    if func is None:
        return
    original = getattr(func, "_sparse_submanifold_conv_backward", None)
    if original is None or getattr(original, "_trellis_patch", False):
        return

    Algorithm = subm.Algorithm
    spconv = subm.spconv
    kernels = subm.kernels

    def _patched(grad_output, feats, neighbor_cache, weight, bias=None):
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kd * Kh * Kw

        if spconv.ALGORITHM == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache["neighbor_map"]

            if feats.requires_grad:
                im2col = torch.zeros((N * V, Co), device=feats.device, dtype=feats.dtype)
                inv_neighbor_map = torch.flip(neighbor_map, [1])
                mask = inv_neighbor_map.view(-1) != 0xFFFFFFFF
                im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Co)

                grad_input = torch.mm(
                    im2col,
                    weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci),
                )
            else:
                grad_input = None

            if weight.requires_grad:
                im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
                mask = neighbor_map.view(-1) != 0xFFFFFFFF
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Ci)

                grad_weight = (
                    torch.mm(im2col.t(), grad_output.view(N, -1))
                    .view(V, Ci, Co)
                    .permute(2, 0, 1)
                    .contiguous()
                    .view(Co, Kw, Kh, Kd, Ci)
                )
            else:
                grad_weight = None

            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None

        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache["neighbor_map"],
            )
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)

        elif spconv.ALGORITHM == Algorithm.IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache["neighbor_map"],
            )
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)

        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache["neighbor_map"],
                neighbor_cache["sorted_idx"],
                neighbor_cache["valid_kernel_callback"],
                neighbor_cache["valid_kernel_seg_callback"],
                neighbor_cache["valid_signal_i"],
                neighbor_cache["valid_signal_o"],
                neighbor_cache["valid_signal_seg"],
            )
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)

        elif spconv.ALGORITHM == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(Co, Kd * Kh * Kw, Ci),
                bias,
                neighbor_cache["neighbor_map"],
                neighbor_cache["sorted_idx"],
                neighbor_cache["valid_kernel_callback"],
                neighbor_cache["valid_kernel_seg_callback"],
                neighbor_cache["valid_signal_i"],
                neighbor_cache["valid_signal_o"],
                neighbor_cache["valid_signal_seg"],
            )
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)

        else:
            raise ValueError(f"Unsupported algorithm {spconv.ALGORITHM}")

        return grad_input, grad_weight, grad_bias

    _patched._trellis_patch = True
    func._sparse_submanifold_conv_backward = staticmethod(_patched)


def _parse_override_value(value: str):
    try:
        return json.loads(value)
    except Exception:
        return value


def _apply_override(cfg_dict, key_path: str, value):
    parts = key_path.split(".")
    cur = cfg_dict
    for part in parts[:-1]:
        if isinstance(cur, dict):
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        else:
            raise ValueError(f"Cannot set override '{key_path}' on non-dict node.")
    if not isinstance(cur, dict):
        raise ValueError(f"Cannot set override '{key_path}' on non-dict node.")
    cur[parts[-1]] = value


def apply_overrides(cfg_dict, overrides):
    if not overrides:
        return
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in key=value form: {override}")
        key, value = override.split("=", 1)
        key = key.strip()
        value = _parse_override_value(value.strip())
        _apply_override(cfg_dict, key, value)


def load_config(path: str) -> Dict[str, Any]:
    if not path.endswith((".yaml", ".yml")):
        raise ValueError("Only YAML configs are supported.")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _get_latest_ckpt(load_dir: str) -> Optional[str]:
    ckpt_dir = os.path.join(load_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _get_latest_legacy_misc_ckpt(load_dir: str) -> Optional[str]:
    ckpt_dir = os.path.join(load_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "misc_step*.pt")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _get_legacy_misc_ckpt_by_step(load_dir: str, step: str) -> Optional[str]:
    if not step.isdigit():
        return None
    ckpt_dir = os.path.join(load_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        return None
    exact = os.path.join(ckpt_dir, f"misc_step{int(step):07d}.pt")
    if os.path.exists(exact):
        return exact
    matches = glob.glob(os.path.join(ckpt_dir, f"misc_step*{step}*.pt"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _is_legacy_misc_ckpt(path: Optional[str]) -> bool:
    if not path:
        return False
    name = os.path.basename(path)
    return bool(path.endswith(".pt") and re.match(r"^misc_step\d+\.pt$", name))


def resolve_ckpt_path(load_dir: str, ckpt: str) -> Optional[str]:
    if not ckpt or ckpt == "none":
        return None
    if ckpt in ("latest", "last"):
        path = os.path.join(load_dir, "ckpts", "last.ckpt")
        if os.path.exists(path):
            return path
        return _get_latest_ckpt(load_dir) or _get_latest_legacy_misc_ckpt(load_dir)
    if ckpt.isdigit():
        pattern = os.path.join(load_dir, "ckpts", f"*{ckpt}*.ckpt")
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
        return _get_legacy_misc_ckpt_by_step(load_dir, ckpt)
    if os.path.exists(ckpt):
        return ckpt
    return None


def get_model_summary(model):
    model_summary = "Parameters:\n"
    model_summary += "=" * 128 + "\n"
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f"{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n"
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += "\n"
    model_summary += f"Number of parameters: {num_params}\n"
    model_summary += f"Number of trainable parameters: {num_trainable_params}\n"
    return model_summary


def build_module(cfg: Dict[str, Any]) -> pl.LightningModule:
    trainer_cfg = cfg.get("trainer", {})
    trainer_name = trainer_cfg.get("name")
    if trainer_name not in TRAINER_MAP:
        raise ValueError(f"Unknown trainer '{trainer_name}'.")

    trainer_args = deepcopy(trainer_cfg.get("args", {}))
    module_args: Dict[str, Any] = {
        "models": cfg.get("models", {}),
        "optimizer": trainer_args.get("optimizer", {}),
        "lr_scheduler": trainer_args.get("lr_scheduler"),
        "grad_clip": trainer_args.get("grad_clip"),
    }
    for key, value in trainer_args.items():
        if key in {"optimizer", "lr_scheduler", "grad_clip"}:
            continue
        module_args[key] = value

    module_cls = TRAINER_MAP[trainer_name]
    return module_cls(**module_args)


def build_datamodule(cfg: Dict[str, Any], data_dir: str, trainer_args: Dict[str, Any]) -> TrellisDataModule:
    dataset_cfg = deepcopy(cfg.get("dataset", {}))
    dataset_args = dict(dataset_cfg.get("args", {}))
    if "roots" not in dataset_args and "root" not in dataset_args:
        dataset_args["roots"] = data_dir
    dataset_cfg["args"] = dataset_args

    batch_size = trainer_args.get("batch_size_per_gpu") or trainer_args.get("batch_size")
    if batch_size is None:
        raise ValueError("trainer.args.batch_size_per_gpu (or batch_size) is required.")

    split_size = trainer_args.get("batch_split")
    if split_size is not None and int(split_size) <= 1:
        split_size = None

    dm = TrellisDataModule(
        dataset=dataset_cfg,
        batch_size=int(batch_size),
        num_workers=int(trainer_args.get("dataloader_num_workers", 0)),
        pin_memory=True,
        persistent_workers=bool(trainer_args.get("dataloader_num_workers", 0) > 0),
        val_split_file=trainer_args.get("val_split_file"),
        val_split_name=trainer_args.get("val_split_name"),
        val_batch_size=trainer_args.get("val_batch_size_per_gpu") or trainer_args.get("val_batch_size"),
        use_balanced_sampler=trainer_args.get("use_balanced_sampler"),
        split_size=split_size,
    )
    return dm


def build_logger(cfg: Dict[str, Any], output_dir: str):
    trainer_args = cfg.get("trainer", {}).get("args", {})
    if not trainer_args.get("wandb_enable", False):
        return CSVLogger(save_dir=output_dir, name="logs")
    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception as exc:  # pragma: no cover
        raise ImportError("wandb logger requested but not available.") from exc

    project = trainer_args.get("wandb_project", "trellis2")
    name = trainer_args.get("wandb_name")
    entity = trainer_args.get("wandb_entity")

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

    config_payload = trainer_args.get("wandb_config", cfg)
    try:
        logger.experiment.config.update(config_payload, allow_val_change=True)
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


def build_trainer(
    cfg: Dict[str, Any],
    output_dir: str,
    dm: TrellisDataModule,
    devices: int,
    num_nodes: int,
    tryrun: bool,
):
    trainer_args = cfg.get("trainer", {}).get("args", {})
    callbacks = []

    ema_rate = trainer_args.get("ema_rate")
    if ema_rate is not None:
        if isinstance(ema_rate, (float, int)):
            ema_rates = [float(ema_rate)]
        else:
            ema_rates = [float(r) for r in ema_rate]
        callbacks.append(EmaCallback(ema_rates))

    if trainer_args.get("elastic") is not None:
        callbacks.append(ElasticControllerCallback(trainer_args.get("elastic")))

    callbacks.append(SamplerStateCallback())

    # Diagnostic: log ckpt save/load events + per-100-step batch time so we
    # can correlate KL/L1 spikes against ckpt I/O (Lustre is slow for 13 GB
    # writes; spikes-after-save and spikes-after-resume are both possible).
    callbacks.append(CheckpointTimingCallback(batch_log_every=100))

    sample_every_n_steps = int(trainer_args.get("i_sample", 0) or 0)
    callbacks.append(
        SnapshotCallback(
            snapshot_batch_size=int(trainer_args.get("snapshot_batch_size", 4)),
            val_num_samples=int(trainer_args.get("val_num_samples", 16)),
            sample_every_n_steps=sample_every_n_steps,
            skip_initial_samples=bool(trainer_args.get("skip_initial_samples", False)),
        )
    )

    ckpt_dir = os.path.join(output_dir, "ckpts")
    callbacks.append(build_checkpoint_callback(trainer_args, ckpt_dir))

    mix_mode = trainer_args.get("mix_precision_mode")
    if mix_mode is None and trainer_args.get("fp16_mode") is not None:
        fp16_mode = str(trainer_args.get("fp16_mode")).lower()
        mix_mode = "amp" if fp16_mode in ("amp", "inflat_all") else fp16_mode
    mix_dtype = str(trainer_args.get("mix_precision_dtype", "float16")).lower()
    precision = "32"
    if mix_mode in ("amp", "fp16", "mixed"):
        precision = "bf16-mixed" if mix_dtype in ("bfloat16", "bf16") else "16-mixed"

    log_every_n_steps = int(trainer_args.get("wandb_log_every_steps", trainer_args.get("i_log", 50)))
    accumulate_grad_batches = max(1, int(trainer_args.get("batch_split", 1)))
    if dm is not None and getattr(dm, "split_size", None) is not None:
        accumulate_grad_batches = 1

    limit_val_batches = trainer_args.get("limit_val_batches", 1.0)
    val_max_samples = trainer_args.get("val_max_samples")
    val_batch_size = trainer_args.get("val_batch_size_per_gpu") or trainer_args.get("val_batch_size") or trainer_args.get("batch_size_per_gpu")
    if val_max_samples and val_batch_size and trainer_args.get("limit_val_batches") is None:
        limit_val_batches = max(1, int((val_max_samples + val_batch_size - 1) // val_batch_size))

    check_val_every_n_epoch = int(trainer_args.get("val_every_epochs", 1))
    logger = build_logger(cfg, output_dir)

    strategy = None
    if devices > 1 or num_nodes > 1:
        try:
            from lightning.pytorch.strategies import DDPStrategy

            ddp_timeout_minutes = int(trainer_args.get("ddp_timeout_minutes", 180))
            strategy = DDPStrategy(
                find_unused_parameters=bool(trainer_args.get("ddp_find_unused_parameters", False)),
                static_graph=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=128,
                timeout=timedelta(minutes=ddp_timeout_minutes),
            )
        except Exception:
            strategy = "ddp"

    trainer_kwargs = dict(
        default_root_dir=output_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        limit_val_batches=limit_val_batches,
    )

    if trainer_args.get("limit_train_batches") is not None:
        trainer_kwargs["limit_train_batches"] = trainer_args.get("limit_train_batches")
    if trainer_args.get("num_sanity_val_steps") is not None:
        trainer_kwargs["num_sanity_val_steps"] = int(trainer_args["num_sanity_val_steps"])

    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    if trainer_args.get("max_steps") is not None:
        trainer_kwargs["max_steps"] = int(trainer_args["max_steps"])
    elif trainer_args.get("max_epochs") is not None:
        trainer_kwargs["max_epochs"] = int(trainer_args["max_epochs"])

    if tryrun:
        trainer_kwargs["max_steps"] = 1
        trainer_kwargs["limit_train_batches"] = 1
        trainer_kwargs["limit_val_batches"] = 1
        trainer_kwargs["num_sanity_val_steps"] = 0

    trainer_sig = inspect.signature(pl.Trainer.__init__)
    if "replace_sampler_ddp" in trainer_sig.parameters:
        trainer_kwargs["replace_sampler_ddp"] = False
    elif "use_distributed_sampler" in trainer_sig.parameters:
        trainer_kwargs["use_distributed_sampler"] = False

    return pl.Trainer(**trainer_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment config file (YAML)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--load_dir", type=str, default="", help="Load directory, default to output_dir")
    parser.add_argument("--ckpt", type=str, default="last", help="Checkpoint to resume: last/latest/none/path")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory")
    parser.add_argument("--auto_retry", type=int, default=0, help="Number of retries on error")
    parser.add_argument("--tryrun", action="store_true", help="Try run without training")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank (handled by launcher)")
    parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs per node, default to all")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Unused; Lightning handles DDP")
    parser.add_argument("--master_port", type=str, default="12345", help="Unused; Lightning handles DDP")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with dot paths. Example: --override trainer.args.batch_size_per_gpu=2",
    )
    opt = parser.parse_args()

    _maybe_patch_flex_gemm()

    opt.load_dir = opt.load_dir if opt.load_dir else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus

    cfg = load_config(opt.config)
    apply_overrides(cfg, opt.override)
    print("\n\nConfig:")
    print("=" * 80)
    print(yaml.safe_dump(cfg, sort_keys=False))

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "samples"), exist_ok=True)

    with open(os.path.join(opt.output_dir, "command.txt"), "w") as fp:
        print(" ".join(["python"] + sys.argv), file=fp)
    with open(os.path.join(opt.output_dir, "config.yaml"), "w") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)

    trainer_args = cfg.get("trainer", {}).get("args", {})
    dm = build_datamodule(cfg, opt.data_dir, trainer_args)
    dm.setup("fit")

    module = build_module(cfg)
    module.output_dir = opt.output_dir
    module.dataset = getattr(dm, "train_dataset", None)

    for name, model in getattr(module, "models", {}).items():
        model_summary = get_model_summary(model)
        print(f"\n\nBackbone: {name}\n{model_summary}")
        with open(os.path.join(opt.output_dir, f"{name}_model_summary.txt"), "w") as fp:
            print(model_summary, file=fp)

    ckpt_path = resolve_ckpt_path(opt.load_dir, opt.ckpt)
    resume_mode = None
    fit_ckpt_path = None
    if ckpt_path is not None:
        if _is_legacy_misc_ckpt(ckpt_path):
            resume_mode = "legacy"
        elif ckpt_path.endswith(".ckpt"):
            resume_mode = "lightning"
            fit_ckpt_path = ckpt_path
        elif ckpt_path.endswith(".pt"):
            raise ValueError(
                f"Unsupported .pt checkpoint path for --ckpt: {ckpt_path}. "
                "Use legacy misc_step*.pt for resume, or set trainer.args.finetune_ckpt for per-model finetune."
            )
    trainer = build_trainer(cfg, opt.output_dir, dm, opt.num_gpus, opt.num_nodes, opt.tryrun)
    module._resume_mode = resume_mode
    module._legacy_resume_path = ckpt_path if resume_mode == "legacy" else None

    def run_fit():
        fit_kwargs = dict(datamodule=dm, ckpt_path=fit_ckpt_path)
        if fit_ckpt_path is not None:
            fit_sig = inspect.signature(trainer.fit)
            if "weights_only" in fit_sig.parameters:
                fit_kwargs["weights_only"] = False
            else:
                os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        trainer.fit(module, **fit_kwargs)

    if opt.auto_retry <= 0:
        run_fit()
    else:
        for rty in range(opt.auto_retry):
            try:
                run_fit()
                break
            except Exception as exc:
                print(f"Error: {exc}")
                print(f"Retrying ({rty + 1}/{opt.auto_retry})...")


if __name__ == "__main__":
    main()
