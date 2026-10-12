"""Training-check hooks, loaded at interpreter start through PYTHONPATH by train-check.sh (and by
coverage.sh, which sets COVERAGE_PROCESS_START to start line coverage in every process).

Inactive unless TRELLIS2_TRAINCHECK_OUT names the record file (JSON lines). Then it:

1. seeds python, numpy and torch (CPU and every CUDA device) with TRAINCHECK_SEED (default 42) at
   start-up, before train.py builds the models (the DiT's re-initialized input projection takes a
   random init), and turns on PyTorch's deterministic algorithms (warn only: an op without a
   deterministic version still runs; two runs of unchanged code measure what is left);
2. reseeds right before every lightning Trainer.fit (the flow-matching noise, `sample_t` and the CFG
   mask draw from unseeded torch and numpy RNGs) and adds the Recorder callback;
3. records, per training step: global step, epoch, the batch's sha256 list, the learning rate, the
   loss, the gradient norm before the optimizer step and the sum of the trained weights after it;
   per validation: every val/* metric and the sum of every EMA copy; at the end of fit: the last
   global step. summarize_ckpt.py adds the saved checkpoint after the run.

train.py itself is not edited: everything here runs from outside the tree under test.
"""
import json
import os

_OUT = os.environ.get("TRELLIS2_TRAINCHECK_OUT")

# coverage.sh: trace every Python process of a check (subprocesses included) when it names an rcfile.
if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()


def _seed_all(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _install():
    import torch

    seed = int(os.environ.get("TRAINCHECK_SEED", "42"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    _seed_all(seed)

    import lightning.pytorch as pl

    out_path = str(_OUT)

    def _write(rec):
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _f(x):
        try:
            return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception:
            return None

    class Recorder(pl.Callback):
        def __init__(self):
            self._grad_norm = None

        @staticmethod
        def _trained(module):
            return [p for p in module.parameters() if p.requires_grad]

        def on_before_optimizer_step(self, trainer, module, optimizer):
            sq = [p.grad.detach().double().pow(2).sum() for p in self._trained(module) if p.grad is not None]
            self._grad_norm = float(torch.stack(sq).sum().sqrt()) if sq else 0.0

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            shas = batch.get("sha256") if isinstance(batch, dict) else None
            opt = trainer.optimizers[0] if trainer.optimizers else None
            wsum = sum(float(p.detach().double().sum()) for p in self._trained(module))
            _write({"kind": "train", "gs": int(trainer.global_step), "epoch": int(trainer.current_epoch),
                    "batch_idx": int(batch_idx), "shas": shas,
                    "lr": opt.param_groups[0]["lr"] if opt else None, "loss": _f(loss),
                    "grad_norm": self._grad_norm, "wsum": wsum})
            self._grad_norm = None

        def on_validation_epoch_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            metrics = {k: _f(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("val/")}
            ema = {}
            for cb in trainer.callbacks:
                if type(cb).__name__ == "EmaCallback":
                    for rate, bucket in getattr(cb, "_ema_state", {}).items():
                        ema[str(rate)] = _ema_sum(bucket)
            _write({"kind": "val", "gs": int(trainer.global_step), "epoch": int(trainer.current_epoch),
                    "metrics": metrics, "ema": ema})

        def on_fit_end(self, trainer, module):
            _write({"kind": "fit_end", "gs": int(trainer.global_step), "epoch": int(trainer.current_epoch)})

    def _ema_sum(val):
        total, n = 0.0, 0
        stack = [val]
        while stack:
            v = stack.pop()
            if isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
            elif isinstance(v, torch.Tensor):
                total += float(v.detach().double().sum()); n += v.numel()
        return {"sum": total, "numel": n}

    _fit = pl.Trainer.fit

    def fit(self, *args, **kwargs):
        if not any(type(cb).__name__ == "Recorder" for cb in self.callbacks):
            rec = Recorder()
            self.callbacks.append(rec)
        _seed_all(seed)
        _write({"kind": "fit_start", "gs": int(self.global_step), "seed": seed,
                "ckpt_path": kwargs.get("ckpt_path")})
        return _fit(self, *args, **kwargs)

    pl.Trainer.fit = fit


if _OUT:
    _install()
