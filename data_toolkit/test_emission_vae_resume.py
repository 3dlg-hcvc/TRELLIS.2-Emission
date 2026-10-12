"""Regression test for the pbr→emission EmissionVaeTrainer checkpoint-resume crash.

Bug (see memory project_twostream_vae_resume_crash): the frozen `_pbr_encoder`
is lazily registered on the first training step, so it gets baked into every
saved checkpoint. On resume, Lightning runs `load_state_dict(strict=True)`
*before* the first step (when `_pbr_encoder` is still None), so the checkpoint's
`_pbr_encoder.*` keys are "Unexpected key(s)" and training crashes on every
requeue.

These tests are CPU-only (no GPU / data / pretrained weights needed) and assert
the two-part fix:
  1. `EmissionVaeTrainer.__init__` sets `strict_loading = False`  -> resume
     tolerates an existing checkpoint that already carries `_pbr_encoder.*`.
  2. `EmissionVaeTrainer.on_save_checkpoint` strips `_pbr_encoder.*` -> future
     checkpoints stay lean and load cleanly even under strict loading.

Run:  python data_toolkit/test_emission_vae_resume.py
"""
import sys
import torch
import torch.nn as nn

from trellis2.trainers.vae.emission_vae import EmissionVaeTrainer


def test_init_sets_strict_loading_false():
    """Fix #1: a freshly-built trainer must load non-strict so the existing
    fat checkpoint (with `_pbr_encoder.*`) can be resumed."""
    t = EmissionVaeTrainer(models={}, pretrained_pbr_encoder=None)
    assert t.strict_loading is False, f"expected strict_loading=False, got {t.strict_loading!r}"


def test_on_save_checkpoint_strips_pbr_encoder():
    """Fix #2: saving must drop the frozen PBR encoder while keeping the
    trainable model + its EMA copy."""
    t = EmissionVaeTrainer.__new__(EmissionVaeTrainer)  # bypass heavy __init__
    state_dict = {
        "_pbr_encoder.input_layer.weight": torch.zeros(2, 2),
        "_pbr_encoder.blocks.0.0.norm.bias": torch.zeros(2),
        "models.encoder.layer.weight": torch.ones(2, 2),          # EMA copy
        "training_models.encoder.layer.weight": torch.ones(2, 2),  # live copy
    }
    ckpt = {"state_dict": dict(state_dict), "global_step": 2006}
    t.on_save_checkpoint(ckpt)
    remaining = list(ckpt["state_dict"].keys())
    assert not any(k.startswith("_pbr_encoder.") for k in remaining), \
        f"_pbr_encoder.* was not stripped: {remaining}"
    assert "models.encoder.layer.weight" in remaining, "trainable EMA weights were dropped"
    assert "training_models.encoder.layer.weight" in remaining, "live trainable weights were dropped"
    assert ckpt["global_step"] == 2006, "non-state_dict checkpoint contents must be untouched"


def test_resume_load_tolerates_saved_pbr_encoder():
    """End-to-end: reproduce the strict-load crash, then show the stripped
    checkpoint loads under strict=True without raising."""
    # Model as reconstructed at resume time: _pbr_encoder is still lazy (absent).
    model = nn.Module()
    model.models = nn.ModuleDict({"encoder": nn.Linear(2, 2)})
    fresh_sd = model.state_dict()

    # Checkpoint as actually saved mid-training: same keys + the frozen encoder.
    saved_sd = dict(fresh_sd)
    saved_sd["_pbr_encoder.input_layer.weight"] = torch.zeros(2, 2)

    # Reproduce the bug: strict load rejects the extra _pbr_encoder.* key.
    raised = False
    try:
        model.load_state_dict(saved_sd, strict=True)
    except RuntimeError:
        raised = True
    assert raised, "bug not reproduced: strict load should reject _pbr_encoder.*"

    # Fix: on_save_checkpoint strips it, so the result loads under strict=True.
    t = EmissionVaeTrainer.__new__(EmissionVaeTrainer)
    ckpt = {"state_dict": dict(saved_sd)}
    t.on_save_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=True)  # must not raise


def main():
    tests = [
        test_init_sets_strict_loading_false,
        test_on_save_checkpoint_strips_pbr_encoder,
        test_resume_load_tolerates_saved_pbr_encoder,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
