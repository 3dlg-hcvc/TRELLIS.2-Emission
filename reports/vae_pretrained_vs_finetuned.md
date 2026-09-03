# Our emission VAE vs the pretrained PBR VAE — emission reconstruction on the test set (2026-08-27)

**Question.** Is our finetuned emission VAE better than using TRELLIS.2's pretrained PBR VAE
directly for emission (emission dropped into its albedo/base_color slot)?

**Answer.** Yes. On the 204 test shapes our VAE reconstructs emission with **30 % lower MAE**
(0.0440 → 0.0307 on the emissive voxels) and higher emission IoU (0.683 → 0.745); it is better on
**94 %** of the shapes (paired 95 % CI on the mean MAE delta [−0.017, −0.010]).

## The two rows

| | weights | encoder input | decoder | prediction |
|---|---|---|---|---|
| **pretrained PBR VAE** | `tex_{enc,dec}_next_dc_f16c32_fp16` from `microsoft/TRELLIS.2-4B`, untouched | emission in the base_color slot, metallic/roughness/alpha = the constants (−1, +1, +1) — exactly the packing our albedo→emission finetune uses, so the two rows differ only in weights and conditioning | 32-ch latent | output channels 0:3 |
| **ours** | the shipped agentic emission VAE `emission_vae_pbr2emission_74k_v3_agentic/ckpts/step0035200-0.0058.ckpt` — the decoder behind the paper's TRELLIS.2 row | emission (3-ch) | cat(z_emission 32, z_pbr 32), z_pbr from the frozen pretrained PBR encoder | 3-ch emission |

## Setup

- **Shapes:** the 204 `agentic_clean` **test** shapes
  (`evaluation/newdata_eval/eval_sets/test_agentic_clean.txt`) — the paper main table's test set
  (heuristic + agentic filtering). The checkpoint was selected on the validation split and never
  saw these shapes. Voxels: `data/lightgen_74k_newbake/{emission,pbr}_voxels_256/`
  (jupiter `trellis2_bw/lightgen_74k_newbake/`), resolution 256.
- **Recipe, identical for both:** encoder mean (`sample_posterior=False`), fp32 end to end, no
  autocast, strict weight loading (284/284 tensors per module; sha256 fingerprint of each weight
  set in `manifest.json`), per-shape metrics averaged with equal weight over shapes.
  Deterministic: two independent runs agree bit-for-bit on every metric.
- **Metrics** (per shape, then mean / median over shapes):
  `MAE` = mean |pred − GT| in [0, 1] emission space on the clamped prediction (the paper
  evaluator's `point_mae` convention); `L1 [-1,1]` = the same error in the training space
  (`loss_utils.l1_loss`; 2× MAE); `PSNR` in [0, 1], capped at 40 dB; `lit` = GT voxels whose max
  channel > 1/255 (the evaluator's IoU eps — GT-mask quantities like its `_gton` columns);
  `IoU` = (pred lit) vs (GT lit), empty union → 1.
  The *lit* columns are the informative ones: ~97 % of voxels are dark and both VAEs
  reconstruct black well, so the all-voxel PSNR saturates at the cap (median 40 dB).
- **Not comparable** to the training-time `val/loss` (fp16 autocast, sampled posterior, KL,
  validation split) nor to the point-sampled "VAE reconstruction ceiling" row in
  `docs/evaluation.md` (a different metric chain).

## Result (204 shapes)

| VAE | emission MAE ↓ | emission IoU ↑ |
|---|---|---|
| pretrained PBR VAE (emission in the albedo slot) | 0.0440 | 0.683 |
| **ours (agentic emission VAE)** | **0.0307** | **0.745** |

MAE = mean |pred − GT| over the GT-emissive voxels (max channel > 1/255), in [0, 1] emission
space; IoU = reconstructed-emissive vs GT-emissive voxels at the same threshold; both per shape,
then averaged over shapes. Paired per shape: MAE **−0.0134** (95 % bootstrap CI [−0.0170, −0.0103]),
ours better on **94 %** of the shapes; IoU **+0.062** [+0.052, +0.072], better on 85 %.

The other views (all-voxel MAE 0.0024 → 0.0017, PSNR on emissive voxels 24.66 → 27.29 dB,
medians, L1 in the [−1, 1] training space) are in
`reports/vae_recon_test204/summary.md` and all point the same way; the pretrained VAE also
under-emits (mean 0.261 vs GT 0.279 on emissive voxels; ours 0.270).

## Files

- Script: `data_toolkit/eval_vae_recon.py` (unit tests `data_toolkit/test_eval_vae_recon.py`, 10 pass).
  The default invocation runs exactly these two rows; optional control arms (real-PBR packing,
  albedo reconstruction, the albedo→emission `_v2`/`_v3` VAEs) exist behind flags and their
  numbers are in the CSV.
- Outputs: `reports/vae_recon_test204/{per_shape.csv, summary.md, summary.json, manifest.json}`;
  the manifest holds the sha256 fingerprint of every weight set scored, the shape-list hash, the
  data root and the metric version.
- Log: `log/<TS>-eval_vae_recon_test204_final.log`. Reviewed 2026-08-27 (four-lens adversarial
  review + two Codex passes; all confirmed findings fixed).

## Reproduce

```bash
cd TRELLIS2
conda run --no-capture-output -n trellis2 python data_toolkit/eval_vae_recon.py \
    --shapes ../evaluation/newdata_eval/eval_sets/test_agentic_clean.txt \
    --out_dir reports/vae_recon_test204 2>&1 | tee log/$(date +%Y%m%d-%H%M%S)-eval_vae_recon.log
# ~3 min on one RTX 4090. The checkpoint resolves to outputs/<run>/ckpts/ and falls back to the
# jupiter copy (/cs/3dlg-jupiter-project/lightgen/outputs/<run>/) when not cached locally.
# --summarize_only rebuilds summary.{md,json} from the CSV; --resume extends a run and refuses
# to mix weights / shapes / data / columns that differ from manifest.json and the CSV header.
```
