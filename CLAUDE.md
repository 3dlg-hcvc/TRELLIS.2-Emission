# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRELLIS.2-Lightning is a 4B-parameter 3D generative model using sparse voxel representations (O-Voxel) for image-to-3D generation with full PBR materials. The `lightgen` branch extends it to generate emissive textures for 3D objects.

## Environment Setup

```bash
# Full install (conda env "trellis2", Python 3.10, PyTorch 2.6.0 + CUDA 12.4)
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Rebuild just the O-Voxel C++ extension
cd o-voxel && pip install -e . && cd ..
```

## Training Commands

```bash
# Launch training (single example)
python train.py --config configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml \
  --output_dir outputs/emission_vae_pbr2emission_lightgenbench_v1 \
  --override dataset.args.data_root=<root>   # and parquet_file, split_json; see script/internal/alliance/lightgenbench_v1.sh

# Resume from checkpoint
python train.py --config <CONFIG> --output_dir <DIR> --ckpt last

# Override config values at runtime
python train.py --config <CONFIG> --output_dir <DIR> --override trainer.args.max_steps=5000

# Dry run (validate config without training)
python train.py --config <CONFIG> --output_dir <DIR> --tryrun

# Multi-GPU / multi-node
python train.py --config <CONFIG> --output_dir <DIR> --num_gpus 4 --num_nodes 2 --node_rank 0
```

### Run logging policy

Every training run must produce **two** logs in parallel:

1. **Stdout tee → `log/<TS>-<NAME>.log`** — full reproducibility trace (config dump, GPU info, stack traces). General pattern: `<TS>=$(date +%Y%m%d-%H%M%S)`, `<NAME>` is the config basename (or job tag for cluster runs).

   ```bash
   mkdir -p log
   TS=$(date +%Y%m%d-%H%M%S)
   CFG=<config_basename>          # e.g. matches configs/scvae/<CFG>.yaml
   python train.py --config "configs/scvae/${CFG}.yaml" \
     --output_dir "outputs/${CFG}" 2>&1 | tee "log/${TS}-${CFG}.log"
   ```

2. **Wandb is mandatory.** Every training config must enable wandb so scalar metrics, samples, and the resolved config end up in the dashboard. Set the following under `trainer.args:`:

   ```yaml
   trainer:
     args:
       wandb_enable: true
       wandb_project: <project>          # e.g. "emission-pbr-vae-finetune"
       wandb_name: <run_name>            # default; override per-submission
       # wandb_entity: <team>            # optional
       # wandb_log_every_steps: <N>      # optional, defaults to i_log
   ```

   Per-submission, override `wandb_name` so every job gets its own dashboard row, e.g. `--override trainer.args.wandb_name=${NAME}` in the launch script. The wandb logger is wired in `train.py:build_logger`; if `wandb_enable` is false (or missing) it silently falls back to `CSVLogger` and **no metrics reach wandb** — do not ship a config in that state.

   Authentication on a cluster node: `wandb login <key>` once, or `export WANDB_API_KEY=...`. To run without network: `export WANDB_MODE=offline` and `wandb sync` later.

Apply the `log/${TS}-<name>.log` tee pattern to data-prep scripts too (e.g. `log/${TS}-prep_split.log`) and inside cluster job bodies (see "Cluster launch scripts" below).

#### Logging-interval gotcha (small datasets)

Lightning flushes scalars every `log_every_n_steps` (= `i_log`, default `50` in `train.py`). If `total_train_batches_per_epoch < log_every_n_steps`, the run can finish having flushed **zero** scalars to wandb / the CSV. For tiny / overfit datasets, set `i_log` (or `wandb_log_every_steps`) low enough that at least one flush happens per epoch — typically `i_log: 1` for `<N>sample` smoke runs, `i_log: 10` for `<N>00sample`, `i_log: 50` for `<N>k`+.

### Cluster launch scripts (star2)

SBATCH wrappers live under `script/internal/star2/<task>.sh` — one script per experiment family. They `ssh star2 sbatch <<EOF ... EOF` to submit to the cluster. **The SBATCH job body must use zsh** (the user's login shell on cluster nodes); the outer wrapper that runs `ssh ... sbatch` locally stays bash. Inside the heredoc:

- Shebang: `#!/bin/zsh`.
- `source ~/.zshrc` before `conda activate trellis2`.
- Pipe exit codes: zsh uses 1-indexed lowercase `${pipestatus[1]}` — **not** bash's `${PIPESTATUS[0]}` — when capturing `train.py`'s exit through a `tee` pipe. End the body with `exit \$TRAIN_EXIT` so Slurm records training failures as failures.
- Tee logs into `log/${TS}-${NAME}.log` per the Run Logs convention (the `mkdir -p log` + `TS=$(date +%Y%m%d-%H%M%S)` block lives inside the SBATCH body so the timestamp is the actual run start on the node).
- **Sync the launcher's commit to the cluster.** Never edit source files on the cluster; edit locally, `git push origin <branch>`, and put `git pull origin <branch>` in the sbatch body right after `cd /localscratch/dya78/TRELLIS.2-lightning`. This makes "what code ran" unambiguous and lets multiple collaborators share the cluster clone without racing edits.
- **Regenerate gitignored artifacts in the body.** Files under `data/` (e.g. `data_splits_*.json`) are gitignored and won't arrive via `git pull` — regenerate them in the body so a fresh cluster clone works end-to-end.

Naming convention: `script/internal/star2/<task>.sh` ↔ job name / output dir `<task>[_<TAG>]` ↔ status report `reports/<task>_status.md`. When adding a new launch script, start from the closest one in tag `pre-refactor` (`git show pre-refactor:script/star2/<file>`; only `backup_run.sh` is kept here) rather than writing from scratch.

#### Interactive debug allocation (probe before sbatch)

**Decision rule:** any config change that could OOM or skew DDP (new batch size, sampler, dataset) gets a ~100-step interactive probe before the full sbatch. Sparse-voxel runs have rank-skewed memory and sbatch crashes burn hours of wall-clock — a probe pays for itself on the first avoided crash.

Allocation flow (see the cluster-access skill for full tmux/srun rationale; keep tmux **local**, not star2-side):

```bash
# From local workstation — session name = node name
tmux new-session -d -s cs-venus-15 \
  'ssh star2 -t "srun --nodelist=cs-venus-15 --gres=gpu:l40s:4 \
  --cpus-per-task=32 --mem=256G --time=8:00:00 --pty zsh -l"'
tmux a -t cs-venus-15
```

Once attached (inside the compute-node shell):
```bash
cd /localscratch/dya78/TRELLIS.2-lightning && conda activate trellis2
git pull --ff-only origin <branch>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p log; TS=$(date +%Y%m%d-%H%M%S)
python train.py --config configs/scvae/<cfg>.yaml --output_dir outputs/<run> \
  --num_gpus 4 --ckpt last \
  --override trainer.args.batch_size_per_gpu=<probe_bs> \
  --override trainer.args.max_steps=<current_step+100> 2>&1 \
  | tee "log/${TS}-probe_bs<probe_bs>.log"
```

Probe passes if it reaches a full val epoch without OOM and with sane `val/loss`. With e.g. BS=8 + TRAIN_SIZE=640 + `val_every_epochs=5`, 100 steps = 5 epochs = 1 val epoch — exactly right. Then commit the config change and submit the full sbatch. When done, exit the compute shell (`--pty` releases on shell exit), or `scancel <jobid>` if it's stuck, then `tmux kill-session -t cs-venus-15`.

Gotcha: `nvidia-smi` via a second `srun --jobid=<id> --overlap` is scoped to the overlap step's cgroup GPU devices — that subset is what you see, *not* the parent training step's full GRES binding. Don't conclude "only 1 GPU is used" from it; the Lightning progress bar (`N/N ... it/s` where N matches `total_batches/rank × ranks`) is the authoritative DDP-across-all-ranks signal.

### Fir deployment (Alliance Canada H100)

Deployed at `/scratch/dya78/TRELLIS.2-lightning/` (`lightgen` branch), venv at `env/`; the fir launch scripts were under `script/fir/` (deleted; tag `pre-refactor` keeps them), the release runs use `script/internal/alliance/`. Unlike star2, three deps are **not** plain pip installs:

- **pyarrow** and **opencv** ship as Alliance *modules*, not wheels. Every fir job body must `module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0` **before** `source env/bin/activate`.
- **diffusers** was missing from the env: `env/bin/pip install --no-index diffusers`.
- **H100 node memory:** request `--mem=1024G` for a 4-GPU full node (= 4×256G). `--mem=1152G` is rejected ("Memory specification can not be satisfied"; allocatable cap is 1125G).
- Pull code from origin to advance; never edit on the cluster (see the cluster-access skill).

**PBR-VAE max batch size** (single H100 80GB, `EmissionPbrFinetuneTrainer`, fp16 + activation checkpointing): hard cap ≈ BS 88 on 1k data (BS 80 = 98% VRAM knife-edge; BS 96 OOMs). For 74k production (fatter voxel-count tail → +15–20% VRAM at the same BS), use **BS=48** (safe, ~76% on 1k) or BS=64 (aggressive — will occasionally OOM, needs `--requeue`). Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Data Paths

- **Data root:** `data/lightgen_test/` — repo-relative path that works on every host. The 10-sample dataset (parquet, emission voxels, PBR voxels, latents, splits) lives here as real files. The `data/` directory is gitignored. To deploy on a new host, copy/rsync `data/lightgen_test/` along with the repo (e.g. `rsync -av data/ host:/path/to/repo/data/`).
- **74k subset (jupyter):** `/3dlg-jupiter-project/lightgen/dataset/lightgen_74k/` (or `/cs/3dlg-jupiter-project/...` from the jupyter side). Holds `metadata.csv`, `df_SomgProc_final.parquet`, `data_splits_emissive_74k_pinned.json`, plus 8 `voxels/part<N>_voxels.tar` and 8 `latents/part<N>_latents.tar` produced by the Blender prep pipeline (deleted; tag `pre-refactor` keeps it).
  - **Current encoding coverage** (as of 2026-05-08): both `pbr_latents/tex_enc_next_dc_f16c32_fp16_256/` and `shape_latents/shape_enc_next_dc_f16c32_fp16_256/` are present per shard. Per-shard counts vary slightly because of per-sample Blender failures on bad GLBs (~1 % loss) plus, for shard 4, **348 dual-grid samples were skipped** because their meshes exceeded a 500 GB single-process memory budget (pathological vertex count). Net coverage ≈ **99 %** of the 73,472 sha256s once the rank-2 backfill finishes (rank 2's first attempt landed only 2,328 samples due to a partial-state artifact during early debug iteration; the backfill is in flight on cs-venus-16).
  - **Data-prep operational notes** (lessons from the first 74k run):
    - **Stage A (Blender dump) is pure CPU** — never request `--gres=gpu` for the AB job; that wastes a GPU for hours. Stage C (latent encoders) does need `--gres=gpu:1`.
    - **`/localscratch` is shared across users on each compute node**; cs-venus-{15,16}'s 5.8 TB volume hovers around 100 % full from other users. Per-rank `pbr_dumps/` peaks at ~270 GB; running 4 ranks concurrently per node blows the disk. Cap at 1–2 ranks per node, or use the inline pickle-cleanup pattern (delete each `.pickle` after voxelize+dual_grid consume it; the one-off `r2_resume.sh` that demonstrated this has been removed).
    - **Sparse-voxel dual_grid spikes RAM unpredictably**; some meshes need >500 GB single-process RAM. Ask for ≥256 GB with few workers, or accept a few-percent loss on pathological samples.
    - **Conda env `trellis2` is per-node**: only cs-venus-{07,08,09,15,16} have `/localscratch/dya78/miniconda3/envs/trellis2/`. cs-venus-{02,05,13,14,17} lack it and any AB/C SBATCH that lands there fails in <1 s.
    - **`pyarrow` is missing from the `trellis2` env** by default; `encode_shape_from_dual_grid.py` needs it to read the parquet. The C launcher idempotently `pip install pyarrow` before invoking encoders.
- **Visualization scripts:** `vis/` — unified 3D interactive: `vis/vis_3d_vae.py` (VAE reconstruction) and `vis/vis_3d_dit.py` (DiT → VAE decode). Both share helpers in `vis/vis_utils.py` and support `--experiment name=path` (repeatable) for side-by-side method comparison.

  **Known issue — GLB↔voxel scale misalignment:** The GLB panel and voxel panels don't bbox-align perfectly. Root cause: Blender's GLTF importer (used by `data_toolkit/blender_script/dump_pbr.py` to produce the pickle the voxelizer consumes) and the browser's GLTFLoader decode the same `.glb` file into geometrically different vertex sets (different vertex counts, different per-axis extents) — Blender applies import-time axis conversion, modifiers, and other transforms not exposed as toggles. The voxels come from Blender's view of the mesh; the browser renders trimesh/GLTFLoader's view. No single transform can reconcile them. Sample 0 happens to match closely (~0.003 on normalized axes); samples with more complex hierarchy show per-axis extent differences of 10–30% of the unit cube. Fix path (not yet done): during data prep, export a GLB from Blender at dump time (via `bpy.ops.export_scene.gltf`) as a sidecar file alongside each `.vxz`, then the viewer loads that "roundtrip" GLB instead of the 1K original — the two share Blender's exact vertex set.
- **Visualization output:** `visualizations/` — generated HTML/PNG files go here.
- **Web publishing:** copy HTML visualizations to `/project/3dlg-hcvc/lightgen/www/` for browser access.

## Data Preparation Pipeline

Run in order before training:
```bash
. ./data_toolkit/setup.sh                                  # setup
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT>  # metadata
python data_toolkit/dump_mesh.py <SUBSET> --root <ROOT>        # extract meshes
python data_toolkit/dump_pbr.py <SUBSET> --root <ROOT>         # extract PBR + emissive textures
python data_toolkit/voxelize_pbr.py ...                        # mesh → O-Voxel
python data_toolkit/encode_pbr_latent.py ...                   # voxel → latent
python data_toolkit/encode_emission_twostream_dataset.py ...  # emission voxel → latent (lightgen)
```

## Testing

No test framework beyond a few pytest files (CPU; the two `vis_utils_test.py` checkpoint tests need
`outputs/emission_vae_10sample_overfit/ckpts/`):
```bash
python -m pytest data_toolkit/test_eval_vae_recon.py data_toolkit/test_emission_vae_resume.py \
  data_toolkit/test_checkpoint_cadence.py vis/vis_utils_test.py
python vis/vis_3d_vae.py                             # 3D interactive HTML visualization
```

## Working conventions

- **Visualizations: one HTML file with a sample dropdown**, not one file per sample. The pattern in `vis/vis_3d_*.py` (single HTML, embedded JSON, JS dropdown to switch samples) is the standard — don't emit N separate files.
- **Label viz panels by what the method does, in plain terms**, not internal jargon. Mapping: channel-hijack / `emission_pbr_*` → **"albedo→emission"**; two-stream / `emission_vae_*` → **"pbr→emission"**. Pass plain `--experiment "<plain name>=<path>"` labels (rendered as plain text).
- **Name shared-utility modules descriptively** (`vis_utils.py`), not with a leading underscore (`_vis_common.py`). Reserve `_`-prefixed names for truly single-caller private helpers.
- **Before finetuning a pretrained model, verify the architecture config matches the checkpoint first** — extract the arch params from the pretrained ckpt JSON and copy them verbatim into the training config (don't guess `num_blocks` etc.), then do a short test-run before launching. Repeated PBR-VAE finetune failures traced back to config/weight mismatch.

## Evaluation lives in the parent repo

This submodule's DiT/VAE are evaluated from the parent lightgen repo via a voxel→UV bridge into
`evaluation/evaluate_texge.py`. Toolchain + metric guide: parent
[`docs/evaluation.md`](../docs/evaluation.md) (baseline stub:
[`docs/baselines/trellis2.md`](../docs/baselines/trellis2.md)). When working inside this submodule
only, those docs are the missing context.

## Architecture

### Two-Stage Generation Pipeline

**Stage 1 — Sparse Compact VAE (SC-VAE):** Encodes 3D voxel attributes into compact latent spaces. Three separate VAEs handle shape, PBR materials, and emission respectively. All share the `SparseUnetVaeEncoder`/`Decoder` architecture (sparse 3D U-Net with residual blocks).

**Stage 2 — Flow Matching DiT:** Generates latents conditioned on input images. Uses `ElasticSLatFlowModel` — a sparse transformer with flow matching objective, DINOv3 image conditioning (cross-attention), and classifier-free guidance.

### Emission (LightGen) Specifics

Two distinct approaches are implemented for the emission VAE — pick the right one when picking a config/trainer.

#### PBR VAE finetune via channel hijack (`EmissionPbrFinetuneTrainer`)

Architecture is **byte-identical** to the production PBR VAE (`tex_{enc,dec}_next_dc_f16c32_fp16`): 6→6 input/output channels, 64→1024 model channels, latent_channels=32. **100% pretrained-weight reuse**, no input/output projection mismatch.

Trick (in `trellis2/trainers/vae/emission_pbr_finetune.py:_pack_emission_as_pbr`): pack emission RGB into the `base_color` slot and fill `metallic / roughness / alpha` with constants `(-1, +1, +1)` in [-1, 1] space. Loss is taken only on the first 3 output channels (emission); the trailing 3 channels of the decoder output are discarded.

I/O note: with the default `pbr_attrs: [base_color, metallic, roughness, alpha]`, `LightGenSLatEmission` reads `pbr_voxels_{res}/{sha}.vxz` for every sample, only for the trainer to overwrite those features with constants. Set `pbr_attrs: []` in the dataset args to skip the read — it's a no-op for the model.

Configs: none kept (the `emission_pbr_*` family was deleted in the refactor; tag `pre-refactor` keeps it).

#### Two-stream concat VAE (`EmissionVaeTrainer`)

Two encoders: a trainable emission encoder (3→32-ch latent) plus a frozen pretrained PBR encoder (PBR→32-ch latent). The decoder receives `concat(z_emission, z_pbr)` = 64 channels along the feature axis, so its input projection is **reinitialized** (cannot reuse the PBR decoder's input weights). Output is 3-channel emission.

> ⚠️ **The shipped `_v2` run was NOT a finetune, and this section used to read as if it were**
> (audit 2026-08-14). Only `pretrained_pbr_encoder` — the *frozen conditioning* encoder, which
> sits off the emission path — was ever pretrained. Its config carried no `finetune_ckpt`, and a
> missing key is a **silent no-op** (`trainers/basic.py:403-409`), so the trainable emission
> encoder **and the whole decoder** trained from random init. Its width and depth were also each
> halved versus the albedo→emission VAE: **128.63 M trainable params vs 828.60 M (6.44×)**.
> Measured signature: first logged `train/l1` 0.2957 at step 9, where an albedo→emission
> finetune logs 0.00329 at step 0.
>
> Adding `finetune_ckpt` to that architecture would have been a measured no-op — only 1 of 284
> encoder tensors and 0 of 284 decoder tensors were shape-compatible. The fix therefore required
> an architecture change: `configs/scvae/emission_vae_pbr2emission_74k_v3.yaml` (deleted; tag `pre-refactor`), which matches
> the albedo→emission VAE apart from I/O channels (564 of 568 tensors identical in shape) and
> finetunes from the same pretrained PBR VAE.
>
> **Never resume a first launch.** `on_fit_start` returns before `_apply_finetune_checkpoint`
> when `resume_mode` is `legacy`/`lightning` (`basic.py:104-106`), so starting into a non-empty
> output dir silently skips the pretrained init — the same defect, by a second route. `--ckpt
> last` is safe only because `resolve_ckpt_path` returns `None` on an empty dir
> (`train.py:640-655`).

The emission DiT (downstream stage) similarly takes 96-channel input (32 emission + 32 PBR + 32 shape) and predicts 32-channel emission noise. Both DiTs **are** genuine finetunes of the same pretrained denoiser, with identical architecture args and step budgets — the asymmetry above is confined to the VAE stage.

Key trainer: `EmissionVaeTrainer` (`trellis2/trainers/vae/emission_vae.py`) — splits input into emission/PBR, encodes emission, freezes PBR encoder via `pretrained_pbr_encoder: ...`, concatenates latents for decoder.

Config: `configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml` (the release run). The paper run's config is the `config.yaml` of its run dir `outputs/emission_vae_pbr2emission_74k_v3_agentic/` (jupiter); the other variants are in tag `pre-refactor`.

#### Emission DiT conditioning (unified 2026-08-09)

One DiT config is kept, `configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml` (the release run;
the paper run's config is the `config.yaml` of `outputs/emission_dit_pbr2emission_74k_v3_agentic/`),
finetuned from `slat_flow_imgshape2tex_dit_1_3B_512_bf16` with DINOv3 thumbnail cross-attention + CFG
(`p_uncond: 0.1`). PBR/shape conditioning is per-token channel concat via `concat_cond`
(order `[x_t | shape | pbr]`, built in `LightGenSLatEmissionLatent`): `in_channels` 96 = 32 + shape 32 +
PBR 32; the two-stream decoder requires `z_pbr`. The `_v2` configs of the 2026-08-09 unification
(albedo→emission with and without PBR conditioning) are in tag `pre-refactor`.

Since the unification, `skip_pbr_cond` no longer separates the two chains — the chain
identity is the emission-latent representation (albedo-slot vs two-stream VAE); display
names live in the parent `AGENTS.md`. The SegviGen-emissive baseline received the matching
channel-concat mode in the same change (`--pbr_cond channel`, PR QhelDIV/SegviGen-emissive#1).

### Tiny-dataset overfit setup (10 samples + 4 GPUs)

All trainers use `drop_last=True` on the train DataLoader. With `len(dataset) = 10` and any `num_gpus * batch_size_per_gpu > 10`, every rank gets ≤ batch_size samples and produces zero training batches — Lightning prints `Total length of DataLoader across ranks is zero` and `Trainer.fit stopped: No training batches.`

Fix: oversample the train indices (repeat them in the split JSON's train list) to a length that is an exact multiple of `num_gpus * batch_size_per_gpu`, so `drop_last=True` drops nothing.

For validation on the same 10 samples, use **only** `val_split_name: val` in the trainer args (and **not** `val_split_file`). Background: `TrellisDataModule` injects `val_split_file` as the kwarg `split_file=...`, but `LightGenSLatEmission.__init__` accepts `split_json` / `split_indices` only — passing `val_split_file` will TypeError. With just `val_split_name: val`, the data module clones the train args (which already include `split_json`) and overrides `split` to `"val"`, which works.

### Load balancing across DDP ranks (BalancedResumableSampler)

Sparse-voxel datasets have ~100× per-sample memory variance (emission `.vxz` files range 5–561 KB in the 10-sample split). Under `ResumableSampler` + DDP, a permutation that concentrates heavy samples on one rank will OOM that rank while the others sit comfortably. This was masked pre-e6df30c because the sampler never actually rotated the shuffle; once it rotated, rank 0 OOMed mid-run.

Fix is `BalancedResumableSampler` (in `trellis2/utils/data_utils.py`), which splits each super-batch (`batch_size * world_size`) into `world_size` equal-size sub-groups via `load_balanced_group_indices(loads, world_size, equal_size=True)`. It requires the dataset to expose `self.loads: List[int]` indexed in the same order as `self.samples`. Enable with `trainer.args.use_balanced_sampler: true`.

**Pick the right load proxy — voxel count, not file size.** Sparse activation memory scales with `coords.shape[0]`, not bytes on disk. `.vxz` files are compressed with variable ratios: in the 10-sample split, bytes/voxel varies 8.5× (0.12–1.02), Pearson correlation between file size and voxel count is only 0.70, and voxel-count range (11.5×) is far narrower than file-size range (101.5×). File-size balancing therefore under-weights high-voxel-count + high-compression samples. For small datasets read `coords.shape[0]` once per sample at `__init__` (via `o_voxel.io.read_vxz`; ~10 ms per sample); for large datasets precompute into the parquet/metadata (sibling `sparse_voxel_pbr.py` uses `num_pbr_voxels`, `structured_latent.py` uses `{latent_key}_tokens`).

- `LightGenSLatEmission` — exposes `self.loads`. First version used `.vxz` file size (fast but only 0.70 correlation with real memory); replace with voxel count for accurate balancing.
- `LightGenSLatEmissionLatent` — does **not** provide `loads` yet. Add it before any balanced DiT run (e.g. `tokens_per_sample` from the latent metadata).
- For any new sparse dataset, add `self.loads` up front using voxel/token count; the cost at init is negligible and it future-proofs DDP scaling.

**Load balancing alone doesn't solve an overprovisioned batch size.** Even with ideal balancing, if all ranks run at the VRAM edge (observed: ~44 GB / 45 GB L40S after rotation), *any* run-to-run memory spike (Triton autotuner, checkpoint recomputation) can OOM a rank. Balancing + right-sized `batch_size_per_gpu` are complementary, not substitutes.

### Key Source Locations

- `trellis2/models/sc_vaes/sparse_unet_vae.py` — VAE encoder/decoder architecture
- `trellis2/models/structured_latent_flow.py` — `SLatFlowModel`, `ElasticSLatFlowModel` (DiT)
- `trellis2/trainers/` — PyTorch Lightning trainer classes (VAE, flow matching, with mixins for image/text conditioning)
- `trellis2/datasets/lightgen_slat.py` — `LightGenSLatEmission` (raw voxels), `LightGenSLatEmissionLatent` (pre-encoded)
- `trellis2/modules/` — Low-level ops: sparse convolution, attention, positional embeddings
- `trellis2/pipelines/` — End-to-end inference pipelines (image-to-3D, texturing)
- `o-voxel/` — C++/CUDA extension for voxelization, serialization (.vxz format), and mesh export

### Configuration System

YAML configs in `configs/` define the full training setup (models, datasets, trainer args, optimizer, scheduler). The trainer is selected via `trainer.name` which maps to a class in `TRAINER_MAP` inside `train.py`. Config values can be overridden at runtime with `--override key.path=value`.

#### Emission VAE config naming convention

All configs live under `configs/scvae/`. Pick trainer by prefix:

| Pattern | Trainer | Approach |
|---|---|---|
| `emission_vae_*.yaml` | `EmissionVaeTrainer` | two-stream concat VAE (trainable emission encoder + frozen PBR encoder) |

(The `emission_pbr_*` channel-hijack family is in tag `pre-refactor`.)

Dataset-size suffix is part of the file stem: `_<N>sample` for tiny overfit (10, 100, …), `_<N>k` / `_<N>m` for full training. Output dir, job name, and log filename all reuse the same stem.

**Validation and wandb are both mandatory.** Every training config must define:

- A val loop (`val_split_name: <name>` in trainer args, plus a non-zero `limit_val_batches`). Configs without validation are not accepted — for tiny / overfit datasets, validate against the same samples as train (the "did the model memorize?" signal).
- Wandb logging (`wandb_enable: true` + `wandb_project` + `wandb_name`). See "Run logging policy" above for the full block.

When adding a new config, copy the val block **and** the wandb block from the release config `configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml` as a template.

#### Checkpoint policy

Two modes, selected per-config in trainer args:

1. **Top-k by validation metric (preferred).** Set `save_top_k: <K>` (e.g. `3`). At the end of every validation epoch, the top K checkpoints by `monitor_metric` (default `val/loss`, `monitor_mode: min`) are kept under `outputs/<run>/ckpts/`, plus `last.ckpt` for resume. Filename format: `step{step:07d}-{val/loss:.4f}.ckpt`. Requires a working val loop. `i_save` is ignored in this mode.
2. **Step interval (legacy).** Set `i_save: <N>` and omit `save_top_k`. A checkpoint is written every N training steps with `save_top_k=-1` (keep all) and `last.ckpt` for resume.

Both modes are wired in `train.py` around the `ModelCheckpoint` construction (search for `save_top_k_cfg`). Convention: any new `configs/scvae/*_<N>sample*.yaml` overfit config uses top-k val mode by default; older / no-val configs (`limit_val_batches: 0`) cannot use top-k mode and must keep `i_save`.

### Sparse Tensor Convention

The codebase uses `SparseTensor` (from `trellis2/modules/sparse/`) throughout — a struct holding `feats` (feature tensor) and `coords` (integer voxel coordinates). Models operate on these rather than dense grids, enabling efficient processing of high-resolution (256-512) voxel data.
