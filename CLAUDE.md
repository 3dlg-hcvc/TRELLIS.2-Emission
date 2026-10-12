# CLAUDE.md

TRELLIS.2-Emission: TRELLIS.2 (the 3DLG-HCVC Lightning port, branch `main`) adapted to generate emission
on sparse voxels. **`README.md` is the doc of record**: what differs from TRELLIS.2, setup, data, the
release chain (`script/lightgenbench/run_chain.sh`), inference on the test split. This file holds what an
agent working here needs beyond it. The refactor's plan and its "Keep stable" list: `REFACTOR_TODO.md`.
The parent lightgen repo holds the evaluation, the display names (`AGENTS.md` § Naming) and the
checkpoint store rules (`.claude/rules/trellis2.md`).

## Environments

| env | where | Python, torch | what ran in it |
|---|---|---|---|
| conda `trellis2` | workstation (cs-3dlg-23, RTX 4090) | 3.10.19, torch 2.6.0 + CUDA 12.4, transformers 5.1.0, flash_attn 2.7.3, lightning 2.6.1 | the paper's inference and VAE table, the refactor checks; o_voxel editable through `/localhome/dya78/code/TRELLIS.2-lightning` → this repo |
| `~/envs/trellis2` (Solar shared home) | cs-venus L40S | the same versions; flex_gemm compiled for sm_89 (L40S) only; o_voxel editable from `~/code/o-voxel-src` | the DiT training check, `infer_solar.sbatch`, the data-preparation bake |
| `trellis2-bw` (jupiter `trellis2_bw/eval_inputs/solar_dit_stage/trellis2bw_env.tar.gz`) | cs-venus-19 / -05, RTX PRO 6000 Blackwell | built by the parent's `evaluation/newdata_eval/sweeps/star2/build_trellis2_bw.sbatch` | the paper DiT's training |
| venv `env/` (`script/internal/alliance/build_env.sh`) | Alliance H100 (fir, rorqual, nibi, killarney, tamia) | 3.11, torch 2.9.1, transformers 5.3.0, flash_attn 2.8.3, lightning 2.6.1; every pin in `script/internal/alliance/requirements-fir-2026-09-07.txt` | the release runs (VAE, encode, DiT) and the paper VAE (fir) |
| runtime pack `t2lgb_sp4090/` | sp4090 (cs-3dlg-24, RTX 4090) | its own env and compiled o_voxel | release inference seeds, the paper gate |

What every env needs: lightning 2.6 (the checkpoint callback's deferral and the callbacks rely on it);
transformers 5.x (DINOv3 keeps its layers at `model.layer` up to 5.1 and `model.model.layer` from 5.5;
`flow_matching/mixins/image_conditioned.py` takes either); flex_gemm and o_voxel compiled for the GPU's architecture;
and, in the Hugging Face cache, `microsoft/TRELLIS.2-4B` (the VAE's frozen PBR encoder, the DiT's finetune
source, the shape/PBR latent encoders) and `facebook/dinov3-vitl16-pretrain-lvd1689m`. `setup.sh` is
upstream's (our one change: the `sudo apt install libjpeg-dev` line is commented out).

## Runs of record

Each DiT is paired with the VAE whose latents it was trained on (its decoder needs that VAE and the
config's `emission_normalization` holds those latents' statistics).

| run | VAE | DiT |
|---|---|---|
| paper | `emission_vae_pbr2emission_74k_v3_agentic` `ckpts/step0035200-0.0058.ckpt` | `emission_dit_pbr2emission_74k_v3_agentic` `ckpts/last-step0050000.ckpt` with `config.workstation.yaml` |
| release (LightgenBench) | `emission_vae_pbr2emission_lightgenbench_v1` `ckpts/step0035400-0.0036.ckpt` | `emission_dit_pbr2emission_lightgenbench_v1` `snaps/last-step0050000.ckpt` |

Jupiter (`/cs/3dlg-jupiter-project/lightgen/outputs/<run>/`) is the source, `outputs/<run>/` here the
cache; name the file, never let anything pick one (`vis_utils._pick_checkpoint` prefers `last.ckpt`).
The paper configs live only in each run's `config.yaml` and in tag `pre-refactor` (decision D17).

## Commands

```bash
python train.py --config <cfg> --output_dir outputs/<run> [--ckpt last] [--num_gpus 4] [--override k=v ...] [--tryrun]
bash script/internal/alliance/emission_vae_pbr2emission_lightgenbench_v1.sh   # the release run on Alliance; usage in lightgenbench_v1.sh
python -m pytest data_toolkit/test_eval_vae_recon.py data_toolkit/test_emission_vae_resume.py \
  data_toolkit/test_checkpoint_cadence.py vis/vis_utils_test.py   # in a gate tree (it links data/ and outputs/)
python vis/vis_3d_vae.py --experiment "pbr→emission=<vae ckpt>" ...   # viewers: --help; HTML into visualizations/
```

`vis/vis_utils_test.py` reads the 10-sample fixture `data/lightgen_test/`; two of its tests fail when
`outputs/emission_vae_10sample_overfit/ckpts/` is empty, as in the main checkout (25 passed, 2 failed is
the baseline), and its decode smoke test needs CUDA.

Tee every training run's stdout to `log/<TS>-<name>.log` (`run_chain.sh` does) and log to wandb project
`LightGen` under a chain-named run (`--override trainer.args.wandb_name=...`); with `wandb_enable` false or
missing, `train.py:build_logger` falls back to a CSV logger silently. Lightning flushes scalars every
`i_log` steps, so a short epoch can end with none: `i_log: 1` for tiny smoke runs. Our two configs have a
validation loop and the wandb block (upstream's have neither); a new emission config copies both. Cluster work goes through the compute-access skill; the internal launchers
are under `script/internal/` (`alliance/` for the release runs, `star2/backup_run.sh`, `gate/`).

## Checking a code change (refactor gate)

Tooling in `script/internal/gate/`; data and outputs outside the repo, in
`C = /local-scratch/localhome/dya78/lightgen_eval_local/trellis2_refactor_checks/`. `baselines.json`
lists, per GPU type and check, the baseline and noise runs of tag `pre-refactor`; compare a run only with
runs of the same GPU type (the workstation and sp4090 4090s differ from each other at float level).

```bash
G=script/internal/gate
bash $G/tree.sh <commit> C/trees/<name>                         # the commit + the main checkout's outputs/, data/
bash $G/infer.sh C/trees/<name> C/gate/<run> [$G/subset20.txt]   # paper gate; GATE_RUN=release: release ckpts on C/release_root
bash $G/compare.sh <baseline run> <run> [<shas>]                 # byte comparison; compare_numeric.py for float differences
bash $G/vae_table.sh C/trees/<name> C/vae/<run>; bash $G/vae_compare.sh <a.csv> <b.csv>
bash $G/encode_check.sh C/trees/<name> C/encode/<run>; python $G/compare_encode.py --a <run> --b <run>
bash $G/train-check.sh C/trees/<name> <out> vae|dit fresh|resume   # real train.py, 16 steps, every step recorded
python3 $G/compare_traincheck.py --a <record> --b <record> [--envelope <record> ...]
```

On the workstation every check repeats exactly. The DiT training check does not: its sparse attention calls
flash-attention without the deterministic option, so the backward pass varies between runs (the first
step's loss repeats exactly; later steps drift by ~1e-3). It runs on Solar L40S (`star2/checks.sbatch`; a
1.3B DiT does not train in 24 GB) and passes within the envelope of four `pre-refactor` runs, the first
step's loss exactly. sp4090 runs the paper gate through `C/sp4090_tooling/sp4090_gate.sh`.

## Source map

| where | what |
|---|---|
| `trellis2/trainers/vae/emission_vae.py` | `EmissionVaeTrainer`: trainable emission encoder (3 → 32 ch) + frozen PBR encoder (`pretrained_pbr_encoder`); decoder on `concat(z_emission, z_pbr)`, 64 ch in, 3 ch out |
| `data_toolkit/init_emission_vae_from_pbr_vae.py` | the VAE's start point from the pretrained PBR VAE (`data/ckpt_init/pbr2emission_v3/`; `--verify-config`) |
| `trellis2/datasets/sparse_voxel_emission.py` | `LightGenSLatEmission` (VAE data), beside upstream's `sparse_voxel_pbr.py` |
| `trellis2/datasets/structured_latent_emission.py` | `LightGenSLatEmissionLatent` (DiT data: `[x_t \| shape \| pbr]` per token, 96 ch, thumbnails), beside `structured_latent_svpbr.py` |
| `trellis2/datasets/emission_helpers.py` | their shared index, split and voxel readers |
| `trellis2/trainers/callbacks.py` | upstream's callbacks, then ours: wandb logger, checkpoint callback, `[ckpt-timing]` and `[train]`/`[val]` lines |
| `trellis2/trainers/flow_matching/` | upstream's DiT trainer (`ImageConditionedSparseFlowMatchingCFGTrainer`), DINOv3 conditioning |
| `trellis2/models/` | `sc_vaes/sparse_unet_vae.py` (VAEs), `structured_latent_flow.py` (`ElasticSLatFlowModel`) |
| `data_toolkit/` ours | `encode_{shape_from_dual_grid,pbr_from_voxels,emission_latents}.py`, `dual_grid_from_glb.py`, `build_lightgenbench_index.py`, `predict_emission.py`, `eval_vae_recon.py` (the VAE table) |
| `vis/` | `vis_3d_vae.py`, `vis_3d_dit.py`, shared `vis_utils.py` |

Upstream's edits that stay in its files are listed in `REFACTOR_TODO.md` step 12 and in the README.

## Traps

1. **Never start a first launch in a non-empty run directory.** `on_fit_start` skips
   `_apply_finetune_checkpoint` when it resumes (`trellis2/trainers/basic.py`), so the pretrained init is
   lost silently; `--ckpt last` is safe only because `resolve_ckpt_path` returns `None` on an empty dir.
2. **A missing `finetune_ckpt` key is a silent no-op.** The 2026-08 `_v2` VAE trained from random init
   this way (with a sixth of the parameters); the `_v3` architecture matches the PBR VAE apart from its I/O
   channels so the init loads.
3. **Before finetuning, copy the architecture args from the pretrained checkpoint's JSON**; guessed
   `num_blocks` and the like caused repeated silent failures.
4. **`drop_last=True` with tiny data:** a train split smaller than `num_gpus × batch_size_per_gpu` gives no
   batches; repeat the indices to an exact multiple. Validate on a tiny set with `val_split_name: val`
   only (`val_split_file` reaches the dataset as `split_file=` and raises a TypeError).
5. **The DiT's snapshot at the end of training** asks for `val_num_samples` (16) samples and crashes in
   `torch.cat` when the training set is smaller (the VAE's caps its batch); set it no larger than that.
6. **Load balancing** (`trainer.args.use_balanced_sampler: true`) needs `self.loads` in sample order:
   `LightGenSLatEmissionLatent` gives token counts, `LightGenSLatEmission` still gives file sizes (0.70
   correlation with voxel count, which is what memory follows). Balancing does not rescue a batch size
   at the memory edge.
7. **Lightning postpones a checkpoint save** whose monitor is missing or that falls on an epoch's last
   batch, and a top-k file name carries the `val/loss` of the last validation before the save: the paper
   VAE's `step0035200-0.0058.ckpt` holds step-35,200 weights, its 0.0058 is of step 35,092. Each
   checkpoint passes through `$TMPDIR` (13 GB VAE, 21 GB DiT), where Unix socket paths must stay short
   (`run_chain.sh` checks both).
8. **The DiT's training does not repeat exactly** (flash-attention's backward is not deterministic); the
   VAE's does on one GPU type. FlexGEMM caches its autotune per GPU name in `~/.flex_gemm/`; on a shared
   NFS home ranks race on it (`FLEX_GEMM_USE_AUTOTUNE_CACHE=0`, as the Alliance launcher sets).
9. **The latent encoders run in fp32** (`.float()`; Triton's `tl.dot` needs equal dtypes). Regenerated
   latents match the release's at float level, not byte level (`README.md`, Data).
10. **Viewers:** the GLB panel and the voxel panels do not bbox-align (Blender's importer and the browser's
    GLTFLoader decode the GLB into different vertex sets); the fix, a Blender-exported sidecar GLB at
    voxelization, is not done.

## Conventions

- Viewers write one HTML with a sample menu, never one file per sample; label panels with the parent's
  display names (`pbr→emission`, `albedo→emission`), never internal shorthand.
- Shared modules get descriptive names (`vis_utils.py`); a leading `_` is for single-caller helpers.
- Evaluation lives in the parent repo (`evaluation/pointsampled/`, doc `docs/evaluation.md`): the DiT is
  sampled with `data_toolkit/predict_emission.py` and scored on mesh-surface points there.
