# TRELLIS.2-Emission refactor — plan

Turn the `lightgen` branch into a clean TRELLIS.2-Emission: only the code behind the paper and release
runs of the pbr→emission chain (emission VAE → emission latents → DiT), our code in files of its
own beside upstream's (the LightGen part kept separate from upstream's PBR part), generic entry
points for LightgenBench users. Modeled on `TEXGen/.claude/worktrees/refactor/REFACTOR_TODO.md` (branch
`refactor`); the rules below are the same unless a line says otherwise.

## Progress

- [x] 1. Housekeeping
- [x] 2. Dead root files and one-off scripts
- [ ] 3. `archive/`, patches, stale reports and plans, stale ignore rules
- [ ] 4. Vendored Pillow (`blender_packages/`)
- [ ] 5. Launchers
- [ ] 5b. Move the kept cluster scripts into `script/internal/`
- [ ] 6. Configs
- [ ] 7. Check setup: inference gate, VAE table, encode check, training check, noise baselines (no code change)
- [ ] 8. Our dataset and trainer files beside upstream's (D13)
- [ ] 9. Coverage run (no code change)
- [ ] 10. The predictor `predict_emission_voxels_twostream.py`
- [ ] 11. The chain scripts in `data_toolkit/`
- [ ] 12. Our hooks out of upstream's `train.py`, `basic.py`, `callbacks.py` (D13)
- [ ] 13. The viewers in `vis/`
- [ ] 14. Environment files
- [ ] 15. Release layer: generic entry points for LightgenBench users
- [ ] 16. Docs
- [ ] 17. Merge (per D11) and the rename to `TRELLIS.2-Emission/` (D4) (needs Dongchen's go)
- [ ] 18. Cut-over to the new code (D8)

## How we work

- One step = one commit with one purpose. Removals and moves never share a commit with a behavior
  change. Cleanups change no behavior; fixes go under "Later", one step each, only if wanted.
- Dongchen gives the go for each step or range of steps (since 2026-10-07; no automatic run). A step
  whose checks pass is committed and reported; its **Result** is at most three lines (what changed,
  commits, verdict), run details go to commit messages. Dongchen can add **Instructions** to any step.
- Within a step, stop and ask only for: (1) an open decision the step needs; (2) a check that fails, unless the
  cause is an obvious slip in the step; (3) something the plan does not cover (a needed fix, a
  behavior change, a file outside the step's list); (4) an action outside the worktree and our
  compute: a push (every cluster checkout pulls `lightgen`, even inside running jobs; Keep stable
  15), a parent-repo commit, the merge, a write to jupiter's shared tree. Jobs on our machines for
  the checks need no go.
- After each step that touches code the inference gate must pass. Steps that touch the VAE, its
  dataset or the encoders also need the VAE table and the encode check; steps that touch training
  code also need the training check. A check passes when outputs are identical, or every difference
  is within the noise of two runs of tag `pre-refactor` on the same GPU type; `missing` must be 0;
  a NaN or an empty input never passes.
- Run checks in parallel across our machines (workstation 4090, sp4090, Solar L40S, Alliance H100;
  the compute-access skill): split the full gate by shape list, run a training check's before,
  after and noise runs at once, prepare the next step while checks run (commit it only after the
  step before passes). Skip a machine that needs a 2FA login. Submit Solar jobs without `-w`: a
  pinned job on our own nodes evicts another lab's running job.
- Compare only within one GPU type and environment: outputs differ between GPU types (Solar L40S
  vs the 4090: mean |d| up to 9e-4 on `pred`). Each GPU type gets its own baseline and noise run
  from `pre-refactor` (step 7). Never gate against the published August predictions: the same code
  today differs from them by up to 1.28e-3 on `pred`.
- Upstream is branch `main` (`98c2548`): the lab's Lightning port of microsoft/TRELLIS.2, squashed
  into `67f883b`; no microsoft commit is recorded and microsoft's repo is not a remote.
  `git diff main -- <file>` shows what we changed. Our code lives in files of its own beside
  upstream's, never as edits inside upstream's files (D13): inside an upstream file only the line
  that registers our class stays (`TRAINER_MAP`, the `datasets` and `trainers` registries). A file of
  ours that only keeps a published registry name alive is a one-line alias. Upstream code stays
  untouched.
- `o-voxel/` is not touched (D14).
- A doc that names a removed file is fixed in the same step.
- Each step updates TRELLIS2's own docs it affects (`CLAUDE.md`, `README.md`, `data_toolkit/README.md`)
  in the same commit. Parent docs (`AGENTS.md`, `.claude/rules/trellis2.md`, `docs/baselines/trellis2.md`,
  `docs/status/checkpoints.md`, `docs/evaluation.md`) describe `lightgen` and change only at the
  merge: a step that makes one stale adds an item to step 17's parent-docs list.
- All edits happen in the worktree `.claude/worktrees/trellis2-emission-refactor` (branch
  `trellis2-emission-refactor`, from tag `pre-refactor` = `fdaba01`); `git diff pre-refactor` shows
  every edit. The main checkout stays on `lightgen`.
- In the main checkout: never `git add -A` and never `git clean -x`/`-X`: its ignored `outputs/`
  (47 GB: the paper checkpoints, `restructure_gate/`) and `data/` (13 GB: the paper's emission
  latents, `ckpt_init/`) hold files the gate needs.
- Parent pointer bumps are path-limited (`git commit -- TRELLIS2`): the parent carries other
  sessions' uncommitted work. Nothing is pushed without Dongchen's go.

## Runs and the gate

| run | VAE (config → checkpoint) | DiT (config → checkpoint) |
|---|---|---|
| paper | `configs/scvae/emission_vae_pbr2emission_74k_v3_agentic.yaml` → `outputs/emission_vae_pbr2emission_74k_v3_agentic/ckpts/step0035200-0.0058.ckpt` (fir, 4 × H100, `b60c551`); local | `configs/gen/emission_dit_pbr2emission_74k_v3_agentic.yaml` → `outputs/emission_dit_pbr2emission_74k_v3_agentic/ckpts/last-step0050000.ckpt`, run with the local `config.workstation.yaml` (the run's `config.yaml` names a DINOv3 path that exists only on cs-venus-19; cs-venus-19, 4 × RTX PRO 6000 Blackwell, env `trellis2-bw`); local |
| release | `configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml` → jupiter `outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/step0035400-0.0036.ckpt` (rorqual, 4 × H100, `2e10c7f`) | `configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml` → jupiter `outputs/emission_dit_pbr2emission_lightgenbench_v1/snaps/last-step0050000.ckpt` (= `ckpts/last.ckpt`; rorqual, 4 × H100, `fdaba01`) |

Each DiT decodes only with the VAE that encoded its latents (the paper's latents
`emission_pbr2emission_74k_v3_agentic`, 72,421 shapes, are local under
`data/lightgen_74k_newbake/emission_latents/`). Name checkpoint files, never a run dir:
`vis_utils._pick_checkpoint` prefers a stale `last.ckpt`.

The inference gate: the paper's command, which `../evaluation/newdata_eval/sweeps/job_dit.sh test
last-step0050000.ckpt 0` builds, with its own `--out_dir` (`job_dit.sh` itself writes into the
published directory): `data_toolkit/predict_emission_voxels_twostream.py`, seed 0, `--guidance 3.0
--steps 50`, inputs `../evaluation/newdata_eval/out/data_root`, the `trellis2` env, the 4090 lock
`/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock`. The scripts are in `script/internal/gate/`
(step 1, from the September restructure's `outputs/restructure_gate/`); check outputs go to
`C=/local-scratch/localhome/dya78/lightgen_eval_local/trellis2_refactor_checks/`:

```bash
G=script/internal/gate
bash $G/tree.sh <commit> C/trees/<name>            # git archive + the main checkout's outputs/, data/
bash $G/infer.sh C/trees/<name> C/gate/<run> [$G/subset20.txt]    # prints GATE_DONE <n>
bash $G/compare.sh C/gate/pre-refactor_run1 C/gate/<run> [$G/subset20.txt]
bash $G/vae_table.sh C/trees/<name> C/vae/<run>; bash $G/vae_compare.sh <a.csv> <b.csv>
```

Step 7 adds the numeric comparison.

- Routine steps: `script/internal/gate/subset20.txt` (TEXGen's 20 gate shapes, all among the 204;
  about 3.4 s per shape on a free 4090 plus model load). Full: the 204 shapes of `../evaluation/newdata_eval/eval_sets/test_agentic_clean.txt`
  (about 12 min), for steps that touch the dataset, model loading, the predictor or the trainers,
  and before the merge.
- Compare every key of every `pred_voxels/<sha>.npz` (`coords`, `pred`, `gt_recon`, `gt_emission`).
  Pass: `same=N different=0 missing=0`, or every difference within noise. The September record:
  two runs of unchanged code on the workstation 4090 were 204/204 byte-identical.
- The VAE table: `data_toolkit/eval_vae_recon.py` on the 204, arms `pretrained_pad` and
  `ours_pbr2emission`; pass when `per_shape.csv` is identical with the `seconds` column dropped
  (`vae_compare.sh`). The committed `reports/vae_recon_test204/per_shape.csv` holds six arms (1,224
  rows); filtered to the two gate arms it matched in September.
- The release checkpoints get their own gate on a release test subset (step 7).
- Tests: `data_toolkit/test_{eval_vae_recon,emission_vae_resume,checkpoint_cadence}.py` and
  `vis/vis_utils_test.py` read 25 passed, 2 failed on `pre-refactor` (the two need a 10-sample VAE
  checkpoint this host lacks); every step keeps that.

## Keep stable (other code depends on these)

1. The directory `TRELLIS2/` and `o-voxel/o_voxel/` until step 17 (`../.gitmodules`;
   `../evaluation/newdata_eval/sweeps/star2/{dit_worker,segvigen_worker,build_trellis2_bw}.sbatch`).
   The `trellis2` env's o_voxel is an editable install through the symlink
   `/localhome/dya78/code/TRELLIS.2-lightning` → `TRELLIS2/`.
2. The ignored `data/` and `outputs/` paths the parent reads:
   `data/lightgen_74k_newbake/data_splits_emissive_74k_stratified_newbake_vae_agentic.json`
   (`../data_processing/splits/make_splits.py:39`, `../evaluation/user_study/build_valtest_manifest.py:25`
   and two `agentic_qc_exp` scripts), `data/lightgen_1k/thumbnails`
   (`../paper_figures/method_overview/config.py:208`), `outputs/emission_dit_74k/eval/pred_voxels`
   (`../paper_figures/pointsampled_eval/config.py:36`), `outputs/emission_dit_twostream_74k/eval/pred_voxels`
   (`../evaluation/pointsampled/viz/make_viz_data.py:15`), and the layout
   `outputs/<run>/ckpts/<file>.ckpt` with `config.yaml` (and `config.workstation.yaml`) one level up
   (`job_dit.sh:12,23,27-28`, the jupiter mirror).
3. `data_toolkit/predict_emission_voxels_twostream.py`: its path and flags (`--dit_ckpt --dit_cfg
   --vae_ckpt --vae_cfg --data_root --thumbnail_dir --shas --n all --guidance --steps --seed
   --pred_only --out_dir`, defaults `--pbr_latent_name`, `--shape_latent_name`, `--resolution 256`).
   Callers: `job_dit.sh:26-30`, `sweeps/star2/dit_worker.sbatch:368-373`,
   `trellis2_release/infer_local.sh:60-63`, `infer_solar.sbatch:119-123`. Its `SKIP` log lines
   (`infer_solar.sbatch:126` counts them). The input tree it reads: `<root>/emission_voxels_256/<id>.vxz`,
   `pbr_latents/tex_enc_next_dc_f16c32_fp16_256/<id>.npz`, `shape_latents/shape_enc_next_dc_f16c32_fp16_256/<id>.npz`,
   `<thumbnail_dir>/<id>.png`. `infer_local.sh:34` runs the live main checkout and refuses when
   `trellis2/` or this file has local changes, so the merge changes what it runs.
4. Output `<out>/pred_voxels/<sha>.npz`: `coords` int32 [N, 3], `pred` float32 [N, 3] in [0, 1],
   plus `gt_recon` and `gt_emission` unless `--pred_only`. Readers: `../evaluation/pointsampled/pipeline.py:290`,
   `score_dit_v2.sh:88-100`, `build_model_gallery.py`, `make_pred_dir.py:19`, `make_viz_data.py`.
5. `sample_seed` (`predict_emission_voxels_twostream.py:57`): identical to
   `../evaluation/newdata_eval/seedutil.py:24-26` (values pinned by `tests/test_seedutil.py`), and the
   global `torch.manual_seed` call it feeds.
6. Run-config keys read from outside: `dataset.args.{emission,shape,pbr}_normalization.{mean,std}`
   (32 values each; `infer_local.sh:37-41`, `infer_solar.sbatch:82-90`),
   `trainer.args.image_cond_model.args.{model_name,image_size}`, `trainer.args.sigma_min`,
   `models.denoiser.args.in_channels`, the `models.{denoiser,encoder,decoder}` sections. Every key
   of the published `config.yaml` files stays accepted: unknown trainer kwargs are silently
   `setattr`'d (`basic.py:83-84`), so a renamed key such as `finetune_ckpt` becomes a silent no-op;
   the September restructure's renamed dataset class rejected the stored `skip_pbr_cond`.
7. The import surface used from outside (`build_trellis2_bw.sbatch:199-214`):
   `trellis2.modules.sparse.SparseTensor`, `trellis2.trainers.builders.build_models`,
   `trellis2.pipelines.samplers.flow_euler.FlowEulerCfgSampler`,
   `trellis2.trainers.flow_matching.mixins.image_conditioned.DinoV3FeatureExtractor`,
   `o_voxel.serialize.{encode_seq,decode_seq}`.
8. The viewers `vis/vis_3d_vae.py`, `vis/vis_3d_dit.py`, `vis/vis_utils.py` (imported as
   `from vis.vis_utils`: `render_html`, `make_voxel_panel_data`, `make_glb_panel_data`,
   `load_vae_from_handle`, `resolve_experiment`, `emission_to_srgb_u8`), their flags and the
   `--experiment` forms `name=<vae ckpt>` and `name=<dit>:<vae>`. Cited by `../docs/hosting/aspis.md:20`,
   `../paper_figures/dataset_emitters/render_voxel.py:21`, `../.claude/rules/trellis2.md:95-98`.
9. Registry names in every stored `outputs/<run>/config.yaml` (local, jupiter, and
   `trellis2_bw/eval_inputs/solar_dit_stage/cfg/`): trainers `EmissionVaeTrainer`,
   `ImageConditionedSparseFlowMatchingCFGTrainer` (`train.py:42-57`); datasets `LightGenSLatEmission`,
   `LightGenSLatEmissionLatent`; models `SparseUnetVaeEncoder`, `SparseUnetVaeDecoder`,
   `ElasticSLatFlowModel`; `DinoV3FeatureExtractor`, `AdaptiveGradClipper`, `LinearMemoryController`,
   `LinearWarmupLRScheduler`.
10. Checkpoint contents: keys `models.<name>.*` and the duplicate `training_models.<name>.*`
    (`basic.py:77-78`), `ema` keyed by `models.*`, `sampler_state`, `elastic_controller`,
    `callbacks`, `loops`; filenames `step{step:07d}-{val/loss:.4f}.ckpt` and `last.ckpt`; the run
    dir's `wandb_run_id.txt`, `config.yaml`, `command.txt`. No trellis2 class is pickled. A renamed
    parameter or submodule is skipped without failing the load by `_load_weights_for_model`
    (`basic.py:371-401`: a warning per key, then `strict=False`; every `finetune_ckpt` and
    `data/ckpt_init/` load), `trellis2.models.from_pretrained` (`strict=False`) and
    `EmissionVaeTrainer.strict_loading=False` (a `--ckpt last` resume); the predictor and the
    viewers load strictly. Attribute and module names inside the VAE and DiT stay.
11. The data contract (`trellis2/datasets/lightgen_slat.py`): `<root>/{emission,pbr}_voxels_<res>/<sha>.vxz`
    (attrs `emissive`; `base_color, metallic, roughness, alpha`),
    `{emission,pbr,shape}_latents/<name>/<sha>.npz` (`feats`, `coords`) with `normalization_stats.json`,
    `<thumbnail_dir>/<sha>.png`, a parquet indexed by sha with `success`, a split JSON
    `{split: {"indices": [...]}}` positional over the rows left after the `success` filter and,
    when `min_emission > 0`, the `#stat_emission_color_max` filter (`lightgen_slat.py:148-152`) (auto-detected names
    `data_splits_emission_filtered.json`, `data_splits_filtered.json`). Producers:
    `../data_processing/data_preparation/voxels.py:250-279` and `run.py:147-161`,
    `data_toolkit/build_lightgenbench_index.py`, `data_toolkit/encode_emission_twostream_dataset.py`,
    `script/alliance/lightgenbench_unpack.sh`.
12. The chain script names the release launcher calls (`script/alliance/lightgenbench_v1.sh:141,171,176,227`):
    `data_toolkit/encode_emission_twostream_dataset.py`, `data_toolkit/build_lightgenbench_index.py`,
    `data_toolkit/bake_pbr2emission_v3_init.py`, `train.py`.
13. The `[ckpt-timing] SAVE_START step=N epoch=E t=T` and `RESUME_AFTER_SAVE step=N gap=Gs t=T`
    log lines and their timing (`CheckpointTimingCallback`, `callbacks.py:225,238`): the Alliance
    launcher's snapshot logic (`lightgenbench_v1.sh:192-218`) and the cs-venus-19 watcher
    (`dit_agentic_snapshot_last_venus19.sbatch:49-50,89`) parse the step and the timestamp.
14. `script/star2/backup_run.sh <src> <store> [<name>]`: `../AGENTS.md:153` calls it; byte-identical
    to the compute-access skill's copy.
15. Cluster checkouts and branch: star2 `/localscratch/dya78/TRELLIS.2-lightning`, fir
    `/scratch/dya78/TRELLIS.2-lightning`, Alliance `$HOME/scratch/TRELLIS.2-lightning`; the
    launchers set `BRANCH="lightgen"` and `git pull --ff-only` at submit and again inside the job
    (`lightgenbench_v1.sh:118,263`; on Alliance compute nodes the in-job pull may fail and is
    allowed to), so a push can reach a requeued job mid-run. The compute-access
    skill's `hedge.sh:31-32` hardcodes the checkout paths. `script/alliance/build_env.sh` needs
    `setup.sh` at the root (`:12`), globs `requirements-fir-*.txt` (`:14`) and builds o_voxel from
    `o-voxel/` (`:57`); Kevin's SegviGen release lane borrows that env.
16. Frozen code on jupiter: `trellis2_bw/eval_inputs/solar_dit_stage/TRELLIS2.tar` (2026-08-26) and
    `trellis2_bw/eval_inputs/trellis2_lightgenbench_v1/TRELLIS2-fdaba01.tar`; `../docs/evaluation.md:1131`
    cites `fdaba01`.
    The refactor never rewrites them; tag `pre-refactor` (= `fdaba01`) stays.
17. Labmates: Kevin's `segvigen-lightning` forked `lightgen` at `202bfb0` and imports
    `ElasticSLatFlowModel`, the private `lightgen_slat._build_parquet_samples`, `_read_npz_as_sparse`,
    `_sparse_collate`, `ClassifierFreeGuidanceMixin` and the DiT trainer; renames break his future
    rebases, not his runs. `SegviGen-emissive/` takes `o_voxel` from `TRELLIS2/o-voxel`.

## Decisions

Carried over from TEXGen (say if one does not apply here):

- D1 Removed files are deleted, not moved to `deprecated/`; git history and the tags (`pre-refactor`,
  `pre-trim-2026-09-16`) keep them. (The September restructure moved them into `deprecated/2026-09-16/`.)
- D2 Checks cover both the paper and the release checkpoints.
- D3 Cleanup steps change no behavior.
- D5 Steps that touch training code get the training check: a short deterministic run of the real
  `train.py` for the VAE and for the DiT, per-step loss compared before and after, plus one resume
  from a published checkpoint. `train.py` sets no global seed; the check seeds through a
  `sitecustomize.py` on `PYTHONPATH` (at process start, before the models are built, and again right
  before `trainer.fit`), not by editing `train.py`.
- D6 Every loss branch and config-switched option in kept code stays; only their configs go.
- D7 Work happens on branch `trellis2-emission-refactor`; the parent pointer is bumped once, at the merge.
- D9 Upstream code we do not use stays untouched, files and blocks alike (upstream's configs, among them
  `slat_flow_imgshape2tex_dit_1_3B_512_bf16.yaml`, the DiT's finetune source; upstream's
  `data_toolkit/` scripts; models, pipelines, samplers, the demo files `app*.py`, `example*.py`, `assets/` (D12).
- D10 (open, before publishing) Internal scripts in the published code: `script/internal/` holds our
  Slurm accounts and paths. Publish as is, publish a snapshot without it, or move it to the parent
  repo? Recommended: a published snapshot without it.

New for TRELLIS2 (Dongchen's answers of 2026-10-10):

- [x] D4 At the merge the directory becomes `TRELLIS.2-Emission/`, the paper's name (as TEXGen became
  `TEXGen-Emission/`). The rename moves `.gitmodules`, the gitlink, the o_voxel symlink target, the
  Solar sbatch paths (Keep stable 1) and every parent reference; the package stays `trellis2` and the
  registry names stay.
- [x] D8 After the merge the cluster checkouts and `hedge.sh` move to the new code (step 18). The
  jupiter tars stay frozen.
- [ ] D11 (open) Where the result lands: merge into `lightgen`, the branch of the submodule's repo
  `3dlg-hcvc/TRELLIS.2-lightning` that holds all our emission work (the parent gitlink points at its
  head, every launcher and checkout pulls it; its upstream is the same repo's `main`), or a new branch
  with `lightgen` frozen at `pre-refactor`? Recommended: merge into `lightgen` when no job runs from
  any checkout, as TEXGen merged into its `main`; tag `pre-refactor` and the frozen tars keep the old code.
- [x] D12 Upstream's demo material stays (`app.py`, `app_texturing.py`, `example.py`,
  `example_texturing.py`, `assets/`): it does no harm (D9).
- [x] D13 The LightGen part stays separate from upstream's PBR part: our code goes in files of its own
  beside upstream's, not as edits inside upstream's files. The dataset classes move into
  `trellis2/datasets/sparse_voxel_emission.py` (beside `sparse_voxel_pbr.py`) and
  `structured_latent_emission.py` (beside `structured_latent_svpbr.py`), `lightgen_slat.py` becomes
  aliases (step 8); our hooks in upstream's `train.py`, `basic.py`, `callbacks.py` move out into files
  of their own (step 12; the September restructure's `utils/train_ext.py` is a reference). Only the
  registration lines stay in upstream files. Not covered: D14.
- [x] D14 `o-voxel/` stays untouched, our two edits included (emissive buffers in
  `o_voxel/convert/volumetic_attr.py`, the `emissiveMipmaps` fix in `src/convert/volumetic_attr.cpp`):
  the data-preparation bake imports it and its output is GPU-architecture dependent. Our edits to
  upstream's `data_toolkit/{blender_script/dump_pbr.py,voxelize_pbr.py,encode_pbr_latent.py}` stay too
  (D3): the September restructure reverted them and the Blender recipes then wrote zero emission.
- [x] D16 The unmerged remote branches `origin/worktree-dreamy-riding-bird` (15 commits: `_v2` VAE
  configs, `recon_emission_voxels.py`, `tests/test_resolve_ckpt_path.py`, the `train.py --ckpt last`
  fixes `3945dc3`, `8b4269e`) and `origin/unified-pipeline-dump-refactor` (2 commits) stay unmerged;
  the `--ckpt last` fixes are listed under Later.
- [x] D17 Only the release launchers and configs stay, as in TEXGen; the paper runs' `config.yaml` on
  jupiter (and `config.workstation.yaml` locally) are their record.

## Done

1. (part) Tag `pre-refactor` = `fdaba01` (the `lightgen` head, equal to the parent gitlink and to
   `origin/lightgen`) and this worktree (branch `trellis2-emission-refactor`), 2026-10-06. The tag is
   local. `.claude/` is already ignored (`.gitignore:14`), so no ignore commit was needed.

## Steps

### 1. Housekeeping

- [x] Tag `pre-refactor` and the worktree (Done 1).
- [x] This plan on the branch (`REFACTOR_TODO.md`).
- [x] Move the gate harness from the main checkout's ignored `outputs/restructure_gate/` into
  `script/internal/gate/` (`tree.sh`, `infer.sh`, `compare.sh`, `vae_table.sh`, `vae_compare.sh`,
  `subset20.txt`): absolute-path arguments, the tree under test as an argument, the shapes file optional.
- [x] `docs/status/refactor.md` in the parent has no TRELLIS2 entry, not even for the September
  restructure: on step 17's list.

**Instructions:**

**Result:** Gate scripts in `script/internal/gate/` (Codex-reviewed; a skipped shape, a failed VAE row
or an unlisted directory now fails). Baseline `C/gate/pre-refactor_run1`: 204/204 npz, byte-identical
to September's `before/`, so the 4090 noise is zero. Tests: 25 passed, 2 failed (need a checkpoint).

### 2. Dead root files and one-off scripts

Not used by the paper or the release run. The 9 live files that hardcode the old path
`/…/code/TRELLIS.2-lightning` are all in this list.

- [x] Root: the 10 PNGs (`emission_dit_*_test.png` and others, about 4.4 MB; `.gitignore:26`
  matches four of them).
- [x] `data_toolkit/`, ours: `ablate_zpbr_v3.py`, `aggregate_{emission,latent}_stats.py`,
  `analyze_dark_regions.py`, `bake_albedo2emission_v3_init.py`, `bridge_voxel_to_uv.py` and its test,
  `check_74k_progress.py`, `compute_normalization_stats.py`, `diag_{dit_decode,emission_range}.py`,
  `dump_vae_recon_voxels.py`, `encode_10_samples.py`, `encode_emission_latent.py`,
  `encode_emission_latent_10samples.py`, `encode_emission_pbrfinetune_{dataset,10samples}.py`,
  `encode_single_sample_{latents,shape_latent}.py`, `extract_74k_tars.sh`, `patch_dit_config_stats.py`,
  `predict_emission_dit_74k.py`, `predict_emission_voxels.py` (the albedo→emission predictor; cited only
  by docs: `../docs/evaluation.md:344`, `../evaluation/newdata_eval/EXECUTION_LOG.md:74`, the
  `release_gt_voxels.py` docstring), `prep_{10sample_as1k,10sample_dataset,1k_dataset}.py`,
  `prep_74k_{bootstrap,dual_grid,dump_dualgrid_stream,dump}.py`, `regenerate_1k_dual_grid.py`,
  `stage_test_voxels.py`, `test_emission_dit{,_10samples,_10samples_pbrvae,_1k,_74k}.py`,
  `test_emission_vae_e2e.py`, `test_pbr_vae.py`, `test_single_emission_sample.py`,
  `vis_emission_vae.py`, `viz_uv_{albedo,alignment,gallery}.py`, `verify_albedo2emission_v3/`,
  `verify_pbr2emission_v3/`, `datasets/lightgen_74k.py`, `blender_script/dump_pbr_nonormalize.py`.
- [x] `vis/vis_emission_vae.py`.
- [x] Kept in `data_toolkit/` (ours): `bake_pbr2emission_v3_init.py`, `dual_grid_from_glb.py`,
  `encode_shape_from_dual_grid.py`, `encode_pbr_from_voxels.py`, `encode_emission_twostream_dataset.py`,
  `predict_emission_voxels_twostream.py`, `eval_vae_recon.py`, `build_lightgenbench_index.py`, and the
  tests `test_eval_vae_recon.py`, `test_emission_vae_resume.py`, `test_checkpoint_cadence.py`.

Parent docs that cite a deleted file go to step 17's list (`docs/baselines/trellis2.md:8`,
`docs/status/checkpoints.md:13-14` for the bridge). Gate: subset.

**Instructions:**

**Result:** 67 files removed (10 root PNGs, 56 `data_toolkit/` one-offs, `vis/vis_emission_vae.py`);
`CLAUDE.md` lines naming them fixed. Gate subset20 20/0/0 byte-identical; tests 25 passed, 2 failed
(unchanged).

### 3. `archive/`, patches, stale reports and plans, stale ignore rules

- [ ] `archive/` (30 files: configs, cvae tests, `lightgen_uv.py`, old viewers), `.patches/` (2),
  `docs/superpowers/` (9 April–June plans and specs), `reports/emission_vae_pbr2emission_74k_v3_status.md`,
  `reports/finetune_pbr_10sample_as1k_2gpu_v15_status.md`. Kept: `reports/vae_recon_test204/` and
  `reports/vae_pretrained_vs_finetuned.md` (`../docs/evaluation.md:445-447`).
- [ ] `.gitignore`: `.deprecated/` (`:11`, no such directory); `data/` (`:2`) is unanchored and also
  matches 12 tracked eigen files under `o-voxel/third_party/eigen/bench/btl/data/` (make it `/data/`
  and check nothing new shows as untracked); the PNG rule (`:26`) after step 2.
- Not here: the 370 MB of `debug_outputs/` blobs in history (out of scope).

**Instructions:**

**Result:**

### 4. Vendored Pillow (`blender_packages/`)

- [ ] `blender_packages/` (133 files, 15.2 MB: Pillow 12.1.1 vendored for Blender). Its users are the
  Blender 74k prep (deleted in steps 2 and 5) and upstream's `data_toolkit/dump_pbr.py:45`, which runs
  `blender_script/dump_pbr.py`; that script prepends the directory to `sys.path` (`:9`, our edit) and
  imports Pillow (`:12`). Neither run uses this path (their voxels come from
  `../data_processing/data_preparation/`), but check that the script still finds a Pillow inside
  Blender without the directory; if not, keep the directory or ask. The script stays unchanged (D14).

**Instructions:**

**Result:**

### 5. Launchers

64 files today (63 scripts and one requirements file): alliance 10, fir 25, star2 25, vulcan 4.

- [ ] Kept (release): `script/alliance/{lightgenbench_v1,lightgenbench_stage,lightgenbench_unpack}.sh`,
  the stage wrappers `script/alliance/{emission_vae_pbr2emission_lightgenbench_v1,encode_emission_pbr2emission_lightgenbench_v1,emission_dit_pbr2emission_lightgenbench_v1}.sh`,
  `script/alliance/{build_env.sh,requirements-fir-2026-09-07.txt,smoke_gpu.sh}`, and
  `script/star2/backup_run.sh` (Keep stable 14).
- [ ] Deleted: the rest, among them the paper's four (D17: `script/fir/emission_vae_pbr2emission_74k_v3_agentic.sh`,
  `script/fir/encode_emission_pbr2emission_74k_v3_agentic.sh`,
  `script/star2/emission_dit_pbr2emission_74k_v3_agentic_venus19.sh`,
  `script/star2/dit_agentic_snapshot_last_venus19.sbatch`), `script/alliance/emission_dit_74k.sh`, all of `script/vulcan/`,
  the standby launchers (`*_v3_agentic_venus05.sh`, `script/fir/emission_dit_pbr2emission_74k_v3_agentic.sh`),
  the albedo→emission, two-stream, `_v2`, `_v3`, `nonzero`, 1k, 10-sample and Blender-prep launchers.

**Instructions:**

**Result:**

### 5b. Move the kept cluster scripts into `script/internal/`

`script/internal/{alliance,star2}/`. Nothing pulls this branch, so the move is safe here; at the merge
it changes the paths that cluster checkouts, `../AGENTS.md:153` (`backup_run.sh`), `hedge.sh` and
`hedged-submission.md:34-35` use (step 17's lists). Check the relative paths inside the scripts
(`build_env.sh:12` finds `setup.sh` at the root).

**Instructions:**

**Result:**

### 6. Configs

- [ ] Kept: upstream's 9 (D9) and the release pair `configs/{scvae/emission_vae,gen/emission_dit}_pbr2emission_lightgenbench_v1.yaml`.
- [ ] Deleted: every other `emission_*` config (the paper pair `*_pbr2emission_74k_v3_agentic.yaml`, D17; `emission_pbr_vae_*`, `emission_vae_74k_{fir,vulcan}`,
  `emission_vae_pbr2emission_74k_v3{,_nonzero_nocopy}`, `emission_dit_{74k,1k_probe,twostream_74k}`,
  `emission_dit_{albedo2emission,pbr2emission}_74k_*`, the 10-sample pairs). The training check
  brings its own config (step 7). `vis/vis_utils_test.py` already skips the tests that need a
  10-sample checkpoint.

**Instructions:**

**Result:**

### 7. Check setup: inference gate, VAE table, encode check, training check, noise baselines (no code change)

All runs on tag `pre-refactor`, in parallel across machines. Tooling goes to `script/internal/gate/`;
data and run outputs outside the repo.

- [ ] A numeric comparison of two `pred_voxels/` trees: per shape and key, byte identity, mean and
  max absolute difference; the evaluator's IoU, MAE and PSNR from `../evaluation/pointsampled/`
  reading the voxels (`--voxel_pred_root`); NaN-safe (NaN on one side fails); verdict EMPTY on an
  empty shapes file. TEXGen's `scripts/internal/gate/compare_numeric.py` is the pattern.
- [x] The 20-shape subset of the 204, fixed in a file (`subset20.txt`, step 1).
- [ ] Gate noise: two runs on unchanged code per GPU type (the workstation 4090 read zero in
  September; re-measure on `pre-refactor`): sp4090 (runtime
  `/local-scratch/localhome/dya78/t2lgb_sp4090`, env pack and compiled o_voxel), Solar L40S (the
  `trellis2_release/infer_solar.sbatch` pattern with a `git archive` of the tree under test; it
  requires compute capability 8.9).
- [ ] The VAE table and its noise run.
- [ ] The encode check: `encode_emission_twostream_dataset.py` on the gate shapes with the paper VAE;
  the latents and `normalization_stats.json` compared before and after (the paper encode recorded
  bitwise-equal tensors across its two ranks).
- [ ] The release checkpoints: pull the release VAE `step0035400-0.0036.ckpt` (13.3 GB) and DiT
  `snaps/last-step0050000.ckpt` (20.7 GB) with their `config.yaml` into the main checkout's
  `outputs/<run>/` (rsync, sha256 on both ends against `SHA256SUMS.jupiter`; check `df` first). A
  20-shape release test subset with inputs from jupiter `datasets/dataset_release{,_latents}` laid
  out as `infer_local.sh:43-50` does; its baseline and noise runs. Call the predictor directly:
  `infer_local.sh` runs the main checkout and refuses a dirty tree.
- [ ] The training check, for the VAE (`EmissionVaeTrainer`) and the DiT
  (`ImageConditionedSparseFlowMatchingCFGTrainer`), through the real `train.py`:
  - a tiny training root: the release layout built with `build_lightgenbench_index.py` over a few
    release shapes (emission voxels for the VAE; emission, shape and PBR latents and thumbnails for
    the DiT), fixed split;
  - a check config extending each release config: local paths, batch 1, a few dozen steps, one
    validation, one save;
  - a `sitecustomize.py` that seeds at process start, before `train.py` builds the models (`:630`;
    the DiT's re-initialized 96-channel input projection takes a random init), and again right
    before `trainer.fit` (`:666`; the noise, `sample_t` and the CFG mask draw from unseeded torch
    and numpy RNGs: `flow_matching.py:130-163`, `sparse_flow_matching.py:95`,
    `mixins/classifier_free_guidance.py:40`), and records per step the
    batch ids, lr, loss, gradient norm and weight sum, and for the saved file its keys, EMA,
    `sampler_state` and `elastic_controller`;
  - fresh and resume; resume from the release VAE `ckpts/last.ckpt` (step 35,800) and DiT
    `ckpts/last.ckpt` (step 50,000), so the check shows the code under test still loads published
    files, and the `SAVE_START` and `RESUME_AFTER_SAVE` lines still print;
  - memory: the 1.3B DiT may not train on a 24 GB 4090 even at batch 1 (measure; else Solar L40S);
    a resume loads 13–21 GB with optimizer state into host RAM (the workstation has 62 GB).
  Not covered: gradient accumulation and multi-GPU DDP.
- [ ] The existing tests on `pre-refactor`: `data_toolkit/test_{eval_vae_recon,emission_vae_resume,checkpoint_cadence}.py`,
  `vis/vis_utils_test.py`; record pass and skip counts.

**Instructions:**

**Result:**

### 8. Our dataset and trainer files beside upstream's (D13)

| ours | upstream beside it |
|---|---|
| `trellis2/datasets/lightgen_slat.py` `LightGenSLatEmission` (VAE data: a plain torch `Dataset` with its own parquet, split and `.vxz` reader and collate) | `sparse_voxel_pbr.py` `SparseVoxelPbrDataset` |
| `lightgen_slat.py` `LightGenSLatEmissionLatent` (DiT data; its own thumbnail loading at `:549`) | `structured_latent_svpbr.py` `ImageConditionedSLatPbr`, `components.py` `ImageConditionedMixin` |
| `trellis2/trainers/vae/emission_vae.py` `EmissionVaeTrainer` | `trainers/vae/pbr_vae.py` `PbrVaeTrainer` (already beside it) |

- [ ] 8a. The dataset file split into the two new files per D13; `lightgen_slat.py` keeps the
  registry names as aliases. Full gate, VAE table, encode check, training check.
- [ ] 8a also: the module docstring of `lightgen_slat.py:14` names `encode_emission_latent.py`
  (deleted in step 2).
- [ ] 8b. Our trainers used by neither run: `emission_pbr_finetune.py` (`EmissionPbrFinetuneTrainer`,
  albedo→emission), `emission_pbr_conditional.py` and `utils/model_surgery.py` (its configs are archived:
  `archive/configs/scvae/emission_pbr_cvae_*.yaml`), with their `TRAINER_MAP` and registry entries and the `vis_utils.load_vae_from_handle`
  branches (`vis_utils.py:368-430`); old albedo→emission runs then no longer open in the viewers.
  Gate subset.

Branch `restructure-2026-09-16` did the dataset split (`01ed03f`) as a reference; take neither its
config, script and launcher renames nor its behavior changes. Reusing upstream's base classes or
`ImageConditionedMixin` in place of our reader is a behavior change unless the gate reads identical:
then it goes under Later.

**Instructions:**

**Result:**

### 9. Coverage run (no code change)

The VAE and DiT training checks (fresh and resume, validation, a save) under a line-coverage tracer,
plus the predictor on a few gate and release shapes, the encoder and the VAE table. Output: per file,
the lines that never run. "Never ran" is not "dead" for the multi-GPU and gradient-accumulation
paths, the elastic memory controller's branches, or `--pred_only`.

**Instructions:**

**Result:**

### 10. The predictor `predict_emission_voxels_twostream.py`

Keep its path, flags, output, `sample_seed` and the strict prefix-filtered load (Keep stable 3–5).
No rename: the September restructure renamed it to `predict_emission_voxels.py`, the name of the
different script step 2 deletes, which the parent's docs still cite.

- [ ] Hardcoded defaults and paths; the `--split_json` default; output-only code. Exact lines after
  step 9.
- [ ] Comments naming `predict_emission_voxels.py` (deleted in step 2): `:7`, `:10`, `:107`, `:147`.

**Instructions:**

**Result:**

### 11. The chain scripts in `data_toolkit/`

`bake_pbr2emission_v3_init.py`, `dual_grid_from_glb.py`, `encode_shape_from_dual_grid.py`,
`encode_pbr_from_voxels.py`, `encode_emission_twostream_dataset.py`, `eval_vae_recon.py`,
`build_lightgenbench_index.py`: hardcoded paths, dead branches, output-only code; the docstring and the error message naming
`encode_emission_pbrfinetune_dataset.py` (deleted in step 2) in `encode_emission_twostream_dataset.py:4,86`.
Names stay (Keep stable 12). Checks: the encode check, the VAE table; the bake's output
`data/ckpt_init/pbr2emission_v3/{encoder,decoder}.safetensors` compared byte for byte.

**Instructions:**

**Result:**

### 12. Our hooks out of upstream's `train.py`, `basic.py`, `callbacks.py` (D13)

Ours in upstream files (`git diff main`): `train.py` +91/−10 (the `TRAINER_MAP` entries, the wandb
run-id restore through `wandb_run_id.txt`, `build_checkpoint_callback` with top-k by `val/loss` and
`last.ckpt` every `ckpt_last_every_n_steps`, `resolve_ckpt_path` for `--ckpt last`, the flex_gemm
patch at `:61`), `trainers/basic.py` +92/−1 (`_resolve_hf_path` and `.safetensors`, the LR log, the
validation summary, the per-step `[train]` trace, the `visualize_sample` guard), `trainers/callbacks.py`
+77 (`CheckpointTimingCallback`, Keep stable 13), and the small ones: `utils/data_utils.py`
(`ResumableSampler.set_epoch`, an upstream bug), `trainers/data.py` (empty validation list),
`flow_matching/mixins/image_conditioned.py` (DINOv3 layer path), the registries in
`datasets/__init__.py` and `trainers/__init__.py`. Per D13 each moves into a file of our own (a
subclass or a callback that upstream's code calls through its registry), so `git diff main` on these
files shows only the registration lines; an edit that cannot move without changing behavior is
listed for Dongchen. Training check for each move. If a hook moves into a callback, `train/lr`
must still be read before the scheduler step (`basic.py:534` does it inside `training_step`; the
September restructure's callback read it late until `b960f52`).

**Instructions:**

**Result:**

### 13. The viewers in `vis/`

`vis_3d_vae.py` (190 lines), `vis_3d_dit.py` (321), `vis_utils.py` (753), `vis_utils_test.py` (181):
hardcoded paths, output-only code; Keep stable 8. Check: both viewers on two shapes before and after,
the HTML identical apart from timestamps.

**Instructions:**

**Result:**

### 14. Environment files

- [ ] `setup.sh` (upstream's, our one change: the `libjpeg-dev` line commented out),
  `script/alliance/build_env.sh` and `requirements-fir-2026-09-07.txt` (the release env): one
  description that matches what runs.
- [ ] The envs that run: workstation `trellis2` (Python 3.10, torch 2.6 + CUDA 12.4; o_voxel editable
  through the symlink); Solar `~/envs/trellis2` (o_voxel rebuilt 2026-09-09; the bake, the release
  packing, `infer_solar.sbatch`); `trellis2-bw` on cs-venus-19 (Blackwell; the paper DiT); the
  Alliance `env/` venv (the release runs, torch 2.9); the sp4090 runtime pack.
- [ ] The pins the release run used (`flash_attn`, `flex_gemm`, `cumesh`, transformers for DINOv3, ...)
  as the invariant.
- [ ] If a dependency goes, update the import probes: `script/alliance/{build_env,smoke_gpu}.sh`,
  `../evaluation/newdata_eval/sweeps/star2/build_trellis2_bw.sbatch:199-214`.

**Instructions:**

**Result:**

### 15. Release layer: generic entry points for LightgenBench users

No user or cluster paths in this layer. Needs steps 10 and 11; its README commands are written in step 16.

- [ ] `script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>`: the generic half of
  `script/alliance/lightgenbench_{stage,unpack}.sh` (untar the Hugging Face voxels and thumbnails into
  the layout of Keep stable 11), then `build_lightgenbench_index.py`.
- [ ] Shape and PBR latents: the release run took them from jupiter
  `datasets/dataset_release_latents/tars/latents.tar` (built in the SegviGen session); the Hugging Face
  release does not carry latents. A user needs the commands that make them
  (`dual_grid_from_glb.py` → `encode_shape_from_dual_grid.py`, `encode_pbr_from_voxels.py`); check
  on a few shapes that they reproduce the jupiter latents.
- [ ] The chain in order with generic paths: bake the VAE init (`bake_pbr2emission_v3_init.py`) →
  train the VAE → encode the emission latents with the best-`val/loss` VAE → put
  `normalization_stats.json` into the DiT config (the release launcher refuses a mismatch; a generic
  check does the same) → train the DiT → predict. The recipe is the release configs plus the launcher's
  overrides (`lightgenbench_v1.sh:43,45,229`: VAE 4 × 16 over the config's 32 per GPU, 35,800 steps;
  DiT 4 × 8, 50,000 steps, from `slat_flow_imgshape2tex_dit_1_3B_512_bf16`): the generic commands
  carry the overrides, and the README states the GPU memory they need.
- [ ] The internal Alliance launchers stay as they are.

**Instructions:**

**Result:**

### 16. Docs

One doc of record, keeping upstream's attribution and citation (`README.md`).

- [ ] Inside: `README.md` (upstream's), `CLAUDE.md` (341 lines), `data_toolkit/README.md`; the
  checkpoint pairing (each DiT with its VAE); the findings that still apply. The restructure branch's
  `TRELLIS2_EMISSION.md` is a source for content, not taken whole.
- [ ] The release commands (step 15) and the inference command on the release test split.

**Instructions:**

**Result:**

### 17. Merge (per D11) and the rename to `TRELLIS.2-Emission/` (D4) (needs Dongchen's go)

- [ ] The full gate, the VAE table, the encode check and the training check on the branch head,
  against the `pre-refactor` baselines; the tests.
- [ ] No job running from any `lightgen` checkout (Keep stable 15).
- [ ] The merge and the push (with the go).
- [ ] In the parent, one commit: the pointer bump and the rename `TRELLIS2/` → `TRELLIS.2-Emission/` (`.gitmodules`, the
  gitlink, the o_voxel symlink target, the Solar sbatch paths of Keep stable 1, every parent
  reference: search again).
- [ ] Parent docs: `AGENTS.md:31,45-47,54,65,152-163` (`:45-47`: the ~20 files with the old path are
  inside TRELLIS2, not in the parent, and step 2 deletes the live ones; the symlink stays for o_voxel),
  `.claude/rules/trellis2.md` (paths, scripts, trainer names; already stale: `:130` names `_v3` as
  the production config, `:140-148` lists only the `_v2` DiTs), `docs/baselines/trellis2.md:3-8` (`:8`
  already stale: scoring reads the voxels directly), `docs/status/checkpoints.md:13-17`,
  `docs/evaluation.md:344`, `docs/status/refactor.md` (the TRELLIS2 entry); files step 2 deleted are
  still named by `.claude/rules/trellis2.md` (`prep_10sample_as1k.py`, `prep_74k_*.py`,
  `extract_74k_tars.sh`, `encode_emission_latent.py`, `test_emission_*`, `vis_emission_vae.py`,
  `viz_uv_*`, `predict_emission_voxels.py`), `docs/status/checkpoints.md:13-14` and
  `docs/baselines/trellis2.md:8` (`bridge_voxel_to_uv.py`), `evaluation/newdata_eval/EXECUTION_LOG.md:74`
  and the `release_gt_voxels.py` docstring (`predict_emission_voxels.py --gt_only`); the spec
  `docs/superpowers/specs/2026-09-16-trellis2-restructure-design.md:115` says the parent never
  imports `trellis2` (it does: `build_trellis2_bw.sbatch`).
- [ ] Outside the repo: the compute-access skill's `hedge.sh:31-32` and `hedged-submission.md:34-35`
  if step 5b moved the launchers; `SegviGen-emissive/emissive/infer/dump_pred_voxels_eval5.py:3` names
  `bridge_voxel_to_uv.py` (another submodule, a docstring).

**Instructions:**

**Result:**

### 18. Cut-over to the new code (D8)

- [ ] Move the cluster checkouts to the merged code (star2, fir, Alliance: nibi, rorqual, killarney,
  tamia, trillium) and rebuild the Alliance `env/` if step 14 changed the requirements; repoint
  `hedge.sh`.
- [ ] The jupiter tars stay frozen; a new tar only for a new run.
- [ ] Re-run `infer_local.sh`'s release smoke on the merged main checkout.
- [ ] Tell Kevin (his fork imports private `lightgen_slat` helpers) — Dongchen's call.

**Instructions:**

**Result:**

## Later, only if wanted (behavior changes, one step each)

- [ ] **`_pick_checkpoint` prefers `last.ckpt`** (`vis/vis_utils.py:272`): require a named file.
- [ ] **Silent loads**: `strict=False` in `_load_weights_for_model` (`basic.py:371-401`) and
  `models.from_pretrained`; `EmissionVaeTrainer.strict_loading=False`; unknown trainer kwargs
  `setattr`'d silently (`basic.py:83-84`); a missing `finetune_ckpt` is a no-op.
- [ ] **A global seed in `train.py`** (none today).
- [ ] **`--ckpt last` fixes** from `origin/worktree-dreamy-riding-bird` (`3945dc3`, `8b4269e`, with
  `tests/test_resolve_ckpt_path.py`), per D16.
- [ ] **Release batch.** The release VAE config says 32 per GPU; the run used 16 from the launcher
  (`lightgenbench_v1.sh:43`).
- [ ] **Fixes and tests from `restructure-2026-09-16`**: the 10 pytest files under `tests/`, the
  encoders' fail-fast on an empty data root (`c077f4d`, `60f06fd`), the coordinate-mismatch raises in
  the dataset readers (`01ed03f`, tests in `f9b5f5e`).
- [ ] **Our dataset reader → upstream's base classes** (if step 8 shows a difference).

## Out of scope

- `o-voxel/` (D14), including the 1,930 tracked eigen files that `.gitmodules` also declares as a submodule.
- The position frame (voxels in [−0.5, 0.5], atlas in [−1, 1]): a data-format change, tracked in
  `../docs/status/refactor.md`.
- The 512 stage and a render loss: the chain stays 256-only by decision (2026-08-23).
- A history rewrite to drop the 370 MB of `debug_outputs/` blobs.
