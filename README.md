# TRELLIS.2 for emission

This repository adapts **TRELLIS.2** from PBR materials to **emission**. Given a 3D shape as sparse
voxels with its PBR materials, and an image of the object, it generates the RGB emission of every
surface voxel: a VAE encodes emission into a latent beside the shape's PBR latent, and a flow-matching
transformer generates that latent. It is the TRELLIS.2-Emission baseline of LightgenBench
([Hugging Face `3dlg-hcvc/LightgenBench`](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)).

TRELLIS.2 and most of this code are the work of its authors:

> **Native and Compact Structured Latents for 3D Generation** (tech report, 2025). Jianfeng Xiang,
> Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao,
> Nicholas Jing Yuan, Jiaolong Yang. Microsoft.
> [Paper](https://arxiv.org/abs/2512.14692) · [Project page](https://microsoft.github.io/TRELLIS.2) ·
> [Code](https://github.com/microsoft/TRELLIS.2) · [Model](https://huggingface.co/microsoft/TRELLIS.2-4B)

The training code is 3DLG-HCVC's port of TRELLIS.2 to PyTorch Lightning (branch `main` of this
repository). If you use this code, please cite TRELLIS.2 ([below](#citation)).

## What differs from TRELLIS.2

- **Emission VAE** (`trellis2/trainers/vae/emission_vae.py`, `EmissionVaeTrainer`). A trainable encoder
  takes the voxels' emission (3 channels) to a 32-channel latent; TRELLIS.2's PBR encoder, frozen, takes
  base color, metallic, roughness and alpha to another 32 channels. The decoder reads both latents and
  outputs emission. Its weights start from TRELLIS.2's PBR VAE: `data_toolkit/init_emission_vae_from_pbr_vae.py`
  builds that start point.
- **Emission DiT.** TRELLIS.2's image-conditioned flow model and trainer, finetuned from its texture
  model `slat_flow_imgshape2tex_dit_1_3B_512_bf16`. Each token carries the noisy emission latent, the
  shape latent and the PBR latent (96 channels, so the input layer is trained anew) and attends to DINOv3
  features of the object's image; the image is dropped 10% of the time for classifier-free guidance.
- **Data.** 256³ voxels: LightgenBench's emission and PBR voxels, plus the shape and PBR latents of
  TRELLIS.2's own encoders. Loaders: `trellis2/datasets/sparse_voxel_emission.py` (VAE) and
  `structured_latent_emission.py` (DiT).
- **Training callbacks** (the end of `trellis2/trainers/callbacks.py`): one wandb run across resumes, the best
  checkpoint by `val/loss` plus `last.ckpt`, and a training log line every `i_log` steps.

Our code sits in files of its own, except those four training callbacks, which follow upstream's in
`trellis2/trainers/callbacks.py`. The other edits left in upstream's files are small: `train.py` and the
`__init__.py` registries of `trellis2/{datasets,trainers}` register ours; `trellis2/trainers/basic.py` logs the learning rate, loads `.safetensors` and Hugging Face paths
and skips previews for datasets without them; `ResumableSampler.set_epoch`
(`trellis2/utils/data_utils.py`, missing upstream, so every epoch shuffled alike); an empty validation
set for Lightning 2 (`trellis2/trainers/data.py`); DINOv3 in transformers 5.5
(`flow_matching/mixins/image_conditioned.py`); and emission in O-Voxel and in
`data_toolkit/{voxelize_pbr.py,blender_script/dump_pbr.py}`, including an O-Voxel fix: it sampled the
emissive texture with the roughness texture's mipmaps; `data_toolkit/encode_pbr_latent.py` runs its encoder
in fp32; `setup.sh` no longer calls `sudo apt`. All of them:

```bash
git diff --diff-filter=M main -- train.py trellis2/ o-voxel/ data_toolkit/ setup.sh
```

## Setup

```bash
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
pip install lightning==2.6.1 pyarrow wandb
```

`setup.sh` is TRELLIS.2's (Linux, CUDA Toolkit 12.4, conda env `trellis2`; `. ./setup.sh --help` lists
the parts). Our runs used torch 2.6.0 + CUDA 12.4, transformers 5.1.0 and flash-attn 2.7.3 (RTX 4090:
data, inference) and torch 2.9.1 + CUDA 12.6, transformers 5.3.0 and flash-attn 2.8.3 (H100: training),
both with lightning 2.6.1. FlexGEMM and O-Voxel are compiled for the GPU they are built on.

On first use the code downloads `microsoft/TRELLIS.2-4B` (the PBR, shape and texture models) and
`facebook/dinov3-vitl16-pretrain-lvd1689m` from Hugging Face. DINOv3 is gated: request access on its
page first, and log in (`huggingface-cli login`). Run every command below from the repository root.

## Data

LightgenBench is gated too: accept its terms on the dataset page.

```bash
bash script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>
```

This downloads `splits.json`, `checksums.sha256` and the voxel and thumbnail tars (about 14 GB), checks
`splits.json` against the released split and every tar against `checksums.sha256`, unpacks them into
`<data_root>/{emission_voxels_256,pbr_voxels_256,thumbnails}/` (15 GB) and writes the two index files the
loaders read: `df_lightgenbench.parquet` (the shapes) and `data_splits_lightgenbench.json` (row numbers of
the 36,426 train / 200 val / 200 test shapes).

The DiT also reads each shape's shape and PBR latents, which the dataset does not carry:

```bash
TEXVERSE_1K=<dir> bash script/lightgenbench/run_chain.sh latents <data_root>
```

`<dir>` holds the shapes' source files from
[TexVerse-1K](https://huggingface.co/datasets/YiboZhang2001/TexVerse-1K), as
`glbs/glbs_1k/<group>/<uuid>_1024.glb`. The stage builds each shape's dual grid
(`data_toolkit/dual_grid_from_glb.py --loader lightgenbench`: the mesh loader LightgenBench's data preparation
uses, from the lightgen repository that holds this one, or `LIGHTGEN_REPO=<dir>`), checks that its voxels are those of the shape's
`pbr_voxels_256`, and encodes both latents with TRELLIS.2's encoders. Checked against our release on 12
shapes: the dual grids are byte-identical, and the latents differ at float level (a mean difference of
about 1e-4 of their spread, on an RTX 4090 and an L40S alike); predictions from them moved voxel IoU by
under 5e-4 on the two test shapes among them.
A dual grid took about a second per shape on our workstation; `WORKERS` (default 8) processes run side by side.

## Training

```bash
bash script/lightgenbench/run_chain.sh vae <data_root>
VAE_CKPT=outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/<file>.ckpt \
    bash script/lightgenbench/run_chain.sh encode <data_root>
bash script/lightgenbench/run_chain.sh dit <data_root>
```

This is the recipe of our release run, on 4 × H100 80 GB: the VAE (`configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml`)
with 16 shapes per GPU (global batch 64; the config's own 32 is not what ran), 35,800 steps; the
DiT (`configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml`) with 8 per GPU (global 32), 50,000
steps. `run_chain.sh` refuses another global batch unless `ANY_BATCH=1`; its header lists the other
settings (`NUM_GPUS`, `BS`, `MAX_STEPS`, `OUTPUTS`, ...). Each stage tees its output to
`log/<time>-lightgenbench_<stage>.log`; the two training stages also log to wandb when you are logged in
(project `LightGen`).

- **Encode** with the VAE checkpoint of lowest `val/loss` (ours: `step0035400-0.0036.ckpt`); nothing picks
  one for you. It writes the emission latents, their `normalization_stats.json` and `vae_ckpt.txt`; it keeps
  latents already there and refuses a folder whose `vae_ckpt.txt` names another checkpoint file (by path
  and size, not by content: do not overwrite a checkpoint in place).
- **The DiT config holds those statistics** (`dataset.args.emission_normalization`); ours are of our
  VAE's latents. After a VAE of your own, copy the mean and std of your `normalization_stats.json` there:
  the `dit` stage stops if they disagree, since wrong statistics mis-scale every input without an error.
- **A first launch starts in an empty run directory.** Both models start from pretrained weights, and a
  run that resumes a checkpoint skips that load; `run_chain.sh` refuses a run directory with checkpoints
  unless `RESUME=1`, which continues an interrupted run.
- **Checkpoints pass through `$TMPDIR`** on their way to disk (13 GB for the VAE, 21 GB for the DiT), and
  Python puts Unix sockets there, whose paths must stay short: with a small `/tmp`, set
  `CHAIN_TMPDIR=<a short path on a disk with room>`. The stage checks before it trains.
- The DiT's training does not repeat exactly: its sparse attention runs flash-attention's backward
  without the deterministic option.

## Inference on the test split

```bash
VAE_CKPT=outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/<file>.ckpt \
DIT_CKPT=outputs/emission_dit_pbr2emission_lightgenbench_v1/ckpts/last.ckpt \
    bash script/lightgenbench/run_chain.sh predict <data_root>
```

A DiT works only with the VAE whose latents it was trained on: give the VAE checkpoint the encode used.
We report the DiT's last checkpoint (step 50,000). This writes `outputs/predict_test_seed0/pred_voxels/<uuid>.npz`
for the 200 test shapes, with `coords` (the voxels, at 256³), `pred` (the generated emission, RGB in
[0, 1]), `gt_emission` (the dataset's emission) and `gt_recon` (the VAE's reconstruction of it), using 50
sampling steps and guidance scale 3.0. Each shape's noise comes from its uuid and `SEED` (default 0), not
from its place in the list; on our RTX 4090s a rerun gave the same bytes. A GPU with 24 GB is enough.
The benchmark's own scoring, on points sampled on the mesh surface, is not part of this repository.

## Repository

| path | contents |
|---|---|
| `configs/` | the release VAE and DiT configs, and TRELLIS.2's own |
| `trellis2/` | models, datasets, trainers (TRELLIS.2's package; ours in files of their own, our training callbacks at the end of `trainers/callbacks.py`) |
| `train.py` | training entry point (`--config`, `--output_dir`, `--override key=value`) |
| `script/lightgenbench/` | `prepare_lightgenbench.sh`, `run_chain.sh`: everything above |
| `data_toolkit/` | TRELLIS.2's data scripts, and ours: the VAE start point, the encoders, the predictor, the index, `eval_vae_recon.py` |
| `vis/` | interactive 3D viewers of VAE reconstructions and DiT samples (one HTML page, a menu of shapes) |
| `o-voxel/` | TRELLIS.2's voxel library (with emission) |
| `script/internal/` | our cluster launchers and regression checks; not needed to use the code |
| `example.py`, `example_texturing.py`, `app.py`, `app_texturing.py` | TRELLIS.2's image-to-3D and texturing demos, unchanged ([their usage](https://github.com/microsoft/TRELLIS.2#-usage)) |

## License

TRELLIS.2 is released under the [MIT License](LICENSE). Some dependencies have their own terms: [nvdiffrast](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt) and
[nvdiffrec](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt). LightgenBench shapes keep the
licenses of their source models (see the dataset card).

## Citation

```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
