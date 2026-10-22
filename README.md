# TRELLIS.2-Emission

A LightGenBench baseline that generates the emission of a 3D shape from its geometry, PBR materials and an image, built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).


## Environment Setup
Create the `trellis2` conda environment on Linux with the CUDA Toolkit 12.4.
Run every command in this README from the root of the repository:
```bash
# Create the environment and build TRELLIS.2's extensions
# (FlexGEMM and O-Voxel are compiled for the GPU they are built on, so build on the GPU type you run on)
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Install the packages for training and data preparation
pip install lightning==2.6.1 pyarrow wandb
```


## Data Setup
The dataset download is described at https://github.com/3dlg-hcvc/lightgenbench.
Prepare the voxels, dual grids and thumbnails, then compute the shape and PBR latents from them:
```bash
# Download the voxels, dual grids and thumbnails into <download_dir>, check them, and unpack them into <data_root>
# (SKIP_DOWNLOAD=1 uses the files already in <download_dir>)
bash script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>

# Compute the shape latents from the dual grids and the PBR latents from the voxels
bash script/lightgenbench/run_chain.sh latents <data_root>
```
A shape without a dual grid in `<data_root>/dual_grid_256/` is built from its [TexVerse-1K](https://huggingface.co/datasets/YiboZhang2001/TexVerse-1K) GLB when `TEXVERSE_1K=<dir>` is set.


## Training
Train the emission VAE, encode the emission latents with it, then train the DiT on those latents:
```bash
# Train the emission VAE in outputs/emission_vae_pbr2emission_lightgenbench_v1/
bash script/lightgenbench/run_chain.sh vae <data_root>

# Encode the emission latents with the VAE checkpoint of lowest val/loss
VAE_CKPT=outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/<checkpoint>.ckpt \
    bash script/lightgenbench/run_chain.sh encode <data_root>

# Copy the mean and std of <data_root>/emission_latents/emission_pbr2emission_lightgenbench_v1/normalization_stats.json
# into dataset.args.emission_normalization of configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml,
# then train the DiT in outputs/emission_dit_pbr2emission_lightgenbench_v1/
bash script/lightgenbench/run_chain.sh dit <data_root>
```
The stages use 4 GPUs; with another number, set `NUM_GPUS=<n> ANY_BATCH=1`.
A run starts in an empty folder, and `RESUME=1` continues it.
The header of `script/lightgenbench/run_chain.sh` lists the other settings.


## Prediction
Predict the emission of the test split, giving the VAE checkpoint whose latents the DiT was trained on and a DiT checkpoint (choose it on the validation split):
```bash
# Write predictions to outputs/predict_test_seed0/pred_voxels/ (SEED=<n> writes outputs/predict_test_seed<n>/)
VAE_CKPT=outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/<checkpoint>.ckpt \
DIT_CKPT=outputs/emission_dit_pbr2emission_lightgenbench_v1/ckpts/<checkpoint>.ckpt \
    bash script/lightgenbench/run_chain.sh predict <data_root>
```
Score the predictions with the LightGenBench evaluation: https://github.com/3dlg-hcvc/lightgenbench

### With our checkpoint
Our VAE and DiT checkpoints, coming soon in the Hugging Face repo `3dlg-hcvc/LightgenBench-baselines`, replace the Training section; the Data Setup is still needed:
```bash
# Download our VAE and DiT checkpoints, each with its config, into checkpoints/TRELLIS.2-Emission/
hf download 3dlg-hcvc/LightgenBench-baselines --include "TRELLIS.2-Emission/*" --local-dir checkpoints

# Write predictions to outputs/predict_test_seed0/pred_voxels/
VAE_CKPT=checkpoints/TRELLIS.2-Emission/vae/ckpts/vae.ckpt \
DIT_CKPT=checkpoints/TRELLIS.2-Emission/dit/ckpts/dit.ckpt \
    bash script/lightgenbench/run_chain.sh predict <data_root>
```


## License
TRELLIS.2 is released under the [MIT License](LICENSE).
nvdiffrast and nvdiffrec, which `setup.sh` installs, have their own licenses.


## Citation
If you use this code, please cite TRELLIS.2:
```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
