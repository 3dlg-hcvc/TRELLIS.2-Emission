#!/usr/bin/env python
"""Bake the pbr->emission _v3 VAE initial weights.

WHY THIS EXISTS
---------------
The v3 config finetunes from the pretrained PBR VAE (tex_enc/tex_dec_next_dc_
f16c32_fp16).  Four tensors differ in shape from the pretrained pair, and
BasicTrainer._load_weights_for_model (trellis2/trainers/basic.py:385-394) drops
any shape-mismatched tensor entirely, leaving it at its xavier init:

    encoder.input_layer.weight    (64, 6)    -> (64, 3)
    decoder.from_latent.weight    (1024, 32) -> (1024, 64)
    decoder.output_layer.weight   (6, 64)    -> (3, 64)
    decoder.output_layer.bias     (6,)       -> (3,)

A random tensor at either end of the information path destroys the pretrained
function, so v3 cold-starts despite inheriting 99.992% of its weights.

Every one of those four has an exact channel correspondence, so we resolve them
OFFLINE and write out checkpoints whose shapes already match the v3
architecture.  The trainer then performs a full 284/284 load with zero shape
mismatches -- no trainer code is touched, and no other config can be affected.

OPT-IN SURFACE
--------------
Only the two `finetune_ckpt` paths of the emission VAE config
(configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml; the paper run's
emission_vae_pbr2emission_74k_v3_agentic did the same) point at this output.
Every other config keeps its hub short-id, so the DiT config, which
deliberately reinitializes its 64->96 input projection, is untouched by
construction -- there is no flag to set wrong.

THE MAPPING
-----------
encoder.input_layer  : SparseLinear is nn.Linear (modules/sparse/linear.py:10),
    so weight is (out_features, in_features) and axis 1 is the input channel.
    The pretrained 6-channel order is [base_color(0:3), metallic, roughness,
    alpha] (configs/scvae/tex_vae_next_dc_f16c32_fp16.yaml attrs), and the
    albedo chain packs emission into the base_color slot
    (EmissionPbrFinetuneTrainer._pack_emission_as_pbr, deleted in the refactor;
    tag pre-refactor).  -> take columns 0:3.
    The dropped columns 3:6 saw the CONSTANTS (-1, +1, +1), so their
    contribution is a fixed offset, not data.  Discarding it silently changes
    the layer's operating point (||offset|| = 2.01 vs ||bias|| = 1.14), so it is
    folded into the bias.  NOTE this overwrites input_layer.bias, which is
    (64,) on both sides and therefore NOT one of the four mismatched tensors.

decoder.from_latent  : axis 1 is the latent channel.  The trainer builds
    z_combined = cat([z_emission, z_pbr]) with EMISSION FIRST
    (emission_vae.py:142), so the pretrained 32 columns belong at 0:32.
    Columns 32:64 are the new PBR-conditioning stream and are ZEROED, making
    the decoder at step 0 exactly the pretrained function of the emission
    latent while leaving the PBR path fully trainable (its gradient is
    dL/dh (x) z_pbr, which is nonzero from step 1).
    from_latent.bias transfers verbatim and stays calibrated because the
    zeroed block contributes nothing at init.

decoder.output_layer : axis 0 is the output channel; emission is the
    base_color slot -> take rows 0:3 of weight and bias.

Usage:
    conda run -n trellis2 python data_toolkit/init_emission_vae_from_pbr_vae.py \
        --out data/ckpt_init/pbr2emission_v3 \
        --verify-config configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from safetensors.torch import load_file, save_file

# The albedo->emission chain's constant pad, in [-1, 1] space
# (EmissionPbrFinetuneTrainer, deleted in the refactor; tag pre-refactor).
PAD = torch.tensor([-1.0, 1.0, 1.0])  # metallic, roughness, alpha
EMI = 3   # emission channels == width of the base_color slot
LAT = 32  # pretrained latent width; v3's decoder takes 2 * LAT


def bake_encoder(sd: dict) -> dict:
    # Arithmetic in fp32, stored back in the source dtype. The pretrained files
    # are fp16 throughout; writing fp32 for the edited tensors alone would make
    # the checkpoint mixed-dtype, and the bias fold is a sum of products that
    # deserves fp32 accumulation even though its magnitude (~2.0) is nowhere
    # near fp16's limits.
    out = dict(sd)
    dt = sd["input_layer.weight"].dtype
    W = sd["input_layer.weight"].float()  # (64, 6) = (out, in)
    b = sd["input_layer.bias"].float()    # (64,)
    out["input_layer.weight"] = W[:, :EMI].contiguous().to(dt)
    out["input_layer.bias"] = (b + W[:, EMI:] @ PAD).contiguous().to(sd["input_layer.bias"].dtype)
    return out


def bake_decoder(sd: dict) -> dict:
    out = dict(sd)
    dt = sd["from_latent.weight"].dtype
    Wf = sd["from_latent.weight"].float()  # (1024, 32) = (out, in)
    new = torch.zeros(Wf.shape[0], 2 * LAT, dtype=torch.float32)
    new[:, :LAT] = Wf  # emission latent first; cols 32:64 (z_pbr) stay zero
    out["from_latent.weight"] = new.contiguous().to(dt)
    ow, ob = sd["output_layer.weight"], sd["output_layer.bias"]
    out["output_layer.weight"] = ow.float()[:EMI].contiguous().to(ow.dtype)
    out["output_layer.bias"] = ob.float()[:EMI].contiguous().to(ob.dtype)
    return out


def resolve_hub(short_id: str) -> str:
    """Resolve a hub short-id to a local .safetensors, honouring HF_HOME.

    A cluster node may keep its HF cache outside the home directory, so a
    hardcoded ~/.cache/huggingface is not portable.
    HF_HOME is checked first, then the default; `~` is only a fallback.
    """
    parts = short_id.split("/")
    repo, sub = f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    roots = []
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    tried = []
    for root in roots:
        pat = os.path.join(root, f"models--{repo.replace('/', '--')}", "snapshots", "*", f"{sub}.safetensors")
        tried.append(pat)
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"No local snapshot for {short_id}; looked in:\n  " + "\n  ".join(tried))


def verify(cfg_path: str, baked: dict) -> int:
    """Diff the baked state dicts against the v3 architecture built on meta."""
    import yaml
    from trellis2.trainers.builders import build_models

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with torch.device("meta"):
        models = build_models(cfg["models"])

    bad = 0
    for name, sd in baked.items():
        tgt = models[name].state_dict()
        mism = [(k, tuple(v.shape), tuple(tgt[k].shape))
                for k, v in sd.items() if k in tgt and tuple(v.shape) != tuple(tgt[k].shape)]
        extra = [k for k in sd if k not in tgt]
        missing = [k for k in tgt if k not in sd]
        print(f"  {name}: src={len(sd)} tgt={len(tgt)} "
              f"SHAPE-MISMATCH={len(mism)} src_not_in_tgt={len(extra)} tgt_missing={len(missing)}")
        for m in mism:
            print(f"     MISMATCH {m}")
        bad += len(mism) + len(extra) + len(missing)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    ap.add_argument("--dec", default="microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-config", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    enc_src, dec_src = resolve_hub(args.enc), resolve_hub(args.dec)
    print(f"[bake] encoder src: {enc_src}\n[bake] decoder src: {dec_src}")

    baked = {
        "encoder": bake_encoder(load_file(enc_src)),
        "decoder": bake_decoder(load_file(dec_src)),
    }
    for name, sd in baked.items():
        dst = os.path.join(args.out, f"{name}.safetensors")
        save_file(sd, dst)
        print(f"[bake] wrote {dst}  ({len(sd)} tensors)")

    if args.verify_config:
        print("[verify] against", args.verify_config)
        bad = verify(args.verify_config, baked)
        print("[verify] PASS -- exact load, no mismatches" if bad == 0
              else f"[verify] FAIL -- {bad} problems")
        raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
