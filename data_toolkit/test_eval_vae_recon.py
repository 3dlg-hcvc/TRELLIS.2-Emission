"""Unit checks for the pure parts of eval_vae_recon.py (metrics + packing). No GPU, no data.

Run:  conda run --no-capture-output -n trellis2 python -m pytest data_toolkit/test_eval_vae_recon.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_vae_recon as m  # noqa: E402


def _shape(n=100, lit_every=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    t01 = torch.zeros(n, 3)
    lit = torch.arange(n) % lit_every == 0
    t01[lit] = torch.rand(int(lit.sum()), 3, generator=g) * 0.8 + 0.2  # clearly lit, > 1/255
    return t01 * 2 - 1, lit


def test_perfect_reconstruction():
    t, lit = _shape()
    r = m.compute_metrics(t.clone(), t)
    assert r["l1_all"] == 0.0 and r["l1_lit"] == 0.0
    assert r["psnr_all"] == m.PSNR_CAP and r["psnr_lit"] == m.PSNR_CAP
    assert r["iou"] == 1.0
    assert r["n_lit"] == int(lit.sum()) and abs(r["lit_frac"] - lit.float().mean().item()) < 1e-6


def test_all_black_prediction():
    t, lit = _shape()
    pred = torch.full_like(t, -1.0)  # 0 in [0,1] space
    r = m.compute_metrics(pred, t)
    t01 = (t + 1) / 2
    assert abs(r["mae01_all"] - t01.abs().mean().item()) < 1e-6
    assert abs(r["l1_all"] - 2 * r["mae01_all"]) < 1e-6  # [-1,1] is 2x the [0,1] scale
    assert r["iou"] == 0.0  # predicted-lit set empty, GT-lit non-empty
    assert r["pred_mean01_lit"] == 0.0
    assert r["pred_max01"] == 0.0


def test_lit_subset_isolates_the_lit_voxels():
    t, lit = _shape()
    pred = t.clone()
    pred[~lit] += 0.5  # corrupt only the dark voxels (in [-1,1]; clamped later for psnr)
    r = m.compute_metrics(pred, t)
    assert r["l1_lit"] == 0.0 and r["psnr_lit"] == m.PSNR_CAP
    assert r["l1_all"] > 0.0 and r["psnr_all"] < m.PSNR_CAP


def test_no_lit_voxels_matches_evaluator_empty_mask_convention():
    t = torch.full((50, 3), -1.0)
    r = m.compute_metrics(t.clone(), t)
    assert r["n_lit"] == 0 and math.isnan(r["l1_lit"]) and math.isnan(r["mae01_lit"])
    assert r["psnr_lit"] == m.PSNR_CAP  # evaluation/pointsampled/pipeline.py: empty GT mask -> 40 dB cap
    assert r["iou"] == 1.0  # empty union convention


def test_summarize_uses_common_shapes_and_flags_gaps(tmp_path):
    import csv
    p = tmp_path / "per_shape.csv"
    rows = []
    def row(arm, sha, status="ok", l1=0.1):
        d = {c: "" for c in m.CSV_COLS}
        d.update(arm=arm, sha256=sha, status=status, seconds="0")
        if status == "ok":
            d.update({k: str(l1) for k in m.METRIC_COLS})
        rows.append(d)
    row("pretrained_pad", "a", l1=0.2); row("pretrained_pad", "b", l1=0.2); row("pretrained_pad", "c", l1=0.9)
    row("ours_pbr2emission", "a", l1=0.1); row("ours_pbr2emission", "b", l1=0.1); row("ours_pbr2emission", "c", status="oom")
    row("ours_pbr2emission", "zzz_not_in_eval_set", l1=5.0)  # must be ignored
    row("pretrained_albedo_ctrl", "a", l1=0.3); row("pretrained_albedo_ctrl", "b", l1=0.3); row("pretrained_albedo_ctrl", "c", l1=0.3)
    for d in rows:
        d["target"] = m.ARM_TARGET.get(d["arm"], "emission")
    rows[1]["l1_lit"] = "nan"  # shape b has no GT-lit voxel for pretrained_pad -> lit n drops to 1
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m.CSV_COLS); w.writeheader(); w.writerows(rows)
    table, summary, warnings = m.summarize(str(p), ["a", "b", "c"], paired_ref="pretrained_pad")
    assert summary["n_common"] == 2 and summary["n_expected"] == 3
    # the marginal mean for pretrained_pad must exclude shape c (not scored by every arm)
    assert abs(summary["arms"]["pretrained_pad"]["l1_all"]["mean"] - 0.2) < 1e-9
    assert summary["arms"]["ours_pbr2emission"]["n_bad"] == 1
    assert any("2 shapes ALL arms scored" in w for w in warnings) and any("failed" in w for w in warnings)
    assert summary["arms"]["pretrained_pad"]["l1_lit"]["n"] == 1 and "(n=1)" in table  # NaN-dropped lit n is shown
    assert any("lit columns average over 1 of 2" in w for w in warnings)
    assert "pretrained_albedo_ctrl †" in table and "target = albedo" in table  # control row is marked
    assert summary["arms"]["pretrained_albedo_ctrl"]["target"] == "albedo"
    assert "MAE all | L1 [-1,1] all" in table  # value spaces are labelled
    pv = summary["arms"]["ours_pbr2emission"]["paired_vs"]
    assert pv["n_common"] == 2 and pv["mae01_all"]["win_frac"] == 1.0
    table2, summary2, _ = m.summarize(str(p), ["a", "b", "c"], paired_ref="pretrained_pad",
                                      only_arms=["pretrained_pad", "ours_pbr2emission"])
    assert set(summary2["arms"]) == {"pretrained_pad", "ours_pbr2emission"} and "albedo_ctrl" not in table2


def test_csv_header_check(tmp_path):
    p = tmp_path / "per_shape.csv"
    p.write_text(",".join(m.CSV_COLS) + "\n")
    m.check_csv_header(str(p))  # current header: fine
    p.write_text(",".join(m.CSV_COLS[:-1]) + "\n")  # a column dropped: must refuse
    try:
        m.check_csv_header(str(p))
        assert False, "expected a header mismatch error"
    except RuntimeError as e:
        assert "header" in str(e)
    (tmp_path / "empty.csv").write_text("")
    m.check_csv_header(str(tmp_path / "empty.csv"))  # empty file: nothing to check


def test_psnr_cap_and_clamp():
    t = torch.zeros(10, 3)  # 0.5 in [0,1]
    pred = torch.full((10, 3), 5.0)  # way above 1 -> clamped to 1 -> |1-0.5| = 0.5
    r = m.compute_metrics(pred, t)
    assert abs(r["mae01_all"] - 0.5) < 1e-6
    assert abs(r["psnr_all"] - 10 * math.log10(1 / 0.25)) < 1e-6
    assert abs(r["l1_all"] - 5.0) < 1e-6  # l1 is on the raw, unclamped [-1,1] prediction
    assert m.psnr_from_mse(0.0) == m.PSNR_CAP and m.psnr_from_mse(1e-9) == m.PSNR_CAP


def test_packing_matches_trainer_constants():
    emit = torch.rand(7, 3) * 2 - 1
    x = m.pack_pad(emit)
    assert x.shape == (7, 6)
    assert torch.equal(x[:, :3], emit)
    assert torch.all(x[:, 3] == -1.0) and torch.all(x[:, 4] == 1.0) and torch.all(x[:, 5] == 1.0)
    pbr = torch.rand(7, 6) * 2 - 1
    y = m.pack_realpbr(emit, pbr)
    assert torch.equal(y[:, :3], emit) and torch.equal(y[:, 3:], pbr[:, 3:])


def test_paired_stats_identical_and_shifted():
    import numpy as np
    a = np.linspace(0.1, 0.5, 40)
    r = m.paired_stats(a, a.copy(), lower_is_better=True)
    assert r["mean_delta"] == 0.0 and r["ci95"] == [0.0, 0.0] and r["win_frac"] == 0.0 and r["n"] == 40
    r = m.paired_stats(a - 0.01, a, lower_is_better=True)  # uniformly better by 0.01
    assert abs(r["mean_delta"] + 0.01) < 1e-9 and r["win_frac"] == 1.0
    assert r["ci95"][0] <= -0.01 <= r["ci95"][1] + 1e-9
    r = m.paired_stats(a + 0.01, a, lower_is_better=False)  # higher-is-better metric, uniformly higher
    assert r["win_frac"] == 1.0 and r["mean_delta"] > 0


def test_summarize_reports_an_arm_with_no_successful_rows(tmp_path):
    import csv
    p = tmp_path / "per_shape.csv"
    rows = []
    for sha in ("a", "b"):
        d = {c: "" for c in m.CSV_COLS}; d.update(arm="pretrained_pad", sha256=sha, status="ok", target="emission", seconds="0")
        d.update({k: "0.1" for k in m.METRIC_COLS}); rows.append(d)
        d = {c: "" for c in m.CSV_COLS}; d.update(arm="ours_pbr2emission", sha256=sha, status="oom", target="emission", seconds="0"); rows.append(d)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m.CSV_COLS); w.writeheader(); w.writerows(rows)
    table, summary, warnings = m.summarize(str(p), ["a", "b"], paired_ref="pretrained_pad")
    assert "ours_pbr2emission" not in summary["arms"]
    assert any("ours_pbr2emission: 0 successful rows (2 failed) -- arm OMITTED" in w for w in warnings)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
