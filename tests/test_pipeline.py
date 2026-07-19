"""Unit tests for the parts where a silent bug would corrupt the science.

Nothing here needs downloaded data. The emphasis is on invariants a wrong
implementation would violate quietly: symmetry, mask/loss agreement, exact
area-weighted resampling, distance matching, and reverse-complement correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from seq2hic import ep_loops as EP
from seq2hic import metrics as M
from seq2hic import targets as T
from seq2hic.config import load_config
from seq2hic.genome import one_hot_encode, reverse_complement
from seq2hic.model import Seq2HiC, masked_mse

CFG = "configs/smoke.yaml"


# --------------------------------------------------------------- sequence


def test_one_hot_marks_ambiguity_as_zero_not_uniform():
    x = one_hot_encode(b"ACGTN")
    assert x.shape == (4, 5)
    assert np.array_equal(x[:, 0], [1, 0, 0, 0])
    assert x[:, 4].sum() == 0.0, "N must be all-zero, not 0.25 everywhere"


def test_reverse_complement_is_an_involution():
    x = one_hot_encode(b"ACGTTGCAAT")
    assert np.allclose(reverse_complement(reverse_complement(x)), x)


def test_reverse_complement_maps_A_to_T():
    x = one_hot_encode(b"AAAA")
    rc = reverse_complement(x)
    assert np.array_equal(rc, one_hot_encode(b"TTTT"))


def test_reverse_complement_leaves_extra_channels_unpermuted():
    """ATAC is strand-agnostic: it must be reversed but never complemented."""
    seq = one_hot_encode(b"ACGT")
    atac = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    x = np.concatenate([seq, atac], axis=0)
    rc = reverse_complement(x)
    assert np.array_equal(rc[4], [4.0, 3.0, 2.0, 1.0])


# --------------------------------------------------------------- resampling


def test_overlap_weights_rows_sum_to_one():
    w = T.overlap_weights(0, 80_000, 10, 0, 5_000, 16)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_area_resample_preserves_a_constant():
    """A constant matrix must resample to the same constant -- if it does not,
    the weights are wrong and every target is biased."""
    mat = np.full((16, 16), 3.0)
    w = T.overlap_weights(0, 80_000, 10, 0, 5_000, 16)
    out = T.area_resample(mat, w)
    assert np.allclose(out, 3.0)


def test_area_resample_is_identity_when_grids_align():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(8, 8))
    mat = 0.5 * (mat + mat.T)
    w = T.overlap_weights(0, 8 * 5000, 8, 0, 5000, 8)
    assert np.allclose(T.area_resample(mat, w), mat)


def test_area_resample_treats_nan_as_missing_not_zero():
    mat = np.full((4, 4), 2.0)
    mat[0, 0] = np.nan
    w = T.overlap_weights(0, 4 * 1000, 2, 0, 1000, 4)
    out = T.area_resample(mat, w)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 2.0), "NaN must be excluded from the average, not counted as 0"


# --------------------------------------------------------- coarse-graining


def test_adaptive_coarsegrain_leaves_dense_pixels_untouched():
    balanced = np.full((8, 8), 1.5)
    counts = np.full((8, 8), 100.0)  # everything already above cutoff
    out = T.adaptive_coarsegrain(balanced, counts, cutoff=5.0, max_levels=3)
    assert np.allclose(out, 1.5)


def test_adaptive_coarsegrain_fills_sparse_pixels():
    rng = np.random.default_rng(1)
    balanced = rng.normal(1.0, 0.5, size=(16, 16))
    balanced = 0.5 * (balanced + balanced.T)
    counts = np.ones((16, 16))  # everywhere below cutoff -> must pool
    out = T.adaptive_coarsegrain(balanced, counts, cutoff=5.0, max_levels=4)
    assert np.all(np.isfinite(out))
    assert out.std() < balanced.std(), "pooling should reduce variance"


# --------------------------------------------------------------- targets


def test_to_target_zeroes_and_masks_the_same_band():
    """The near-diagonal band must be zero in the target AND false in the mask.
    Doing only one of the two is the bug this test exists for."""
    raw = np.full((16, 16), 2.0)
    expected = np.ones(16)
    tgt, valid = T.to_target(raw, expected, clip=2.0, diagonal_offset=2)
    ii, jj = np.indices((16, 16))
    band = np.abs(ii - jj) < 2
    assert np.all(tgt[band] == 0.0)
    assert not valid[band].any()
    assert valid[~band].all()


def test_to_target_is_symmetric_and_clipped():
    rng = np.random.default_rng(2)
    raw = np.abs(rng.lognormal(0, 2, size=(24, 24)))
    raw = 0.5 * (raw + raw.T)
    tgt, valid = T.to_target(raw, np.ones(24), clip=2.0, diagonal_offset=2)
    assert np.allclose(tgt, tgt.T)
    assert np.array_equal(valid, valid.T)
    assert tgt.max() <= 2.0 and tgt.min() >= -2.0


def test_to_target_detrend_removes_distance_decay():
    """Given a pure distance-decay matrix, the detrended target must be ~0."""
    n = 32
    ii, jj = np.indices((n, n))
    d = np.abs(ii - jj)
    decay = np.exp(-d / 5.0)
    expected = np.exp(-np.arange(n) / 5.0)
    tgt, valid = T.to_target(decay, expected, clip=2.0, diagonal_offset=2)
    assert np.allclose(tgt[valid], 0.0, atol=1e-9)


# ---------------------------------------------------------------- model


def _smoke_model(atac: bool):
    cfg = load_config(CFG)
    cfg["inputs"]["atac"] = atac
    return cfg, Seq2HiC(4 + int(atac), cfg["window"]["n_bins"], dict(cfg["model"]))


@pytest.mark.parametrize("atac", [False, True])
def test_model_shapes(atac):
    cfg, model = _smoke_model(atac)
    model.eval()
    x = torch.randn(2, 4 + int(atac), cfg["window"]["bp"])
    with torch.no_grad():
        y = model(x)
    n = cfg["window"]["n_bins"]
    assert y.shape == (2, n, n)


def test_model_output_is_symmetric():
    _, model = _smoke_model(False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 4, 4096))
    assert torch.allclose(y, y.transpose(-1, -2), atol=1e-5)


def test_model_rejects_wrong_channel_count():
    _, model = _smoke_model(False)
    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(1, 5, 4096))


def test_default_config_receptive_field_covers_the_window():
    """The 2D stack must reach across the whole window, or corner bins cannot
    see their own anchors. This caught a real 6-block config that only reached
    half the window."""
    cfg = load_config("configs/seq_atac.yaml")
    model = Seq2HiC(5, cfg["window"]["n_bins"], dict(cfg["model"]))
    assert model.covers_window(), (
        f"2D receptive field {model.receptive_field_2d_bins} bins < "
        f"{cfg['window']['n_bins']} bins; increase model.dilated2d_blocks"
    )


def test_receptive_field_uses_configured_dilation_base():
    cfg = load_config(CFG)
    m = dict(cfg["model"])
    m["dilated2d_blocks"], m["dilation_base"] = 3, 2
    model = Seq2HiC(4, cfg["window"]["n_bins"], m)
    assert model.receptive_field_2d_bins == 1 + 2 * (1 + 2 + 4)


def test_trunk_pools_product_matches_bin_size():
    cfg, model = _smoke_model(False)
    assert model.downsample_factor == cfg["window"]["bp"] // cfg["window"]["n_bins"]


def test_config_rejects_mismatched_pools():
    cfg = load_config(CFG)
    from seq2hic.config import _validate

    cfg["model"]["trunk_pools"] = [2, 2]
    cfg["model"]["trunk_channels"] = [8, 8]
    with pytest.raises(ValueError, match="prod\\(trunk_pools\\)"):
        _validate(cfg)


# ----------------------------------------------------------------- loss


def test_masked_mse_ignores_masked_elements():
    n = 16
    pred = torch.zeros(1, n, n)
    targ = torch.full((1, n, n), 5.0)
    mask = torch.zeros(1, n, n, dtype=torch.bool)
    mask[0, 0, 8] = True  # a single valid upper-triangle element
    assert torch.isclose(masked_mse(pred, targ, mask), torch.tensor(25.0))


def test_masked_mse_survives_an_all_masked_window():
    pred = torch.zeros(1, 8, 8, requires_grad=True)
    loss = masked_mse(pred, torch.ones(1, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)


def test_masked_mse_uses_only_upper_triangle():
    n = 8
    pred, targ = torch.zeros(1, n, n), torch.zeros(1, n, n)
    targ[0, 5, 1] = 10.0  # lower triangle only
    mask = torch.ones(1, n, n, dtype=torch.bool)
    assert torch.isclose(masked_mse(pred, targ, mask), torch.tensor(0.0))


# --------------------------------------------------------------- metrics


def test_map_correlation_is_one_for_perfect_prediction():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(1, 32, 32))
    t = 0.5 * (t + t.transpose(0, 2, 1))
    m = np.ones((1, 32, 32), dtype=bool)
    assert np.isclose(M.map_correlation(t, t, m)[0], 1.0)


def test_map_correlation_is_nan_for_constant_prediction():
    """A constant prediction has undefined correlation. Returning 0 would be a
    silent lie that averages into the reported score."""
    t = np.random.default_rng(4).normal(size=(1, 32, 32))
    p = np.zeros_like(t)
    m = np.ones((1, 32, 32), dtype=bool)
    assert np.isnan(M.map_correlation(p, t, m)[0])


def test_insulation_edges_are_nan_not_truncated():
    mat = np.random.default_rng(5).normal(size=(40, 40))
    ins = M.insulation_score(0.5 * (mat + mat.T), window_bins=10)
    assert np.all(np.isnan(ins[:10])) and np.all(np.isnan(ins[-10:]))
    assert np.isfinite(ins[10:-10]).any()


def test_paired_bootstrap_detects_a_real_shift():
    rng = np.random.default_rng(6)
    a = rng.normal(0.5, 0.05, 200)
    b = a + 0.03
    d = M.paired_bootstrap_delta(list(a), list(b), n_boot=300)
    assert d["lo"] > 0, "a uniform +0.03 shift must have a CI excluding zero"


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        M.paired_bootstrap_delta([1.0, 2.0], [1.0], n_boot=10)


# -------------------------------------------------------------- E-P pairs


def test_filter_distal_drops_promoter_proximal_peaks():
    centers = np.array([1_000, 50_000, 100_000])
    tss = np.array([1_200, 99_000])
    kept = EP.filter_distal(centers, tss, exclusion_bp=2_000)
    assert kept.tolist() == [50_000]


def test_distance_matched_negatives_match_the_positive_distribution():
    rng = np.random.default_rng(7)
    n = 4000
    dist = rng.integers(20_000, 2_000_000, n)
    # positives concentrated at short range -- unmatched sampling would be caught
    labels = (dist < 300_000) & (rng.random(n) < 0.25)
    keep = EP.distance_matched_subset(labels, dist, negatives_per_positive=5, seed=0)
    kl, kd = labels[keep], dist[keep]
    assert kl.sum() == labels.sum(), "all positives must be retained"
    pos_med, neg_med = np.median(kd[kl]), np.median(kd[~kl])
    assert abs(pos_med - neg_med) / pos_med < 0.35


def test_distance_baseline_scores_are_monotone_in_distance():
    d = np.array([100_000, 50_000, 500_000])
    s = EP.distance_only_baseline(d)
    assert np.argmax(s) == 1 and np.argmin(s) == 2


def test_ep_score_reports_positive_rate_with_auprc():
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    scores = np.array([9, 1, 2, 3, 8, 4, 5, 6, 7, 0], dtype=float)
    r = EP.score(labels, scores)
    assert r["auprc"] > 0.9
    assert np.isclose(r["positive_rate"], 0.2)


def test_ep_score_handles_degenerate_label_sets():
    r = EP.score(np.zeros(10, dtype=bool), np.arange(10.0))
    assert np.isnan(r["auprc"])
