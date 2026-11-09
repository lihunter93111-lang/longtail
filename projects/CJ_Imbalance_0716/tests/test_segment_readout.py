"""Numerical contracts for the integrated residual graph."""
import numpy as np
import pytest
import torch
from cjlt.engine.feature_transform import FeatureTransform
from cjlt.engine.model_bundle import load_bundle, save_bundle
from cjlt.models.classwise_fusion import ClasswiseReadoutModel, FrozenLogitFusion
from cjlt.engine.selective_fusion import apply_fusion, fit_affine_fusion, fit_selective_fusion
from cjlt.models.segment_readout import segment_statistics


def test_temporal_statistics_preserve_local_information():
    left = torch.tensor([[[1., 1., 0., 0., -1., -1., 0., 0.]]])
    right = left.roll(2, -1)
    a, b = segment_statistics(left), segment_statistics(right)
    assert a.shape == (1, 14)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert not torch.equal(a[:, 2:], b[:, 2:])


def test_transform_is_training_only_and_has_matching_numpy_projection():
    values = np.random.default_rng(2).normal(size=(32, 20)).astype(np.float32)
    with pytest.raises(ValueError, match="fit identities"):
        FeatureTransform.fit(values, 8, role="report")
    transform = FeatureTransform.fit(values, 8)
    np.testing.assert_allclose(transform(torch.from_numpy(values)).numpy(),
                               transform.project(values), rtol=1e-5, atol=2e-6)


def test_native_bundle_preserves_fusion_and_non_target_columns(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = ClasswiseReadoutModel(backbone={"width": 4, "dropout": 0.2},
                                 expert=True, linear=True, local=True, paired=True).eval()
    signals = torch.randn(4, 2, 64)
    with torch.inference_mode():
        before = model(signals)
    model.add_readout("boundary", ["SawFM", "MFT"], dimensions=8)
    with torch.no_grad():
        model.readouts["boundary"].output.bias.copy_(torch.tensor([0.2, -0.1]))
    with torch.inference_mode():
        after = model(signals)
    torch.testing.assert_close(after[:, [0, 1, 3, 5]], before[:, [0, 1, 3, 5]], rtol=0, atol=0)
    torch.testing.assert_close(after[:, [2, 4]] - before[:, [2, 4]],
                               torch.tensor([[0.2, -0.1]]).expand(4, 2))
    path = tmp_path / "model.pt"
    save_bundle(path, model, [0.5] * 6)
    restored, thresholds, _ = load_bundle(path)
    with torch.inference_mode():
        torch.testing.assert_close(after, restored(signals), rtol=0, atol=0)
    assert thresholds == [0.5] * 6


@pytest.mark.parametrize("kind", ["affine", "selective"])
def test_frozen_logit_fusion_matches_offline_and_reloads(tmp_path, kind):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    left = ClasswiseReadoutModel(backbone={"width": 4, "dropout": 0.2}).eval()
    right = ClasswiseReadoutModel(backbone={"width": 4, "dropout": 0.2}).eval()
    signal = torch.randn(12, 2, 64)
    with torch.inference_mode():
        a, b = left(signal).numpy(), right(signal).numpy()
    target = np.random.default_rng(1).integers(0, 2, size=a.shape)
    if kind == "affine":
        state = fit_affine_fusion(a, b, target, targets=["SawFM"], epochs=2)
    else:
        state = fit_selective_fusion(a, b, target, [0.4] * 6, [0.6] * 6,
                                     targets=["SawFM"], epochs=2)
    expected, _ = apply_fusion(a, b, state)
    model = FrozenLogitFusion(left, right, state).eval()
    with torch.inference_mode():
        actual = model(signal).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(actual[:, [0, 1, 3, 4, 5]], a[:, [0, 1, 3, 4, 5]])
    path = tmp_path / "fusion.pt"
    save_bundle(path, model, [0.5] * 6)
    restored, _, _ = load_bundle(path)
    with torch.inference_mode():
        torch.testing.assert_close(model(signal), restored(signal), rtol=0, atol=0)
