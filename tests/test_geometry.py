import numpy as np
import pytest

from coastsnap_align.config import RunConfig
from coastsnap_align.geometry import (
    RegistrationRejected,
    estimate,
    project,
    validate_footprint,
    warp,
)
from coastsnap_align.images import original_points, points_in_mask, resize_for_matching
from coastsnap_align.metrics import calculate, rejection_reasons


def config(**kwargs):
    return RunConfig(target="t", images="i", mask="m", output="o", **kwargs)


@pytest.mark.parametrize("model", ["homography", "affine", "similarity"])
def test_robust_fit_maps_source_to_target_despite_outliers(model):
    rng = np.random.default_rng(71)
    source = rng.uniform(20, 450, (100, 2))
    expected = np.array([[1.02, -0.04, 31], [0.04, 1.02, -17], [0, 0, 1]])
    target = project(source, expected)
    target[:25] = rng.uniform(0, 500, (25, 2))
    fitted, inliers = estimate(source, target, config(model=model))
    held_out = rng.uniform(20, 450, (20, 2))
    np.testing.assert_allclose(project(held_out, fitted), project(held_out, expected), atol=0.01)
    assert inliers.sum() == 75


def test_degenerate_points_and_projective_horizon_are_rejected():
    points = np.ones((20, 2))
    with pytest.raises(RegistrationRejected):
        estimate(points, points, config())
    horizon = np.array([[1, 0, 0], [0, 1, 0], [0.02, 0, -1]])
    with pytest.raises(RegistrationRejected, match="horizon"):
        validate_footprint(horizon, (100, 100))


def test_masks_reject_outside_and_nonfinite_points_without_clipping():
    mask = np.ones((10, 10), bool)
    mask[2:5, 2:5] = False
    points = np.array([[0, 0], [-0.1, 0], [9, 9], [9.1, 9], [3, 3], [np.nan, 1]])
    assert points_in_mask(points, mask).tolist() == [True, False, True, False, False, False]


def test_resize_uses_actual_rounded_scales_and_pixel_centers():
    image = np.zeros((301, 503, 3), np.uint8)
    small, scale = resize_for_matching(image, 200)
    assert small.shape[:2] == (120, 200)
    np.testing.assert_allclose(original_points([[0, 0]], scale), [(scale - 1) / 2])
    point = np.array([[321.5, 190.25]])
    resized_point = (point + 0.5) / scale - 0.5
    np.testing.assert_allclose(original_points(resized_point, scale), point)


def test_metrics_ignore_padding_and_compare_ncc_on_same_support():
    rng = np.random.default_rng(2)
    source = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
    matrix = np.array([[1.0, 0, 10], [0, 1, 0], [0, 0, 1]])
    target, valid = warp(source, matrix, source.shape)
    points = rng.uniform([5, 5], [80, 70], (30, 2))
    projected = project(points, matrix)
    mask = np.ones(source.shape[:2], bool)
    metrics, _, _ = calculate(
        points, projected, np.ones(30, bool), matrix, source, target, target, valid, mask
    )
    assert metrics["mask_overlap_fraction"] == pytest.approx(0.9)
    assert metrics["ncc_after"] == pytest.approx(1)
    assert metrics["ncc_before"] < 0.1
    assert metrics["ncc_common_pixels"] == valid.sum()
    assert metrics["inlier_error_rmse_px"] < 1e-9
    assert not rejection_reasons(metrics, config())


def test_textureless_ncc_is_missing_instead_of_nan():
    from coastsnap_align.metrics import ncc

    image = np.zeros((10, 10, 3), np.uint8)
    assert ncc(image, image, np.ones((10, 10), bool)) is None


@pytest.mark.parametrize(
    "setting",
    [
        {"min_overlap": 2},
        {"max_keypoints": 0},
        {"model": "warp"},
        {"ransac_threshold": float("nan")},
        {"warmup": "false"},
        {"seed": -1},
        {"flex_descriptor_size": 513},
    ],
)
def test_invalid_config(setting):
    with pytest.raises(ValueError):
        config(**setting).validate()
