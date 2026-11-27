"""Internal fit and appearance diagnostics; none is independent ground-truth accuracy."""

import cv2
import numpy as np

from .geometry import project


def ncc(first, second, support):
    if support.sum() < 16:
        return None
    a = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)[support].astype(float)
    b = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)[support].astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-10 else None


def spatial_coverage(points, mask):
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, np.float32))
    coverage = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(coverage, np.rint(hull).astype(np.int32), 1)
    return float(np.count_nonzero((coverage != 0) & mask) / mask.sum())


def calculate(source, target, inliers, matrix, source_image, target_image, aligned, valid, mask):
    forward = np.linalg.norm(project(source, matrix) - target, axis=1)
    backward = np.linalg.norm(project(target, np.linalg.inv(matrix)) - source, axis=1)
    # Dimensionless symmetric error avoids mixing differently sized image pixel units.
    src_diagonal = np.hypot(*source_image.shape[:2])
    dst_diagonal = np.hypot(*target_image.shape[:2])
    symmetric = np.sqrt(((forward / dst_diagonal) ** 2 + (backward / src_diagonal) ** 2) / 2)
    selected = forward[inliers]
    overlap = valid & mask
    baseline = np.zeros_like(target_image)
    baseline_valid = np.zeros(mask.shape, bool)
    h = min(source_image.shape[0], target_image.shape[0])
    w = min(source_image.shape[1], target_image.shape[1])
    baseline[:h, :w] = source_image[:h, :w]
    baseline_valid[:h, :w] = True
    # Both NCC values use exactly the same pixels. The baseline is identity, without resizing.
    common = overlap & baseline_valid
    before = ncc(baseline, target_image, common)
    after = ncc(aligned, target_image, common)
    metrics = {
        "matches": len(source),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "inlier_error_median_px": float(np.median(selected)),
        "inlier_error_rmse_px": float(np.sqrt(np.mean(selected**2))),
        "inlier_error_p95_px": float(np.percentile(selected, 95)),
        "all_match_error_median_px": float(np.median(forward)),
        "inlier_reverse_error_median_source_px": float(np.median(backward[inliers])),
        "inlier_symmetric_error_median_normalized": float(np.median(symmetric[inliers])),
        "mask_overlap_fraction": float(overlap.sum() / mask.sum()),
        "inlier_mask_coverage_fraction": spatial_coverage(target[inliers], mask),
        "ncc_common_pixels": int(common.sum()),
        "ncc_before": before,
        "ncc_after": after,
        "ncc_change": after - before if before is not None and after is not None else None,
    }
    return metrics, forward, backward


def rejection_reasons(metrics, config):
    checks = [
        ("inliers", config.min_inliers, "minimum", lambda a, b: a < b),
        ("inlier_ratio", config.min_inlier_ratio, "minimum", lambda a, b: a < b),
        ("inlier_error_median_px", config.max_median_error, "maximum", lambda a, b: a > b),
        ("mask_overlap_fraction", config.min_overlap, "minimum", lambda a, b: a < b),
        ("inlier_mask_coverage_fraction", config.min_coverage, "minimum", lambda a, b: a < b),
    ]
    return [
        f"{key}={metrics[key]:.4g} violates {direction} {limit}"
        for key, limit, direction, failed in checks
        if failed(metrics[key], limit)
    ]
