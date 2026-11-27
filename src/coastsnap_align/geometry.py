"""Robust source-to-target estimation and projective plausibility checks."""

import cv2
import numpy as np


class RegistrationRejected(ValueError):
    """Expected quality rejection, distinct from an unexpected processing error."""


def project(points, matrix):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    denominator = homogeneous[:, 2]
    result = np.full((len(points), 2), np.nan)
    valid = np.abs(denominator) > 1e-10
    result[valid] = homogeneous[valid, :2] / denominator[valid, None]
    return result


def estimate(source, target, config):
    if len(source) < config.min_matches:
        raise RegistrationRejected(
            f"Only {len(source)} valid masked matches; need {config.min_matches}"
        )
    cv2.setRNGSeed(config.seed)
    kwargs = dict(
        ransacReprojThreshold=config.ransac_threshold,
        maxIters=config.ransac_iterations,
        confidence=config.ransac_confidence,
    )
    if config.model == "homography":
        matrix, inliers = cv2.findHomography(source, target, method=cv2.RANSAC, **kwargs)
    else:
        solver = cv2.estimateAffine2D if config.model == "affine" else cv2.estimateAffinePartial2D
        matrix, inliers = solver(source, target, method=cv2.RANSAC, **kwargs)
        if matrix is not None:
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix is None or inliers is None:
        raise RegistrationRejected("Robust estimation did not find a transform")
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise RegistrationRejected("Transform is singular or non-finite")
    matrix = matrix / np.linalg.norm(matrix)
    # Recompute using the declared target-pixel threshold after OpenCV refinement.
    error = np.linalg.norm(project(source, matrix) - target, axis=1)
    inliers = (inliers.ravel() != 0) & np.isfinite(error) & (error <= config.ransac_threshold)
    if not inliers.any():
        raise RegistrationRejected("No inliers after refined-transform validation")
    return matrix, inliers


def validate_footprint(matrix, source_shape):
    h, w = source_shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
    denominators = np.column_stack([corners, np.ones(4)]) @ matrix[2]
    if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
        raise RegistrationRejected("Projective horizon crosses the source image")
    transformed = project(corners, matrix)
    if not np.isfinite(transformed).all() or np.max(np.abs(transformed)) > 1e8:
        raise RegistrationRejected("Transform has an unbounded image footprint")
    contour = transformed.astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1:
        raise RegistrationRejected("Transform has a collapsed or nonconvex footprint")
    return transformed


def warp(image, matrix, target_shape):
    h, w = target_shape[:2]
    aligned = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    # Linear interpolation of a support image identifies fully supported output pixels.
    support = cv2.warpPerspective(
        np.ones(image.shape[:2], np.float32), matrix, (w, h), flags=cv2.INTER_LINEAR
    )
    return aligned, support >= 1 - 1e-6
