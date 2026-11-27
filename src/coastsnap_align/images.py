"""Consistent EXIF orientation, pixel-center coordinates and binary masks."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 127
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} differs from oriented target shape {shape}")
    if not mask.any():
        raise ValueError("Target mask contains no included pixels")
    return mask


def resize_for_matching(image: np.ndarray, longest: int):
    h, w = image.shape[:2]
    ratio = min(1.0, longest / max(h, w))
    wh = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    small = cv2.resize(image, wh, interpolation=cv2.INTER_AREA) if wh != (w, h) else image
    # Mapping from resized pixel centers to original pixel centers.
    return small, np.array([w / wh[0], h / wh[1]], dtype=float)


def original_points(points, scale):
    return (np.asarray(points, dtype=float) + 0.5) * scale - 0.5


def points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-pixel mask sampling. Never clip out-of-bounds points onto a valid edge."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    h, w = mask.shape
    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] <= w - 1)
    valid &= (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
    indices = np.flatnonzero(valid)
    pixels = np.floor(points[indices] + 0.5).astype(int)
    valid[indices] &= mask[pixels[:, 1], pixels[:, 0]]
    return valid


def polygon_mask(target: Path, polygons: Path, output: Path):
    """Rasterize JSON {include: [polygon, ...], exclude: [polygon, ...]}."""
    import json

    data = json.loads(polygons.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"include", "exclude"}:
        raise ValueError("Polygon JSON accepts only include and exclude lists")
    image = read_rgb(target)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8) if data.get("include") else np.full((h, w), 255, np.uint8)
    for key, value in (("include", 255), ("exclude", 0)):
        for polygon in data.get(key, []):
            pts = np.asarray(polygon, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
                raise ValueError("Each polygon needs at least three finite [x,y] vertices")
            if (pts < 0).any() or (pts[:, 0] > w - 1).any() or (pts[:, 1] > h - 1).any():
                raise ValueError("Polygon vertices must lie inside the oriented target image")
            cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], value)
    if not mask.any():
        raise ValueError("Polygon mask contains no included pixels")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    preview = image.copy()
    preview[mask == 0] = preview[mask == 0] // 4
    Image.fromarray(preview).save(output.with_name(output.stem + "_preview.png"))
