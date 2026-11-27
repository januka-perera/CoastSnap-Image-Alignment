"""Batch orchestration shared by every matcher backend."""

import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .backends import backend_names, create_backend
from .geometry import RegistrationRejected, estimate, validate_footprint, warp
from .images import (
    IMAGE_SUFFIXES,
    original_points,
    points_in_mask,
    read_mask,
    read_rgb,
    resize_for_matching,
)
from .metrics import calculate, rejection_reasons
from .reporting import comparison_plots, save_diagnostics, summarize

LOG = logging.getLogger(__name__)


def write_json(path, value):
    # Refuse NaN/Infinity, which are not valid interoperable JSON values.
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def environment():
    result = {
        "package": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {},
    }
    for name in (
        "numpy",
        "opencv-python",
        "Pillow",
        "matplotlib",
        "torch",
        "torchvision",
        "lightglue",
        "flexglue",
    ):
        try:
            dist = importlib.metadata.distribution(name)
            item = {"version": dist.version}
            direct = dist.read_text("direct_url.json")
            if direct:
                item["source"] = json.loads(direct)
            result["dependencies"][name] = item
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        result["cuda_available"] = False
    return result


def _folder_name(relative):
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", relative.stem)[:60] or "image"
    return slug + "_" + hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12]


def discover(config):
    target = Path(config.target).resolve()
    mask = Path(config.mask).resolve()
    root = Path(config.images).resolve()
    output = Path(config.output).resolve()
    if not target.is_file() or not mask.is_file() or not root.is_dir():
        raise ValueError("target/mask must be files and images must be a directory")
    # Prevent a run from consuming its own output, even on a future invocation.
    if output == root or root in output.parents or output in root.parents:
        raise ValueError(
            "Output and input image directories must be separate, non-nested directories"
        )
    if (
        output == target.parent
        or output == mask.parent
        or output in target.parents
        or output in mask.parents
    ):
        raise ValueError("Output directory must not contain the target or mask")
    paths = root.rglob("*") if config.recursive else root.iterdir()
    files = sorted(
        (
            p
            for p in paths
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and p.resolve() not in {target, mask}
        ),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("No source images found (the target and mask are excluded)")
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory must be new or empty; choose a new path for each experiment"
        )
    return target, mask, root, output, files


def _process(backend, path, target, target_scale, mask, config, folder):
    row = {"status": "error", "reason": "", "match_seconds": None}
    start = time.perf_counter()
    try:
        source = read_rgb(path)
        small, scale = resize_for_matching(source, config.max_image_size)
        backend.synchronize()
        match_start = time.perf_counter()
        matches = backend.match(small)
        backend.synchronize()
        row["match_seconds"] = time.perf_counter() - match_start
        matches.validate()
        src = original_points(matches.source, scale)
        dst = original_points(matches.target, target_scale)
        keep = points_in_mask(dst, mask) & np.isfinite(matches.scores)
        # Source bounds are checked too, including predicted FlexGlue correspondences.
        sh, sw = source.shape[:2]
        keep &= np.isfinite(src).all(axis=1)
        keep &= (src[:, 0] >= 0) & (src[:, 0] <= sw - 1)
        keep &= (src[:, 1] >= 0) & (src[:, 1] <= sh - 1)
        src, dst, scores = src[keep], dst[keep], matches.scores[keep]
        row.update(
            source_width=sw,
            source_height=sh,
            raw_matches=len(keep),
            matches=len(src),
            source_keypoints=matches.source_keypoints,
            target_keypoints=matches.target_keypoints,
            backend_metadata=matches.metadata,
        )
        # Keep the original returned correspondences even if robust estimation rejects the pair.
        np.savez_compressed(folder / "matches.npz", source=src, target=dst, scores=scores)
        matrix, inliers = estimate(src, dst, config)
        validate_footprint(matrix, source.shape)
        aligned, valid = warp(source, matrix, target.shape)
        metrics, forward, backward = calculate(
            src, dst, inliers, matrix, source, target, aligned, valid, mask
        )
        row.update(metrics)
        reasons = rejection_reasons(metrics, config)
        row["status"] = "rejected" if reasons else "accepted"
        row["reason"] = "; ".join(reasons)
        row["source_to_target"] = matrix.tolist()
        np.savez_compressed(
            folder / "matches.npz",
            source=src,
            target=dst,
            scores=scores,
            inliers=inliers,
            forward_error_px=forward,
            reverse_error_source_px=backward,
            source_to_target=matrix,
        )
        # Excludes diagnostic/output-image writing and model/target setup.
        row["processing_seconds"] = time.perf_counter() - start
        if config.save_aligned:
            filename = "aligned.png" if row["status"] == "accepted" else "rejected_candidate.png"
            Image.fromarray(aligned).save(folder / filename)
            Image.fromarray(valid.astype(np.uint8) * 255).save(folder / "valid_pixels.png")
        if config.save_diagnostics:
            save_diagnostics(folder, source, target, aligned, valid, src, dst, inliers, mask)
    except RegistrationRejected as exc:
        row.update(status="rejected", reason=str(exc))
    except Exception as exc:
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        LOG.exception("Image processing failed for %s", path)
    row.setdefault("processing_seconds", time.perf_counter() - start)
    row["pair_wall_seconds"] = time.perf_counter() - start
    return row


def _write_tables(records, output, methods):
    # Full JSONL preserves structured transforms/metadata; CSV stays analysis-friendly.
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    columns = sorted(
        set().union(*(r.keys() for r in records)) - {"source_to_target", "backend_metadata"}
    )
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records, methods)
    write_json(output / "summary.json", summary)
    return summary


def run(config, backend_factory=None):
    """Run an experiment; backend_factory is injectable for integrations and offline tests."""
    config.validate()
    if backend_factory is None:
        unknown = set(config.methods) - set(backend_names())
        if unknown:
            raise ValueError(f"Unknown methods: {', '.join(sorted(unknown))}")
        backend_factory = create_backend
    target_path, mask_path, root, output, files = discover(config)
    target = read_rgb(target_path)
    mask = read_mask(mask_path, target.shape[:2])
    target_small, target_scale = resize_for_matching(target, config.max_image_size)
    small_mask = cv2.resize(
        mask.astype(np.uint8),
        (target_small.shape[1], target_small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if not small_mask.any():
        raise ValueError("Mask vanished at matching resolution; increase max_image_size")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "environment": environment(),
        "target": fingerprint(target_path),
        "mask": fingerprint(mask_path),
        "images": [],
        "methods": {},
        "evaluation": "Automated proxy QC only; no independent accuracy measurement",
    }
    for path in files:
        try:
            manifest["images"].append(fingerprint(path))
        except OSError as exc:
            manifest["images"].append({"path": str(path), "read_error": str(exc)})
    write_json(output / "manifest.json", manifest)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "target_mask.png")
    preview = target.copy()
    preview[~mask] //= 4
    Image.fromarray(preview).save(output / "target_mask_preview.png")
    records = []
    for method in config.methods:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", method):
            raise ValueError(
                "Backend names may contain only letters, digits, underscores and hyphens"
            )
        method_dir = output / method
        method_dir.mkdir()
        backend = None
        method_meta = {}
        try:
            begin = time.perf_counter()
            backend = backend_factory(method, config)
            backend.synchronize()
            method_meta["initialization_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            backend.prepare_target(target_small, small_mask)
            backend.synchronize()
            method_meta["target_preparation_seconds"] = time.perf_counter() - begin
            if config.warmup:
                # Warm up on the first readable source, never secretly drop a benchmark image.
                for path in files:
                    try:
                        warmup_image, _ = resize_for_matching(read_rgb(path), config.max_image_size)
                    except (OSError, ValueError):
                        continue
                    begin = time.perf_counter()
                    try:
                        backend.match(warmup_image)
                        backend.synchronize()
                        method_meta["warmup_seconds"] = time.perf_counter() - begin
                    except Exception as exc:
                        method_meta["warmup_error"] = f"{type(exc).__name__}: {exc}"
                    break
        except Exception as exc:
            method_meta["setup_error"] = f"{type(exc).__name__}: {exc}"
            (method_dir / "setup_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            LOG.exception("Backend setup failed: %s", method)
        manifest["methods"][method] = method_meta
        write_json(output / "manifest.json", manifest)
        for path in files:
            relative = path.relative_to(root)
            folder = method_dir / _folder_name(relative)
            folder.mkdir()
            if "setup_error" in method_meta:
                row = {
                    "status": "error",
                    "reason": method_meta["setup_error"],
                    "match_seconds": None,
                }
            else:
                row = _process(backend, path, target, target_scale, mask, config, folder)
            row.update(
                image=relative.as_posix(),
                method=method,
                artifacts=folder.relative_to(output).as_posix(),
            )
            write_json(folder / "result.json", row)
            records.append(row)
            # Append a durable record without rewriting the entire experiment every pair.
            with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            LOG.info("%s | %s | %s %s", method, relative, row["status"], row["reason"])
        _write_tables(records, output, config.methods)
        del backend
        # Release cached allocator blocks before loading the next model on a small GPU.
        import gc
        import sys

        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()
    summary = _write_tables(records, output, config.methods)
    comparison_plots(records, output / "plots", config.methods)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)
    return summary
