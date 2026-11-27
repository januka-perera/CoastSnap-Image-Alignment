"""Headless numerical plots and compact per-pair visual diagnostics."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_diagnostics(folder, source, target, aligned, valid, points0, points1, inliers, mask):
    plt = _pyplot()

    # Render thumbnails rather than two full-resolution photos in the plotting canvas.
    def thumbnail(image):
        thumb = Image.fromarray(image)
        thumb.thumbnail((1000, 750))
        return np.asarray(thumb)

    a, b = thumbnail(source), thumbnail(target)
    sa = np.array([a.shape[1] / source.shape[1], a.shape[0] / source.shape[0]])
    sb = np.array([b.shape[1] / target.shape[1], b.shape[0] / target.shape[0]])
    canvas = np.zeros((max(a.shape[0], b.shape[0]), a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] :] = b
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)
    indices = np.linspace(0, len(points0) - 1, min(200, len(points0)), dtype=int)
    for i in indices:
        p, q = (points0[i] + 0.5) * sa - 0.5, (points1[i] + 0.5) * sb - 0.5
        ax.plot(
            [p[0], q[0] + a.shape[1]],
            [p[1], q[1]],
            color="lime" if inliers[i] else "red",
            alpha=0.5,
            linewidth=0.6,
        )
    ax.set_title("Source → target: green = fitted inlier, red = outlier (up to 200 shown)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "matches.png", dpi=120)
    plt.close(fig)
    overlay = target.copy()
    overlay[valid] = ((target[valid].astype(float) + aligned[valid]) / 2).astype(np.uint8)
    overlay[~mask] //= 3
    Image.fromarray(thumbnail(overlay)).save(folder / "overlay.png")


def comparison_plots(records, output: Path, methods):
    plt = _pyplot()
    output.mkdir(exist_ok=True)
    states = ("accepted", "rejected", "error")
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(methods), dtype=int)
    for state, color in zip(states, ("#408f64", "#d49b38", "#bb4d4d")):
        counts = [
            sum(r["method"] == method and r["status"] == state for r in records)
            for method in methods
        ]
        ax.bar(methods, counts, bottom=bottom, label=state, color=color)
        bottom += counts
    ax.set_ylabel("Attempted images")
    ax.set_title("Automated QC outcome (acceptance is not verified accuracy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=150)
    plt.close(fig)

    panels = [
        ("inlier_error_median_px", "Inlier fit error (target px)"),
        ("inlier_ratio", "Inlier ratio"),
        ("mask_overlap_fraction", "Target mask overlap"),
        ("match_seconds", "Matching time (seconds)"),
        ("ncc_change", "NCC change on common pixels"),
        ("inlier_mask_coverage_fraction", "Inlier spatial coverage"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        groups, names = [], []
        for method in methods:
            values = [
                r[key]
                for r in records
                if r["method"] == method and r["status"] == "accepted" and r.get(key) is not None
            ]
            if values:
                groups.append(values)
                names.append(method)
        if groups:
            ax.boxplot(groups, tick_labels=names, showmeans=True)
        else:
            ax.text(0.5, 0.5, "No accepted values", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Accepted images only — compare alongside rejection/error rates")
    fig.tight_layout()
    fig.savefig(output / "metrics.png", dpi=150)
    plt.close(fig)

    # Each dot is the same input image accepted by both methods.
    if len(methods) == 2:
        first = {
            r["image"]: r
            for r in records
            if r["method"] == methods[0] and r["status"] == "accepted"
        }
        second = {
            r["image"]: r
            for r in records
            if r["method"] == methods[1] and r["status"] == "accepted"
        }
        common = sorted(first.keys() & second.keys())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, label in zip(
            axes,
            ("inlier_error_median_px", "match_seconds"),
            ("Median fit error (px)", "Matching time (s)"),
        ):
            x, y = [first[n][key] for n in common], [second[n][key] for n in common]
            if common:
                ax.scatter(x, y, alpha=0.7)
                limit = max(max(x), max(y), 1e-6) * 1.05
                ax.plot([0, limit], [0, limit], "--", color="gray")
            ax.set_xlabel(methods[0])
            ax.set_ylabel(methods[1])
            ax.set_title(f"{label}; {len(common)} paired images")
        fig.tight_layout()
        fig.savefig(output / "paired.png", dpi=150)
        plt.close(fig)


def summarize(records, methods):
    result = {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        accepted = [r for r in rows if r["status"] == "accepted"]
        counts = Counter(r["status"] for r in rows)
        result[method] = {
            "attempted": len(rows),
            **{state: counts[state] for state in ("accepted", "rejected", "error")},
            "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
            "accepted_median_fit_error_px": float(
                np.median([r["inlier_error_median_px"] for r in accepted])
            )
            if accepted
            else None,
            "median_match_seconds_all_completed": float(
                np.median([r["match_seconds"] for r in rows if r.get("match_seconds") is not None])
            )
            if any(r.get("match_seconds") is not None for r in rows)
            else None,
        }
    return result
