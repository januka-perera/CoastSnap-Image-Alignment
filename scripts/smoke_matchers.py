"""Opt-in real-model integration check. Needs .[matchers], weights, and optionally CUDA."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from coastsnap_align.config import RunConfig
from coastsnap_align.geometry import project
from coastsnap_align.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("Choose an output path that does not already exist")
    inputs = root / "inputs"
    sources = inputs / "sources"
    sources.mkdir(parents=True)
    rng = np.random.default_rng(19)
    target = rng.integers(40, 210, (480, 640, 3), dtype=np.uint8)
    target = cv2.GaussianBlur(target, (3, 3), 0)
    for index in range(180):
        center = tuple(rng.integers([25, 25], [615, 455]).tolist())
        color = tuple(rng.integers(0, 255, 3).tolist())
        cv2.circle(target, center, int(rng.integers(3, 15)), color, -1)
        if index % 6 == 0:
            cv2.putText(target, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # The source is displaced right/down, so source→target translates left/up.
    source = cv2.warpAffine(target, np.float32([[1, 0, 17], [0, 1, 11]]), (640, 480))
    Image.fromarray(target).save(inputs / "target.png")
    Image.fromarray(source).save(sources / "source.png")
    mask = np.zeros((480, 640), np.uint8)
    mask[25:-25, 25:-25] = 255
    Image.fromarray(mask).save(inputs / "mask.png")
    config = RunConfig(
        target=str(inputs / "target.png"),
        images=str(sources),
        mask=str(inputs / "mask.png"),
        output=str(root / "comparison"),
        device=args.device,
        max_image_size=640,
        max_keypoints=1024,
    )
    summary = run(config)
    rows = [
        json.loads(line)
        for line in (root / "comparison" / "results.jsonl").read_text().splitlines()
    ]
    evaluation = {}
    checks = np.array([[x, y] for y in range(60, 420, 60) for x in range(60, 580, 60)])
    for row in rows:
        if row["status"] != "accepted":
            evaluation[row["method"]] = {"passed": False, "reason": row["reason"]}
            continue
        errors = np.linalg.norm(
            project(checks, np.array(row["source_to_target"])) - (checks - [17, 11]), axis=1
        )
        evaluation[row["method"]] = {
            "passed": bool(np.max(errors) < 3),
            "grid_median_error_px": float(np.median(errors)),
            "grid_max_error_px": float(np.max(errors)),
        }
    (root / "grid_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "known_transform_check": evaluation}, indent=2))
    return 0 if all(item["passed"] for item in evaluation.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
