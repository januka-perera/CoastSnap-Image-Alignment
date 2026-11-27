"""Command-line entry point. JSON configuration paths are relative to their file."""

import argparse
import json
import logging
from pathlib import Path

from .backends import backend_names
from .config import RunConfig, resolve_config_paths


def parser():
    result = argparse.ArgumentParser(
        prog="coastsnap-align", description="Masked CoastSnap registration"
    )
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Align a directory and compare methods")
    run.add_argument("--config", type=Path, help="JSON configuration file")
    for name in ("target", "images", "mask", "output"):
        run.add_argument(f"--{name}", help=f"{name} path")
    run.add_argument("--methods", nargs="+", help="Backend names (default: lightglue flexglue)")
    run.add_argument("--device", help="cuda (default), cuda:N, or cpu")
    run.add_argument("--model", choices=("homography", "affine", "similarity"))
    for name in (
        "max_image_size",
        "max_keypoints",
        "min_matches",
        "min_inliers",
        "seed",
        "ransac_iterations",
        "flex_descriptor_size",
    ):
        run.add_argument("--" + name.replace("_", "-"), type=int)
    for name in (
        "ransac_threshold",
        "ransac_confidence",
        "min_inlier_ratio",
        "max_median_error",
        "min_overlap",
        "min_coverage",
        "match_threshold",
    ):
        run.add_argument("--" + name.replace("_", "-"), type=float)
    for name in ("recursive", "warmup", "save_aligned", "save_diagnostics", "flex_sparse"):
        run.add_argument(
            "--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None
        )
    run.add_argument("--verbose", action="store_true")
    mask = sub.add_parser("mask", help="Create a binary mask and preview from polygon JSON")
    mask.add_argument("--target", type=Path, required=True)
    mask.add_argument("--polygons", type=Path, required=True)
    mask.add_argument("--output", type=Path, required=True)
    sub.add_parser("methods", help="List built-in and installed plugin backends")
    sub.add_parser(
        "doctor", help="Report package versions and CUDA availability without loading weights"
    )
    return result


def config_from_args(args):
    data = {}
    if args.config:
        path = args.config.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        data = resolve_config_paths(data, path.parent)
    for key, value in vars(args).items():
        if key not in {"command", "config", "verbose"} and value is not None:
            data[key] = value
    return RunConfig.from_dict(data)


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.command == "methods":
            print("\n".join(backend_names()))
        elif args.command == "doctor":
            from .pipeline import environment

            print(json.dumps(environment(), indent=2))
        elif args.command == "mask":
            from .images import polygon_mask

            if args.output.suffix.lower() != ".png":
                raise ValueError("Use a .png output for a lossless binary mask")
            if args.output.resolve() in {args.target.resolve(), args.polygons.resolve()}:
                raise ValueError("Mask output must not overwrite an input")
            polygon_mask(args.target, args.polygons, args.output)
            print(f"Saved mask and preview to {args.output.parent.resolve()}")
        else:
            from .pipeline import run

            summary = run(config_from_args(args))
            print(json.dumps(summary, indent=2))
            # A method with no accepted pairs needs attention, even if another method worked.
            return 0 if all(value["accepted"] > 0 for value in summary.values()) else 1
    except (ValueError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        return 2
    return 0
