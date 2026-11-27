"""Shared settings; all geometric thresholds use original-image pixels."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class RunConfig:
    target: str = ""
    images: str = ""
    mask: str = ""
    output: str = ""
    methods: tuple[str, ...] = ("lightglue", "flexglue")
    device: str = "cuda"
    recursive: bool = False
    max_image_size: int = 1600
    max_keypoints: int = 4096
    model: str = "homography"
    ransac_threshold: float = 3.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 10000
    min_matches: int = 12
    min_inliers: int = 10
    min_inlier_ratio: float = 0.2
    max_median_error: float = 3.0
    min_overlap: float = 0.1
    min_coverage: float = 0.01
    seed: int = 42
    warmup: bool = True
    save_aligned: bool = True
    save_diagnostics: bool = True
    flex_sparse: bool = False
    flex_descriptor_size: int = 512
    match_threshold: float = 0.1
    backend_options: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        data = dict(data)
        if "methods" in data:
            if not isinstance(data["methods"], (list, tuple)):
                raise ValueError("methods must be a list of backend names")
            data["methods"] = tuple(data["methods"])
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self):
        for name in ("target", "images", "mask", "output"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        if not self.methods or any(not isinstance(m, str) or not m for m in self.methods):
            raise ValueError("At least one backend name is required")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Method names must be unique")
        if self.model not in {"homography", "affine", "similarity"}:
            raise ValueError("model must be homography, affine or similarity")
        if (
            self.device != "cpu"
            and self.device != "cuda"
            and not (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        for name in (
            "max_image_size",
            "max_keypoints",
            "ransac_iterations",
            "min_matches",
            "min_inliers",
            "flex_descriptor_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_matches < 4 or self.min_inliers < 4:
            raise ValueError("min_matches and min_inliers must be at least 4")
        if self.flex_descriptor_size % 8:
            raise ValueError("flex_descriptor_size must be divisible by 8")
        for name in ("ransac_threshold", "max_median_error"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "min_inlier_ratio",
            "min_overlap",
            "min_coverage",
            "match_threshold",
            "ransac_confidence",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be between zero and one")
        if not 0 < self.ransac_confidence < 1:
            raise ValueError("ransac_confidence must be strictly between zero and one")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be a nonnegative 32-bit signed integer")
        for name in ("recursive", "warmup", "save_aligned", "save_diagnostics", "flex_sparse"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        if not isinstance(self.backend_options, dict) or any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in self.backend_options.items()
        ):
            raise ValueError("backend_options must map method names to setting objects")

    def to_dict(self):
        return asdict(self)


def resolve_config_paths(data, base: Path):
    """Resolve file paths relative to the configuration file, not the working directory."""
    for key in {"target", "images", "mask", "output"} & data.keys():
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"{key} must be a nonempty path string")
    return {
        k: str((base / v).resolve()) if k in {"target", "images", "mask", "output"} else v
        for k, v in data.items()
    }
