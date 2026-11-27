"""Plugin boundary: RGB working images in, corresponding pixel centers out."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Matches:
    source: np.ndarray
    target: np.ndarray
    scores: np.ndarray
    source_keypoints: int
    target_keypoints: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        self.source = np.asarray(self.source, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        self.scores = np.asarray(self.scores, dtype=float)
        n = len(self.source)
        if self.source.shape != (n, 2) or self.target.shape != (n, 2) or self.scores.shape != (n,):
            raise ValueError("Backend must return source/target (N,2) and scores (N,)")


class Backend(Protocol):
    """Instances process one target and multiple sources sequentially.

    Coordinates use (x,y), with the first pixel center at (0,0). Target keypoints
    must be filtered before matching; the runner also checks every returned pair.
    prepare_target caches only pair-independent state. synchronize must wait for
    pending accelerator work so host timing includes actual inference.
    """

    def prepare_target(self, image: np.ndarray, mask: np.ndarray) -> None: ...
    def match(self, image: np.ndarray) -> Matches: ...
    def synchronize(self) -> None: ...
