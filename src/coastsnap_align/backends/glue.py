"""Adapters for the pinned cvg LightGlue and FlexGlue inference APIs."""

import numpy as np

from ..images import points_in_mask
from .base import Matches


class GlueBackend:
    def __init__(self, name, config):
        try:
            import torch
            from lightglue import LightGlue, SuperPoint
        except ImportError as exc:
            raise RuntimeError("Install CUDA PyTorch, then pip install -e '.[matchers]'") from exc
        self.torch = torch
        self.name = name
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; check PyTorch/driver or use --device cpu"
            )
        torch.manual_seed(config.seed)
        self.offset = 0.5 if name == "flexglue" else 0.0
        self.detector = (
            SuperPoint(
                max_num_keypoints=config.max_keypoints,
                origin="corner" if self.offset else "center",
            )
            .eval()
            .to(self.device)
        )
        if name == "lightglue":
            self.matcher = LightGlue(features="superpoint", filter_threshold=config.match_threshold)
        else:
            try:
                from flexglue import FlexGlue
            except ImportError as exc:
                raise RuntimeError("FlexGlue missing; pip install -e '.[matchers]'") from exc
            self.matcher = FlexGlue(
                filter_threshold=config.match_threshold,
                describe_resize=config.flex_descriptor_size,
                mp="float16" if self.device.type == "cuda" else None,
            )
        self.matcher = self.matcher.eval().to(self.device)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _tensor(self, image):
        return (
            self.torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(
                device=self.device, dtype=self.torch.float32
            )
            / 255.0
        )

    @staticmethod
    def _numpy(tensor):
        return tensor.detach().float().cpu().numpy()

    def prepare_target(self, image, mask):
        with self.torch.inference_mode():
            self.target_image = self._tensor(image)
            features = self.detector.extract(self.target_image, resize=None)
            keep = points_in_mask(self._numpy(features["keypoints"][0]) - self.offset, mask)
            keep = self.torch.as_tensor(keep, device=self.device)
            # Explicit fields: image_size must never be mistaken for a per-point field.
            for key in ("keypoints", "descriptors", "keypoint_scores"):
                if key in features:
                    features[key] = features[key][:, keep]
            self.target_features = features
            self.target_count = int(keep.sum().item())

    def match(self, image):
        with self.torch.inference_mode():
            source_image = self._tensor(image)
            features = self.detector.extract(source_image, resize=None)
            source_count = features["keypoints"].shape[1]
            if self.target_count == 0 or source_count == 0:
                return Matches(
                    np.empty((0, 2)), np.empty((0, 2)), np.empty(0), source_count, self.target_count
                )
            metadata = {}
            if self.name == "lightglue":
                prediction = self.matcher({"image0": features, "image1": self.target_features})
                indices = prediction["matches"][0]
                src = features["keypoints"][0][indices[:, 0]]
                dst = self.target_features["keypoints"][0][indices[:, 1]]
                scores = prediction["scores"][0]
            else:
                from flexglue.utils import pack_matches

                # Fresh dictionaries: pair-specific transformer outputs must never be cached.
                first = {**features, "image": source_image[None]}
                second = {**self.target_features, "image": self.target_image[None]}
                prediction = pack_matches(
                    self.matcher.match(first, second, sparse=self.config.flex_sparse)
                )
                indices = prediction["matches"][0]
                src = prediction["keypoints0"][0][indices[:, 0]]
                dst = prediction["keypoints1"][0][indices[:, 1]]
                scores = prediction["scores"][0]
                if "covisibility_score" in prediction:
                    metadata["covisibility_score"] = float(
                        prediction["covisibility_score"].reshape(-1)[0].item()
                    )
            return Matches(
                self._numpy(src) - self.offset,
                self._numpy(dst) - self.offset,
                self._numpy(scores),
                source_count,
                self.target_count,
                metadata,
            )
