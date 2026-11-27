import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from coastsnap_align.backends.base import Matches
from coastsnap_align.cli import config_from_args, main, parser
from coastsnap_align.config import RunConfig
from coastsnap_align.images import polygon_mask, read_mask
from coastsnap_align.pipeline import discover, run


class SyntheticBackend:
    """Known source→target translation; deliberately includes masked/out-of-bounds matches."""

    def __init__(self, name, config):
        self.empty = name == "empty"

    def prepare_target(self, image, mask):
        self.shape = image.shape

    def synchronize(self):
        pass

    def match(self, image):
        if self.empty:
            return Matches(np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0, 0)
        grid = np.array([[x, y] for x in range(10, 80, 10) for y in range(10, 60, 10)], float)
        src = np.vstack([grid, [[10, 10], [20, 20], [np.nan, 1]]])
        dst = np.vstack([grid + [5, 3], [[1000, 1000], [2, 2], [1, 1]]])
        return Matches(src, dst, np.ones(len(src)), len(src), len(dst))


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(8)
    source = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
    target = np.zeros_like(source)
    target[3:, 5:] = source[:-3, :-5]
    Image.fromarray(source).save(images / "source.png")
    Image.fromarray(target).save(tmp_path / "target.png")
    mask = np.full((80, 100), 255, np.uint8)
    mask[:8] = 0
    Image.fromarray(mask).save(tmp_path / "mask.png")
    return RunConfig(
        target=str(tmp_path / "target.png"),
        images=str(images),
        mask=str(tmp_path / "mask.png"),
        output=str(tmp_path / "results"),
        methods=("synthetic", "empty"),
        device="cpu",
    )


def test_full_batch_retains_failures_and_writes_plots(dataset):
    root = Path(dataset.images)
    (root / "broken.jpg").write_text("corrupt", encoding="utf-8")
    summary = run(dataset, backend_factory=SyntheticBackend)
    output = Path(dataset.output)
    assert summary["synthetic"]["accepted"] == 1
    assert summary["synthetic"]["error"] == 1
    assert summary["synthetic"]["acceptance_rate"] == 0.5
    assert summary["empty"]["rejected"] == 1
    assert summary["empty"]["error"] == 1
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    row = next(r for r in records if r["status"] == "accepted")
    folder = output / row["artifacts"]
    assert row["matches"] == 35  # Masked and invalid predictions were removed.
    with np.load(folder / "matches.npz") as data:
        np.testing.assert_allclose(data["target"] - data["source"], np.tile([5, 3], (35, 1)))
    aligned = np.asarray(Image.open(folder / "aligned.png"))
    target = np.asarray(Image.open(dataset.target))
    np.testing.assert_array_equal(aligned[3:, 5:], target[3:, 5:])
    for name in ("outcomes.png", "metrics.png", "paired.png"):
        with Image.open(output / "plots" / name) as image:
            assert min(image.size) > 300
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["completed_utc"]
    assert all(len(item["sha256"]) == 64 for item in manifest["images"])
    with pytest.raises(ValueError, match="new or empty"):
        run(dataset, backend_factory=SyntheticBackend)


def test_setup_failure_does_not_disappear_from_denominator(dataset):
    def fail(name, config):
        raise RuntimeError("model not installed")

    summary = run(dataset, backend_factory=fail)
    assert all(value["error"] == value["attempted"] == 1 for value in summary.values())
    assert all(value["acceptance_rate"] == 0 for value in summary.values())


def test_recursive_names_do_not_collide(dataset):
    root = Path(dataset.images)
    (root / "nested").mkdir()
    (root / "nested" / "source.png").write_bytes((root / "source.png").read_bytes())
    dataset.recursive = True
    dataset.methods = ("synthetic",)
    dataset.save_diagnostics = False
    summary = run(dataset, backend_factory=SyntheticBackend)
    assert summary["synthetic"]["accepted"] == 2
    assert len(list((Path(dataset.output) / "synthetic").glob("*/aligned.png"))) == 2


def test_quality_rejection_retains_candidate_but_never_labels_it_aligned(dataset):
    dataset.min_inliers = 100
    dataset.methods = ("synthetic",)
    summary = run(dataset, backend_factory=SyntheticBackend)
    assert summary["synthetic"]["rejected"] == 1
    assert not list(Path(dataset.output).glob("**/aligned.png"))
    assert list(Path(dataset.output).glob("**/rejected_candidate.png"))


def test_config_paths_are_relative_to_config_and_cli_overrides(tmp_path):
    folder = tmp_path / "configuration"
    folder.mkdir()
    path = folder / "run.json"
    path.write_text(
        json.dumps(
            {
                "target": "t.jpg",
                "images": "photos",
                "mask": "mask.png",
                "output": "out",
                "max_keypoints": 32,
            }
        ),
        encoding="utf-8",
    )
    args = parser().parse_args(
        ["run", "--config", str(path), "--max-keypoints", "100", "--no-warmup"]
    )
    config = config_from_args(args)
    assert config.target == str((folder / "t.jpg").resolve())
    assert config.max_keypoints == 100
    assert config.warmup is False


def test_polygon_inclusion_exclusion_and_mask_size(dataset, tmp_path):
    path = tmp_path / "polygons.json"
    path.write_text(
        json.dumps(
            {
                "include": [[[10, 10], [80, 10], [80, 60], [10, 60]]],
                "exclude": [[[20, 20], [30, 20], [30, 30], [20, 30]]],
            }
        )
    )
    output = tmp_path / "polygon_mask.png"
    polygon_mask(Path(dataset.target), path, output)
    mask = read_mask(output, (80, 100))
    assert mask[15, 15] and not mask[25, 25] and not mask[0, 0]
    with pytest.raises(ValueError, match="shape"):
        read_mask(output, (10, 10))
    assert output.with_name("polygon_mask_preview.png").exists()


def test_output_inside_inputs_is_rejected(dataset):
    dataset.output = str(Path(dataset.images) / "results")
    with pytest.raises(ValueError, match="non-nested"):
        discover(dataset)


def test_original_resolution_transform_when_source_is_twice_target_size(dataset):
    source = Path(dataset.images) / "source.png"
    with Image.open(source) as image:
        image.resize((200, 160), Image.Resampling.NEAREST).save(source)
    dataset.max_image_size = 100
    dataset.methods = ("synthetic",)
    dataset.save_diagnostics = False
    summary = run(dataset, backend_factory=SyntheticBackend)
    assert summary["synthetic"]["accepted"] == 1
    row = json.loads((Path(dataset.output) / "results.jsonl").read_text())
    matrix = np.array(row["source_to_target"])
    matrix /= matrix[2, 2]
    expected = np.array([[0.5, 0, 4.75], [0, 0.5, 2.75], [0, 0, 1]])
    np.testing.assert_allclose(matrix, expected, atol=1e-5)


def test_cli_help_methods_and_config_errors(capsys):
    assert main(["methods"]) == 0
    assert "lightglue" in capsys.readouterr().out
    assert main(["run", "--target", "missing"]) == 2
