# CoastSnap image alignment

An extensible Python package for registering a directory of CoastSnap photographs
to one target image, using a target-space mask to select stable scene features.
The first two backends are **SuperPoint + LightGlue** and **SuperPoint + FlexGlue**.
Each uses the same preprocessing, masked target keypoints, RANSAC transform model,
quality thresholds, metrics and reports. Inference runs sequentially on one NVIDIA
GPU, so models do not compete for GPU memory. CPU execution is also supported.

## Install on the processing machine

Use Python 3.10 or newer (3.12 recommended), Git, and an NVIDIA driver compatible
with your PyTorch CUDA installation. Linux and Windows are supported.

```bash
git clone <your-repository-url>
cd CoastSnap-Image-Alignment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install **torch and torchvision together** using the command selected for your
machine at [PyTorch's official installer](https://pytorch.org/get-started/locally/).
Select a CUDA build for the NVIDIA machine. Then install this package:

```bash
python -m pip install -e '.[matchers]'
coastsnap-align doctor
```

Check that `cuda_available` is `true`. First use downloads pretrained detector
and matcher weights into PyTorch's cache; the machine needs internet access and
enough free disk space. Set `TORCH_HOME` to a persistent location if needed. The
two upstream source dependencies are pinned to Git commits in `pyproject.toml`;
the run manifest records installed versions and available Git provenance. Core
dependencies use version ranges: retain `pip freeze` with a final benchmark to
recreate its entire environment.

For development or report processing without neural inference:

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

## Prepare the images and mask

```text
data/
  target.jpg
  target_mask.png
  images/
    photo_001.jpg
    photo_002.jpg
```

The mask is a single-channel PNG with the **same dimensions as the EXIF-oriented
target**. Values above 127 include pixels; values 127 or below exclude them.
Choose stable structures, rocks, fixed markers or other persistent features.
Exclude sky, sea, swash, mobile sand boundaries and other changing regions.
The target and source images are EXIF-oriented consistently on reading; create
your mask against the upright displayed target. Mask EXIF orientation is not
applied, and masks are never silently stretched to fit a mismatched target.
Use RGB/8-bit image inputs; higher-depth inputs are converted to RGB/8-bit.
Common JPEG, PNG, TIFF, BMP and WebP files are discovered by extension.

You can paint a binary mask in any image editor or define polygons in JSON.
Copy and edit [the polygon example](examples/mask_polygons.json) to your own image
dimensions. Coordinates are `[x, y]` in the upright original target, with `(0,0)`
at the center of the top-left pixel. Multiple inclusion polygons are combined;
exclusion polygons are subtracted. If inclusion polygons are absent, start with
the full image and subtract exclusions.

```bash
coastsnap-align mask --target data/target.jpg --polygons examples/mask_polygons.json --output data/target_mask.png
```

Inspect `data/target_mask_preview.png` before running. Excluded regions are dark.
The example polygons are illustrative coordinates, not a mask for arbitrary
CoastSnap images.

The mask constrains which target keypoints may participate **before matching**.
Every output correspondence is also checked against the original-resolution
mask before geometric fitting. FlexGlue can predict new target locations, so
this final check is essential. Masking controls correspondence eligibility;
neural descriptors and attention can still use surrounding image context. The
photos are not blacked out, avoiding artificial edges at the mask boundary.
No source mask is inferred from target pixel coordinates, because the images
are not yet aligned.

## Run the comparison

```bash
coastsnap-align run --target data/target.jpg --images data/images --mask data/target_mask.png --output results/comparison-001 --methods lightglue flexglue --device cuda
```

Or edit [examples/run.json](examples/run.json), then run:

```bash
coastsnap-align run --config examples/run.json
```

Paths inside JSON are relative to the JSON file. Command-line arguments override
JSON settings and their paths are relative to the current working directory.
Run `coastsnap-align run --help` for available options. Useful examples:

```bash
# Include subdirectories; source relative paths remain distinct in the reports.
coastsnap-align run --config examples/run.json --recursive
# Lower memory use, or run a CPU diagnostic (use a new output for every run).
coastsnap-align run --config examples/run.json --max-image-size 1024 --max-keypoints 2048 --output results/smaller
coastsnap-align run --config examples/run.json --device cpu --output results/cpu
# Compare sparse-to-sparse matching instead of FlexGlue's bidirectional predictions.
coastsnap-align run --config examples/run.json --flex-sparse --output results/sparse
# Try another shared geometric model.
coastsnap-align run --config examples/run.json --model affine --output results/affine
```

The output directory must be new or empty and separate from the input directory.
The target and mask are excluded if they are present among source images.
This prevents accidental overwrites or reprocessing previous aligned products.
Exit codes: `0` means every selected method accepted at least one image; `1`
means at least one method accepted none (inspect the retained results); `2`
means a configuration, input or fatal I/O error. Individual rejected/error pairs
are always reported, even when the command exits successfully overall.

## Outputs and metrics

```text
results/comparison-001/
  manifest.json                  configuration, input hashes, environment, setup times
  results.csv                    one row per attempted method/image combination
  results.jsonl                  same records, including transforms and backend metadata
  summary.json                   counts, acceptance rates and aggregate statistics
  target_mask.png
  target_mask_preview.png
  plots/
    outcomes.png                 accepted / rejected / error counts
    metrics.png                  distributions on accepted images
    paired.png                   same-image comparison when exactly two methods are selected
  lightglue/<image-id>/           same structure under flexglue/
    result.json
    matches.npz                  original-resolution points, scores, fit, residuals, inliers
    aligned.png                  accepted transform, at original target resolution
    valid_pixels.png             valid warped source support; black borders are not observations
    matches.png                  inlier/outlier diagnostic
    overlay.png                  target/aligned blend; excluded target areas darkened
```

Quality-rejected fits use `rejected_candidate.png` instead of `aligned.png`.
Pairs rejected before a fit retain available masked matches and their reason.
Unexpected exceptions retain a traceback; backend setup failures generate an
error row for every source image. JSONL and individual results are checkpointed
after each pair; CSV and summaries are updated after each method. An interrupted
run can be inspected, but automatic resumption is not
implemented: use a fresh output directory to rerun.

All stored points and geometric thresholds refer to **original oriented image
pixels**, regardless of inference resizing. Matrices map **source → target**
pixel centers. Their arbitrary homogeneous scale is normalized by matrix norm;
the bottom-right element need not be one. FlexGlue's corner-origin coordinates
are converted to the same pixel-center convention used by LightGlue and OpenCV.

| Statistic | Interpretation |
| --- | --- |
| `raw_matches`, `matches` | Before and after original-mask, bounds and finite-value checks |
| `inliers`, `inlier_ratio` | Robust-fit support among valid masked matches |
| `inlier_error_*_px` | Forward source→target residuals of fitted inliers, in original target pixels |
| `all_match_error_median_px` | Forward residual including outliers |
| `inlier_reverse_error_median_source_px` | Inverse transform residual in original source pixels |
| `inlier_symmetric_error_median_normalized` | RMS forward/reverse residual after dividing each by its own image diagonal |
| `mask_overlap_fraction` | Fraction of included target pixels fully covered by warped source support |
| `inlier_mask_coverage_fraction` | Included-mask area inside the inlier target convex hull, divided by mask area |
| `ncc_before`, `ncc_after`, `ncc_change` | Grayscale normalized correlation before/after, measured on identical stable-mask pixels valid in both identity and warped source |
| `ncc_common_pixels` | Support size for the NCC comparison; constant/insufficient support yields null NCC |
| `match_seconds` | Synchronized source extraction and matching, including transfer to/from device; excludes setup and warmup |
| `processing_seconds` | Per-image processing through quality evaluation, excluding diagnostic/output-image writing on completed fits |
| `pair_wall_seconds` | Per-image wall time including artifacts; excludes aggregate CSV/plot updates |

Initialization, target preparation and warmup are recorded separately in the
manifest. Both methods cache target keypoints/descriptors. FlexGlue's dense image
description is computed within matching and included in match time. One readable
source is warmed up and then processed again as a normal measured sample.
GPU work is synchronized around timing. Match confidence values are backend
specific and should not be treated as calibrated or directly interchangeable.
Seeds are recorded, but exact GPU determinism across hardware/software is not
guaranteed. Repeated trials are needed for a defensible speed comparison.

The default transform is a homography, with affine and similarity alternatives.
RANSAC failures, singular fits and projective horizons crossing the source image
are rejected. Configurable gates cover match count, inlier count/ratio, median
fit residual, overlap and spatial coverage. Defaults are starting settings,
**not validated CoastSnap acceptance criteria**. A fitted transform accepted by
these gates is labeled automated QC acceptance, not a verified registration.

These automated metrics measure **fit consistency and appearance agreement**.
They do not establish true geometric accuracy: a wrong or tightly clustered
match set may have a tiny residual. Different matchers also choose different
points, so comparing their inlier error alone can be misleading. Inspect paired
plots alongside all-attempt acceptance/error rates and visual diagnostics. For
accuracy claims, add independent withheld stable landmarks in a future
evaluation stage. The existing [validation notes](reports/CoastSnap%20alignment%20validation.md)
explain that distinction. No meter conversion, camera calibration, lens
undistortion or shoreline extraction is performed. A single homography may be
insufficient with camera translation, scene depth, changing phone intrinsics or
lens distortion; mask coverage does not guarantee accuracy on the beach.

## Add another registration method

Implement the small [backend protocol](src/coastsnap_align/backends/base.py):

1. `prepare_target(rgb, mask)` caches pair-independent target data and filters target keypoints.
2. `match(source_rgb)` returns a `Matches` object with `(N,2)` source and target points,
   `(N,)` confidence values, detector counts and optional metadata.
3. `synchronize()` waits for accelerator work, or does nothing for CPU methods.

Images are RGB uint8 arrays at working resolution; points must use pixel-center
coordinates in those arrays. The runner handles scaling, final mask checks,
robust geometry, metrics and reporting for every plugin.
Register a factory accepting `RunConfig` in your plugin's `pyproject.toml`:

```toml
[project.entry-points."coastsnap_align.backends"]
my_matcher = "my_package.backend:create_backend"
```

Install that plugin, check `coastsnap-align methods`, and select
`--methods lightglue my_matcher`. Reserved or duplicate backend names are rejected.
Plugin-specific JSON settings can live under
`"backend_options": {"my_matcher": {"setting": "value"}}`; factories read their
own entry from `config.backend_options`. Built-in adapters use the documented
shared fields and do not consume `backend_options`.
See the synthetic test backend in `tests/test_pipeline.py` for a minimal example.

## Verification and upstream projects

The CPU tests use known synthetic transforms and corrupted/empty inputs to verify
transform direction, outlier rejection, resizing coordinates, masks, padding,
quality gates, retained failures and generated artifacts without downloading
neural weights. Run a real inference smoke check on the processing machine:

```bash
python scripts/smoke_matchers.py --device cuda --output results/neural-smoke
```

This creates a textured synthetic image and translated source, runs both neural
backends and evaluates each recovered transform on a separate regular grid.
It is an integration check, not a CoastSnap performance benchmark.

Adapters follow the upstream [LightGlue](https://github.com/cvg/LightGlue) and
[FlexGlue](https://github.com/cvg/FlexGlue) inference interfaces. Detector/matcher
code and pretrained weights retain their own upstream terms, including the
[DINOv3 license accompanying FlexGlue](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md).
