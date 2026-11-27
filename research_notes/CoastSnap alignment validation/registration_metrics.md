# Registration metrics

## What should be measured when ground truth is available?

### Takeaway
Use independently annotated, spatially distributed correspondences as the primary measure of geometric accuracy. Report the full error distribution and task-relevant regional errors; image similarity and residuals on the points used to fit the transform are not substitutes for held-out target registration error (TRE).

### Cited Findings
- For paired landmarks, TRE is the Euclidean distance between the transformed source landmark and its corresponding reference landmark. Manual correspondences are widely treated as a gold standard, although annotations themselves have uncertainty and sparse landmarks do not fully characterize dense/deformable mappings. [Survey of image registration evaluation metrics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11730935/)
- Distinguish fit error from predictive error: residuals at control points used to estimate a transformation can be small by construction, especially for flexible models. Independent test points better estimate accuracy at unseen locations. [Computer Vision Online: Image Registration](https://computervisiononline.com/explains/image-registration/)
- A large-landmark validation study found the uncertainty of average spatial-error estimates decreases with the square root of the number of landmark pairs, and that too few validation points can misrepresent comparative performance. The study estimated sample sizes from observed error variance rather than assuming a universal number. [A framework for evaluation of deformable image registration spatial accuracy](https://pubmed.ncbi.nlm.nih.gov/19265208/)
- In remote sensing geometric evaluation, USGS JACIE guidance recommends independent higher-accuracy reference data; it notes ASPRS guidance that reference data be about three times more accurate than the data under assessment. It recommends summarizing residual mean and standard deviation, examining plots for bias or higher-order distortion, and reporting a CE90 estimate and error distribution for positional accuracy. [USGS JACIE best practices](https://pubs.usgs.gov/publication/ofr20241023/full)
- Point-based and boundary-based metrics answer different questions. Dice/IoU report region overlap; average surface distance and Hausdorff distance report contour discrepancy. Hausdorff is sensitive to a single outlier, so percentile Hausdorff (e.g., HD95) is often used as a robust tail summary. [Image registration evaluation metrics review](https://doi.org/10.1016/j.imu.2024.101540); [medical image registration evaluation chapter](https://www.diva-portal.org/smash/get/diva2%3A1694512/FULLTEXT01.pdf)
- Fiducial registration error is not the same as target registration error: a fit can have low error at the landmarks used to estimate it while error at independent targets remains larger. [Medical Image Analysis 2008 study](https://citeseerx.ist.psu.edu/document?doi=8fea7450a0b2fa33b39d35a9649109b1dd39aaf6&repid=rep1&type=pdf)

### Inferences
- For a CoastSnap pair, create a held-out set of manually marked stable scene points (for example, permanent structures or fixed terrain features) distributed across the image, including the regions where alignment quality matters. Exclude moving water, foam, clouds, people, vegetation moving in wind, and the current shoreline from the stable-point ground truth; these are scene changes, not reliable correspondences.
- Report pixel TRE (and optionally normalize horizontal/vertical error by image width/height for comparison across resolutions). If the aligned image is intended to support shoreline positions in physical coordinates, also convert point error into the relevant ground-distance units using the camera/site mapping; pixel error alone is not an operational tolerance.
- Use median, mean or RMSE, 90th/95th percentile, maximum, and a spatial error map/vector plot. Mean alone can hide occasional catastrophic failures; maximum alone can be dominated by a misannotated point. Include a per-image-pair summary so pairs with many annotations do not dominate the aggregate.
- Have a second annotator or repeated annotations for a subset and retain annotation uncertainty; a tolerance band / hit-rate metric can be more interpretable than pretending landmark positions are exact. Set a hit threshold from the use case and annotation repeatability rather than adopting a medical-imaging threshold.

### Gaps
- No CoastSnap-specific publication was located in this subtask that prescribes an accepted point count or pixel tolerance for landmark TRE. The threshold must be tied to image resolution, site calibration, and the shoreline analysis use case.

## Which metrics can help when exact ground truth is unavailable?

### Takeaway
No-ground-truth metrics are quality-control signals, not direct measures of alignment accuracy. They are useful to detect gross failures and compare before/after results under controlled conditions, but should be calibrated against ground-truth samples before being interpreted as accuracy.

### Cited Findings
- An experimental study demonstrated that image similarity, tissue overlap, and inverse-consistency surrogates could all look good for intentionally inaccurate transformations; the authors conclude these are not valid evidence of registration accuracy when used alone. In that study, overlap of sufficiently small localized regions was more discriminative than global overlaps. [Rohlfing, “Image Similarity and Tissue Overlaps as Surrogates … Widely Used but Unreliable”](https://ieeexplore.ieee.org/abstract/document/5977031/)
- A follow-up analysis states that increased similarity after alignment does not prove the alignment is correct; image-based optimization can settle in local optima and similarity is uninformative about the magnitude of correspondence errors. [“Nonparametric Local Smoothing” is not image registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC3740790/)
- Common image metrics have distinct assumptions: MSE/PSNR penalize pixelwise intensity differences; NCC measures normalized intensity correlation; MI/NMI measure statistical dependence and are commonly used with different modalities. Their objective values depend on image characteristics and are difficult to interpret as a universal accuracy scale. [Czolbe et al., Medical Image Analysis](https://backend.orbit.dtu.dk/ws/files/320834381/1_s2.0_S1361841523000907_main.pdf); [DIR quality assurance review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424976/)
- PSNR is sensitive to illumination changes, and the available review advises it cannot be relied upon where images have lighting variation. [Image registration similarity metrics discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6427184/)
- Inverse consistency (warp A→B and B→A, then assess round-trip discrepancy) can reveal direction-dependent behavior, but does not establish correctness by itself; a consistently wrong transformation can be inverse-consistent. [Rohlfing 2012](https://ieeexplore.ieee.org/abstract/document/5977031/); [inverse-consistency discussion](https://en.wikipedia.org/wiki/Inverse_consistency)
- Precision/repeatability without ground truth can be assessed by repeated registrations under small perturbations, alternative initializations, or swapped source/reference direction. Such tests reveal convergence failures and instability but not systematic bias shared by all runs. [Assessing intrinsic precision without ground truth](https://pmc.ncbi.nlm.nih.gov/articles/PMC2881924/)

### Inferences
- For CoastSnap, calculate a small set of masked metrics on stable, visible land regions: NCC or SSIM on grayscale/gradient images, plus edge-map alignment. Exclude the sea and sky from whole-image metrics or report them separately because waves, water level, glare, clouds, and shadows can change independently of camera alignment.
- Do not make MI, NCC, SSIM, MSE, or PSNR a single pass/fail score. Show their change from the unaligned baseline, use the same masks/resampling for every method, and empirically chart their relationship to held-out TRE on a labeled validation subset.
- Transformation smoothness, plausible warp magnitude, Jacobian/no-folding checks for deformable transforms, inverse consistency, and repeated-run stability are useful guardrails. None alone answers whether a physical scene point lands in the correct place.

### Gaps
- The sources reviewed do not establish that SSIM, MI, or any other image-similarity metric has a calibrated relationship to geometric error for CoastSnap images. That relationship would require a CoastSnap-specific calibration study.

## How should reprojection, overlap, edge, and uncertainty metrics be combined?

### Takeaway
Use each score only for the property it measures: held-out TRE for pointwise geometry; symmetric transfer/reprojection error for matched-feature consistency; overlap and contour distances for task-specific regions; and photometric similarity for appearance consistency. A compact dashboard is more defensible than collapsing unlike metrics into one unvalidated score.

### Cited Findings
- Reprojection error measures the distance between measured reference points and the points predicted by a fitted transformation. Symmetric transfer error checks transfer in both image directions. They are standard geometric-consistency diagnostics for homography/feature matching, but an inlier-only score is conditional on the matching and fitting process. [Hartley & Zisserman, Multiple View Geometry](https://www.changjiangcai.com/files/text-books/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf); [remote-sensing feature-matching study](https://www.mdpi.com/2072-4292/10/5/747)
- USGS JACIE describes automated conjugate-point evaluation using local image correlation and statistical analysis, and recommends plotting residual vectors to reveal systematic offsets or higher-order distortion in addition to scalar summaries. [USGS JACIE best practices](https://pubs.usgs.gov/publication/ofr20241023/full)
- Dice and IoU quantify area overlap; they can conceal local boundary errors, especially for large masks. Average surface distance and HD95 make boundary displacement more explicit, while the maximum Hausdorff distance is vulnerable to outliers. [DIR evaluation metrics review](https://doi.org/10.1016/j.imu.2024.101540); [medical image registration evaluation chapter](https://www.diva-portal.org/smash/get/diva2%3A1694512/FULLTEXT01.pdf)
- Landmark annotations have observer uncertainty; an uncertainty-aware annotation protocol proposes representing spatial uncertainty and accounting for repeated annotators rather than treating each landmark as a noiseless point. [Uncertainty-Aware Annotation Protocol](https://arxiv.org/abs/2104.01217)
- Registration error maps can be learned from synthetic or annotated examples, but generalization depends on training data and requires validation against true landmark errors. [Error estimation of deformable image registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC5943620/); [DIR quality assurance review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424976/)

### Inferences
- If the pipeline estimates a homography, report held-out TRE as the headline measure and symmetric transfer error as an internal geometric diagnostic. Include number of tentative matches, inlier count/ratio, and spatial coverage so many clustered matches do not create false confidence. The inlier fit residual should not be presented as independent accuracy.
- For featureless / contour-led methods, annotate stable line or contour segments and measure symmetric point-to-curve distances or average surface distance, plus a robust tail (95th percentile). A nearest-contour score needs a constrained matching region; otherwise unrelated nearby edges can produce artificially low values.
- A practical validation panel could include: (1) held-out landmark TRE median and 90th/95th percentile; (2) residual-vector plot and spatial-bin TRE; (3) matched-feature symmetric transfer error and inlier spatial coverage; (4) stable-region edge distance or SSIM/NCC as a secondary appearance check; (5) success/failure rate and repeatability across image pairs. Report metrics per site/camera and stratify by conditions such as light, tide, weather, and seasonal appearance.
- Avoid combining pixel TRE, Dice, NCC, and reprojection error into a single weighted score unless weights and the score are validated against downstream CoastSnap shoreline extraction performance.

### Gaps
- There is no universal metric threshold for “good alignment”; acceptable error depends on intended shoreline measurement accuracy, image scale, and how the transformation is used. No examined source recommends a universal threshold applicable to CoastSnap.
