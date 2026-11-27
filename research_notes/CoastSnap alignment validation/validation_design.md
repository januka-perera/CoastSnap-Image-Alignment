# Validation design for CoastSnap image alignment

## How should a trustworthy reference and test split be constructed?

### Takeaway
The strongest test is independent geometric ground truth: identify stable, well-distributed scene landmarks in the source and reference images, fit alignment using one subset, and score only withheld landmarks. For CoastSnap, repeat this over images from different dates, phones, conditions, and stations, and where possible validate the final rectified shoreline against contemporaneous RTK-GNSS/DGPS surveys as an end-to-end check.

### Cited Findings
- CoastSnap relies on permanent, surveyed ground-control points (GCPs) identifiable in the image, but its varied smartphone models mean camera intrinsics are approximated on the fly from those points. A CoastSnap paper reports that mapped shoreline agreement with surveys was strongest near the camera and degraded over the wider beach: cross-shore RMSE was 1.4 m nearfield and 3.9 m over nearly 1 km at North Narrabeen. [Harley et al., 2018 review, Remote Sensing](https://www.mdpi.com/2072-4292/10/11/1744)
- The CoastSnap workflow may co-register images to a rectified target/control image before bulk rectification; the registration step is distinct from shoreline extraction and georectification. [CoastSnap: A global citizen science program](https://doi.org/10.1016/J.CSR.2022.104796); [Automating shoreline extraction from beach imaging systems](https://doi.org/10.1016/j.envsoft.2026.107078)
- ASPRS states that checkpoints used to test positional accuracy must be truly independent: they must not be used as control or elsewhere in processing. Checkpoints should be distributed across the product extent or representative of tested features. The standard calls for a minimum of 30 coordinates for statistically robust positional accuracy testing when feasible. [ASPRS Positional Accuracy Standards, 2nd ed.](https://aagsmo.org/wp-content/uploads/2023/03/ASPRS_PosAcc_Edition2_MainBody.pdf)
- The USGS UAS calibration guidance describes repeated random subsets of GCPs as a way to generate calibration parameters and use the unused points as validation points; it recommends storing the check-point residuals across multiple trials (typically at least 30) when independent points are limited. [USGS Open-File Report 2023-1033](https://pubs.usgs.gov/publication/ofr20231033/full)
- A CoastSnap dataset paper describes image pairs co-registered to a target image and shoreline products evaluated against RTK-GNSS; it reports overall MAD 2.76 m (site range 1.68–5.95 m), demonstrating an end-to-end field reference approach and the need to report site-level differences. [Automating shoreline extraction from beach imaging systems](https://doi.org/10.1016/j.envsoft.2026.107078)

### Inferences
- For a registration-only benchmark, each image pair should have a set of stable correspondences (e.g. rock corners, built structures, fixed markers) annotated independently by two reviewers or adjudicated, then split spatially into fit and test subsets. Points used by RANSAC, a homography solver, or any tuning step cannot also be counted as independent accuracy checks.
- Test points should cover the full field of view, especially image edges and the beach region where shoreline positions are measured. A central cluster of points can hide rotational, scale, or perspective errors that grow toward the edges.
- Avoid random image-level train/test splits that place near-duplicate frames from one phone/date/site in both sets. Hold out entire dates, phone models, and preferably entire stations for generalization claims; also report within-station temporal holdout results separately because these answer a different operational question.
- A useful design has two ground-truth tiers: (1) repeatable image-space correspondences to isolate registration error; and (2) same-day surveyed shoreline or fixed surveyed targets, run through registration plus rectification, to capture accumulated real-world error. Coastal shoreline is dynamic, so shoreline itself is not a stable landmark for the registration-only tier.

### Gaps
- No universal CoastSnap-specific benchmark dataset or standard split for image registration was found. Available CoastSnap studies validate shoreline position or overall photogrammetry, rather than publishing a broadly standardized held-out landmark dataset for the image registration module alone.
- The cited ASPRS minimum of 30 applies to geospatial positional accuracy testing; it is a useful design target, not a demonstrated minimum for every image pair or every CoastSnap site. CoastSnap scenes may not contain 30 distinct stable, unambiguous landmarks.

## Which experimental factors and failure modes should be represented?

### Takeaway
The evaluation set should reflect real CoastSnap variation and should report results by condition, not just one pooled average. Camera position/device, image resolution and compression, illumination, weather, occlusion, and scene geometry can change the apparent image and the available stable features; accuracy also varies spatially with distance from the camera.

### Cited Findings
- CoastSnap images come from public smartphones with varying models and image processing; the CoastSnap process approximates intrinsics using surveyed GCPs. The North Narrabeen study found similar shoreline accuracy between compressed low-resolution social-media images (~0.3 MP) and direct smartphone images (>2 MP), but that finding concerns shoreline outputs at the tested station and period. [Harley et al., 2018 review](https://www.mdpi.com/2072-4292/10/11/1744)
- CoastSnap supports standard permanent stations, DIY stations where positioning is controlled to some degree, and free-roaming images where it is uncontrolled, indicating distinct acquisition regimes that should not be collapsed into one claim. [CoastSnap: A global citizen science program](https://doi.org/10.1016/J.CSR.2022.104796)
- A coastal camera autocalibration study reports automatic calibration of over 90% of images for a camera with favorable fixed features, versus around 40% for the most poorly conditioned, nearly featureless camera. It reports homography reprojection error as one quality measure. [UCalib: Cameras Autocalibration on Coastal Video Monitoring Systems](https://www.mdpi.com/2072-4292/13/14/2795)
- Recent CoastSnap shoreline validation found the largest errors at a more dynamic area and farther from the camera; at one transect, shoreline RMSD was 3.47 m compared with 1.81 m at another. [Assessing shoreline dynamics and beach use in a tropical beach](https://pmc.ncbi.nlm.nih.gov/articles/pmid/42726327/)
- ASPRS recommends investigating large residuals/blunders rather than silently discarding them, and calls for reporting min/max, mean/median, standard deviation, RMSE, and residuals at each checkpoint. [ASPRS Positional Accuracy Standards, 2nd ed.](https://aagsmo.org/wp-content/uploads/2023/03/ASPRS_PosAcc_Edition2_MainBody.pdf)

### Inferences
- Stratify the test corpus across: station/site; phone make/model or camera/lens; source image dimensions and crop/aspect ratio; social-media compression; camera displacement/rotation in cradle; lighting (sun/shade/backlight), weather/haze, wet/dry beach, vegetation and human occlusion; feature density; and near/mid/far or image-center/edge regions. Record these as metadata and publish per-stratum error and success rate.
- Include difficult but valid pairs, not only clean examples. Separately label unalignable or insufficient-feature pairs so the algorithm is assessed on whether it rejects them safely rather than returning a plausible but wrong transform.
- Report all attempted pairs in the denominator. Give successful registration coverage, failure/rejection rate, and catastrophic-misalignment rate; calculating accuracy only on successful alignments can hide reliability problems.
- Use paired comparisons on the same held-out pairs to compare pipeline variants, and retain condition strata so average gains do not conceal regressions for distant beach regions, mobile cameras, or low-feature images.

### Gaps
- The sources establish effects of spatial distance and feature availability, but do not provide a full CoastSnap factorial analysis across phone models, lighting, tide/wave conditions, and camera handling.
- A single universal pass/fail pixel or meter threshold is not specified by CoastSnap literature; the acceptable tolerance must be set from intended shoreline-change use and local ground sampling geometry.

## How should numerical results, uncertainty, and operational impact be reported?

### Takeaway
Report geometric error on withheld points in both pixels and ground units, plus bias, spread, tails, and failure rates. Then validate the downstream shoreline product against same-day field surveys, because small image-space residuals or high image similarity do not by themselves guarantee useful coastal measurements.

### Cited Findings
- ASPRS defines horizontal RMSE from the component RMSEs and recommends reporting checkpoint residuals, maximum/minimum, mean/median, standard deviation, and RMSE. It also notes that its accuracy standard does not itself specify application-specific thresholds; thresholds should follow project requirements. [ASPRS Positional Accuracy Standards, 2nd ed.](https://aagsmo.org/wp-content/uploads/2023/03/ASPRS_PosAcc_Edition2_MainBody.pdf)
- ASPRS distinguishes “tested to meet” accuracy (comparison to independent, higher-accuracy coordinates) from “produced to meet” (process/design supports a claim without independent accuracy testing). [ASPRS Positional Accuracy Standards, Addendum III (2024)](https://cnig.gouv.fr/IMG/pdf/251007_gt_cnig_quadogeo_251007_2024_asprs_positional_accuracy_standards_edition2_version2.pdf)
- CoastSnap shoreline validation has used cross-shore deviations against GPS/DGPS and reports bias-like mean deviations, standard deviation, RMSD, and proportions within a tolerance (e.g. 79% of shoreline estimates within ±3 m in one microtidal study). [Assessing shoreline dynamics and beach use in a tropical beach](https://pmc.ncbi.nlm.nih.gov/articles/pmid/42726327/)
- CoastSnap shoreline positional accuracy has also been validated against RTK-GNSS with MAD and site-specific ranges, showing the benefit of robust absolute-error summaries and site-level reporting. [Automating shoreline extraction from beach imaging systems](https://doi.org/10.1016/j.envsoft.2026.107078)
- Image similarity and overlap measures are unreliable surrogates for true registration accuracy in a medical-image registration benchmark; the authors recommend direct errors at independent landmarks or surfaces/lines and emphasize independence from the registration procedure. This is a different image domain, but the methodological warning applies to any registration benchmark. [Rohlfing, “Image Similarity and Tissue Overlaps as Surrogates…”](https://pmc.ncbi.nlm.nih.gov/articles/PMC3274625/)
- Coastal shoreline positional error depends on alongshore location and morphology; one CoastSnap study reports larger errors at the more dynamic/farther transect. [Assessing shoreline dynamics and beach use in a tropical beach](https://pmc.ncbi.nlm.nih.gov/articles/pmid/42726327/)

### Inferences
- For withheld image landmarks, report Euclidean target-registration error (TRE) per point, with median, mean, RMSE, 95th percentile, maximum, signed x/y bias, and a cumulative proportion below meaningful pixel thresholds. Convert pixel errors to meters using the local image-to-ground mapping at the tested point; because oblique perspective changes scale, do not apply one global pixels-to-meters factor.
- Give confidence intervals, preferably by bootstrapping at the image-pair or date level rather than treating many points from one image as independent observations. This accounts for within-image clustering; retain per-site/per-condition intervals where sample size allows. This is a recommended design inference, not a CoastSnap-published standard.
- Define tolerances before testing from the downstream use. For example, if the goal is to detect shoreline changes of a certain magnitude, registration error should be a suitably smaller part of that error budget; report the fraction of held-out locations exceeding this tolerance and count catastrophic failures explicitly.
- Compare variants on identical held-out sets; report paired change in TRE, confidence interval, and failure coverage. Perform ablations for registration model class, feature/matching strategy, distortion correction, image resizing/cropping, and reference/target choice.
- Add an end-to-end impact metric: rectify the registered image, extract shoreline with a fixed downstream method, and compare its cross-shore location to contemporaneous RTK-GNSS/DGPS along predefined transects. Report signed bias, MAE/MAD, RMSE/RMSD, median absolute error, 90th/95th percentile, fraction within application tolerance, and coverage. Keep registration-only and shoreline-output scores separate so the source of error remains diagnosable.
- Include repeatability by rerunning the pipeline with fixed settings and, if randomized matching/RANSAC is used, multiple seeds. Report both run-to-run transform dispersion and whether the same images cross the pass threshold. Repeatability is not a substitute for accuracy.

### Gaps
- The sources do not prescribe confidence-interval methodology, bootstrap resampling unit, or pass/fail threshold for CoastSnap registration; these should be chosen before evaluation and justified against intended monitoring resolution and survey uncertainty.
- CoastSnap's observed shoreline errors include shoreline-proxy, water-level, rectification, and survey timing effects in addition to image registration. A shoreline-vs-GNSS score therefore validates the complete operational chain, not registration alone.

## Practical benchmark protocol

1. Build a labeled corpus containing ordinary and challenging CoastSnap pairs from multiple dates, devices, and stations. Store the raw/reference pair, metadata, a set of surveyed or visually stable landmark correspondences, and an independent review/adjudication record.
2. For each pair, use a subset of landmarks for any fitting or tuning. Score the transform on withheld landmarks spread across the image, including edges and near/far beach regions. Keep a final holdout split by date/device/site that is never used to tune thresholds or select the best model.
3. Report registration-only scores in pixels and spatial units: signed x/y bias, median and mean TRE, RMSE, 95th percentile, maximum, per-location residuals, percentage within predeclared tolerances, and registration coverage/failure rate. Break down by site, camera/device, feature density, image region, and acquisition condition.
4. Use paired comparisons across pipeline variants, with confidence intervals clustered/resampled by image pair or date. Rerun stochastic methods over multiple seeds and record transform variability.
5. On a representative subset with concurrent RTK-GNSS/DGPS, run the full registration → rectification → shoreline mapping workflow and compare cross-shore shoreline positions at fixed transects. Report end-to-end bias and error distribution separately from TRE.
6. Publish failure examples and residual maps; investigate gross residuals instead of trimming them without explanation. Establish accept/reject limits from the scientific use case, e.g. the smallest shoreline change the project aims to resolve, then state whether the pipeline meets that requirement and for which strata.
