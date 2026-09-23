# Top ROI selection from the 60-stimulus EEG-fMRI fusion

## What was found in the project

- `fusion_movie_60stimuli.mp4` is the all-60-stimulus whole-brain fusion movie (574 rendered frames, 4 fps, 3228 x 2050).
- `FSLeyes_outputs/merged_RSA_movie.mp4` is the condition-comparison movie (574 frames, 4 fps, 3202 x 2060).
- `rsa_subject_mean_nonan.nii.gz` is the group whole-brain space-by-time fusion volume.
- `aal3_fusion_connectivity_outputs/all_trials_aal3_roi_eeg_fmri_fusion_timeseries.npz` contains the all-trials ROI fusion: 20 subjects x 64 valid AAL3 masks x 575 times (-296 to 2000 ms).
- Each ROI/time value is a Spearman correlation between the 1,770 entries of a 60 x 60 fMRI ROI RDM and the corresponding EEG RDM. It is representational alignment, not BOLD activation and not conventional functional connectivity.
- The requested AAL3 list contains 65 masks, but `ThalRe` has insufficient overlap after resampling, so 64 are present in the cached fusion.

## Why the ranking was not based on one peak

The original peak correlated with AAL3 parcel size (Spearman rho about 0.67). The reanalysis therefore recomputed each eligible ROI RDM from 20 independent samples of 100 finite, nonconstant fMRI voxels. Forty-five of the 64 cached ROIs met this criterion; small nuclei and masks with fewer than 100 globally valid voxels remain in the full table but are not assigned a balanced rank.

The robust score averages percentile ranks for:

1. Gaussian-smoothed peak (sigma 20 ms).
2. Positive and signed mean fusion over 0-2000 ms.
3. Subject-level t value and fraction positive near the group peak.
4. Leave-one-subject-out temporal reliability.
5. Consistency across pleasant, neutral, and unpleasant RDM fusions.
6. Maximum t value over time.

Rank stability was estimated with 2,000 subject bootstraps. Positive effects were also tested with 5,000 sign-flip permutations using the maximum t across all eligible ROIs and post-stimulus time points.

## Literal data-driven top 10

| Rank | AAL3 ROI | Smoothed peak r | Peak ms | Bootstrap top-10 frequency | Max-t FWER p |
|---:|---|---:|---:|---:|---:|
| 1 | Cerebellum | 0.0697 | 740 | 0.997 | 0.018 |
| 2 | OccipitalMid | 0.0609 | 756 | 0.975 | 0.280 |
| 3 | SupraMarginal | 0.0615 | 556 | 0.718 | 0.322 |
| 4 | Postcentral | 0.0473 | 556 | 0.683 | 0.533 |
| 5 | Fusiform | 0.0533 | 576 | 0.974 | 0.110 |
| 6 | OccipitalInf | 0.0497 | 764 | 0.885 | 0.232 |
| 7 | Lingual | 0.0547 | 756 | 0.913 | 0.584 |
| 8 | Calcarine | 0.0423 | 756 | 0.475 | 0.969 |
| 8 | TemporalMid | 0.0247 | 1600 | 0.162 | 0.810 |
| 10 | ParietalSup | 0.0370 | 556 | 0.341 | 0.514 |

Only cerebellum survives the deliberately strict global max-t correction. The others are discovery ranks, not confirmed significant effects. Peak latency is also not onset latency.

## Recommended 10-ROI confirmatory panel

The literal list mixes visual, cerebellar, parietal, and somatosensory parcels. For an emotion-fusion study, use a preregistered panel that separates measured discovery targets from independent emotion-network priors:

| Tier | AAL3 ROI | Reason |
|---|---|---|
| Observed | OccipitalMid | Strong visual-association fusion |
| Observed | Fusiform | Strong ventral-stream fusion and prior affective-scene ROI |
| Observed | OccipitalInf | Strong lateral/ventral occipital fusion |
| Observed | Lingual | Strong ventromedial visual fusion |
| Observed | Calcarine | Early-visual anchor |
| Observed | TemporalMid | Ranked temporal-association/cascade target |
| Observed/exploratory | Cerebellum | Strongest corrected discovery; needs focused replication |
| Prior-defined | Insula | Emotion/salience node tested in the source study |
| Prior-defined | Amygdala | Emotion/reentry node tested in the source study |
| Prior-defined | ACCsup | AAL3 supracallosal/dorsal ACC prior tested in the source study |

Do not describe amygdala or ACCsup as top correlations in this averaged-trial analysis: their balanced ranks are 34 and 43. They are included because they are independent hypotheses.

## Compact four-ROI visual-stream panel

Assuming the repeated request for a second dorsal-stream ROI meant ventral stream, use:

| Visual role | AAL3 ROI | Balanced data rank | Main cross-atlas correspondence |
|---|---|---:|---|
| V1 / early visual cortex | Calcarine | 8 | Harvard-Oxford Intracalcarine Cortex; HCP V1/V2 |
| Dorsal visual stream | ParietalSup | 10 | Harvard-Oxford Superior Parietal Lobule; HCP 7AL/7Am/7PC/7PL/IPS1 |
| Ventral visual stream | Fusiform | 5 | Harvard-Oxford fusiform divisions; HCP FFC/V8/VMV1-3 |
| Lateral visual cortex | OccipitalMid | 2 | Harvard-Oxford Lateral Occipital Cortex; HCP LO1/LO2/V3CD/V4t |

This panel is saved as `recommended_4_roi_visual_stream_panel.csv`.

For the revised Harvard-Oxford-only implementation, use `Occipital Pole` (surface-visible early visual proxy), `Superior Parietal Lobule` (dorsal stream), `Temporal Occipital Fusiform Cortex` (ventral stream), and `Lateral Occipital Cortex, inferior division` (closest single parcel to lateral occipitotemporal cortex). The atlas variant is `cort-maxprob-thr25-2mm`; the panel is saved as `recommended_4_roi_harvard_oxford_panel_surface_otc.csv`. `Occipital Pole` is broader and less specifically V1 than `Intracalcarine Cortex`; that tradeoff is intentional here.

## What prior work did

- Bo et al. (2022), using this same 20-subject/60-IAPS experiment, defined early, ventral, and dorsal visual ROIs from retinotopic anatomy, assessed onset with a five-baseline-SD plus consecutive-time criterion, and bootstrapped onset differences. It separately tested amygdala, dACC, anterior insula, and fusiform cortex. The paper's main fusion used 300 single trials, whereas this ranking uses 60 averaged stimulus patterns; its approximately 80-100 ms visual onset should not be compared directly with the later peaks here. https://pubmed.ncbi.nlm.nih.gov/35931307/
- Cichy, Pantazis, and Oliva (2014) used hypothesis-driven V1 and inferior-temporal ROIs, whole-brain fusion, sign-permutation cluster correction, and an independent fMRI dataset for reproducibility. https://www.nature.com/articles/nn.3635
- ROI selection and testing on the same noisy statistic creates circular inference. These ranks should define discovery targets; confirmatory effect estimates require held-out subjects/sessions, nested cross-validation, or an external sample. https://www.nature.com/articles/nn.2303

## Atlas interpretation

- AAL3 supplies volumetric cortical, subcortical, thalamic, brainstem, and cerebellar masks. https://pubmed.ncbi.nlm.nih.gov/31521825/
- Harvard-Oxford overlap in `aal3_harvardoxford_hcpmmp_crosswalk.csv` was computed spatially against the 25% maximum-probability cortical and subcortical atlases.
- HCP-MMP1.0 is a finer, surface-based cortical atlas with 180 areas per hemisphere. Its entries in the crosswalk are anatomical approximations, not voxelwise equivalences; amygdala and cerebellum are outside cortical HCP-MMP1.0. https://humanconnectome.org/study/hcp-young-adult/article/nature-article-cortical-brain-maps-at-the-highest-resolution-to-date
- The current AAL3 masks combine left and right hemispheres. A directional or connectivity follow-up should split hemispheres before estimating edges; HCP-MMP1.0 is already hemisphere-specific.

## Main outputs

- `aal3_roi_robust_ranking.csv`: all metrics for all 64 cached AAL3 ROIs.
- `literal_top10_data_driven_with_atlas_crosswalk.csv`: literal top 10 plus atlas mappings.
- `recommended_10_roi_confirmatory_panel.csv`: observed-plus-prior panel.
- `aal3_harvardoxford_hcpmmp_crosswalk.csv`: computed Harvard-Oxford and approximate HCP labels.
- `figures/top10_balanced_fusion_timecourses.png`: group mean and SEM curves.
- `figures/top20_ranking_metric_heatmap.png`: method agreement/disagreement.
- `select_top_fusion_rois.py` and `build_roi_study_panel.py`: reproducible code in the project root.
