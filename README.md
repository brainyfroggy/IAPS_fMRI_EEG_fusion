# IAPS fMRI-EEG Fusion

Representational similarity analysis (RSA) fusion of fMRI and EEG responses
to 60 IAPS images (20 pleasant / 20 neutral / 20 unpleasant), following the
approach of Cichy, Pantazis & Oliva (2014) applied to the 20-subject IAPS
dataset of Bo et al. (2022).

This repository contains only the analysis code and documentation. All
neuroimaging data, arrays, figures, movies, and result tables (roughly 61 GB
in the original project) have been excluded — see **Excluded from this
repository** below.

## Purpose

fMRI and EEG each measure brain activity to the same visual stimuli but
trade off spatial and temporal resolution in opposite directions. RSA fusion
lets you ask, at millisecond resolution, *when* a given brain region's
fMRI-derived representational geometry (a 60x60 dissimilarity matrix over
the 60 IAPS images) starts to match the EEG's representational geometry at
each time point — "representational alignment," not raw activation or
functional connectivity.

This code:

1. Computes whole-brain and searchlight/ROI fMRI-EEG fusion time courses per
   subject.
2. Builds ROI-based fusion "dynamic connectivity" summaries and ranks AAL3
   ROIs by fusion strength (robust, bootstrap-based ranking).
3. Renders FreeSurfer/FSLeyes cortical-surface movies of the fusion time
   course, for both the pooled 60-stimulus analysis and split by emotion
   category (pleasant/neutral/unpleasant).
4. Runs onset-latency statistics (when does fusion become significant, and
   do different ROI/model pairs diverge in onset timing) with bootstrap and
   permutation testing.

## Contents

- `RSA_fmri_eeg.ipynb` — computes subject-level fMRI-EEG RSA fusion
  (loads GIST/behavioral/CLIP-derived ROI masks, builds fMRI RDMs, and
  correlates them against EEG RDMs at each time point). This is the source
  of the main fusion file for the 60-stimulus (non-emotion-split) analysis.
- `RSA_fmri_eeg_wholebrain.ipynb` — a separate, voxelwise whole-brain
  variant (no ROI masks): generates subject-level fusion, averages across
  subjects, then builds the FreeSurfer surface mesh directly. Used for the
  by-emotion analysis (see **Pipeline: fusion split by emotion category**
  below).
- `rsa_4d.ipynb` — averages per-subject fusion volumes into the group mean
  4D fusion volume (`rsa_subject_mean.nii.gz` in the original project).
- `AAL3_EEG_fMRI_representational_connectivity.ipynb` — computes ROI-wise
  (AAL3 atlas) fMRI-EEG fusion time series ("representational connectivity")
  per subject and condition.
- `append_step18_voxel_connectivity.py`, `top25_voxel_dynamic_connectivity.py`
  — voxel-level dynamic-connectivity follow-up steps appended to the AAL3
  pipeline.
- `select_top_fusion_rois.py`, `build_roi_study_panel.py` — rank AAL3 ROIs by
  robust fusion metrics (bootstrap/permutation based) and assemble the
  confirmatory/exploratory ROI study panels documented in
  `top_roi_selection_outputs/README.md`.
- `create_harvard_oxford_4roi_mask.py`, `create_top25_peak_fusion_4roi_mask.py`
  — build reduced 4-ROI masks (Harvard-Oxford and data-driven top-25-peak
  variants) for follow-up analyses.
- `create_surface.sh`, `create_inflated_surface.sh` — FreeSurfer/WSL shell
  pipelines that project the volumetric fusion NIfTI onto the fsaverage
  pial/inflated cortical surface and combine per-frame GIFTI files into a
  multi-frame surface file for FSLeyes movie rendering (see **Pipeline:
  whole-brain surface movie** below).
- `generate_video.ipynb`, `generate_video_3emo.ipynb`, `merge_frames_to_video.sh`
  — break an FSLeyes-rendered GIF into PNG frames and re-encode them into an
  MP4 video (all-60-stimuli and by-emotion variants respectively).
- `nih_new_iso.cmap` — FSLeyes colormap used for rendering the fusion
  surface movies.
- `Searchlight ROI EEG/` — searchlight and ROI-based onset/sensitivity
  analyses of the fusion time course:
  - `sensitivity_top_percent_fusion.py` — top-percent-voxel sensitivity
    analysis for the searchlight fusion maps.
  - `analysis_main_clean_combined.ipynb` — the main ROI/searchlight
    onset-latency analysis notebook: loads per-condition fusion RSA arrays,
    detects onset latency per ROI/model via baseline-corrected threshold
    crossing, and produces a combined publication figure (time courses,
    bootstrap onset confidence intervals, and pairwise onset-difference
    tests with permutation p-values and Bonferroni correction). This is the
    final, cleaned version of the analysis — earlier drafts
    (`analysis.ipynb`, `analysis_main.ipynb`, `analysis_main_configurable.ipynb`,
    `analysis_main_clean_combined_with_permutation.ipynb`) were removed as
    superseded/broken (see the git history for details).
  - `onset_sensitivity_bootstrap_test.ipynb` — bootstrap sensitivity test for
    onset-latency estimates across a grid of analysis settings (smoothing,
    threshold, outlier handling).
  - `environment.yml`, `requirements.txt` — conda/pip environment
    specifications for this subfolder's analyses.
- `top_roi_selection_outputs/README.md` — write-up of the AAL3 ROI ranking
  methodology and results (robust/bootstrap ranking metrics, literal top-10
  ROIs, recommended confirmatory and compact visual-stream ROI panels,
  atlas-crosswalk notes). Kept as-is; the result data/figures it describes
  are excluded.
- `top_roi_selection_outputs/atlases/README.md` — a one-line note that this
  folder is Nilearn's atlas/dataset cache and can be safely deleted and
  re-downloaded.

## How to Use

None of these notebooks/scripts include data — you need your own fMRI
single-trial betas and EEG epochs for the same 60 IAPS stimuli/subjects
(the original project used Bo et al.'s 20-subject dataset). Each notebook
documents its own expected input file names/paths in its early cells; adjust
those paths to your own data layout before running.

### Pipeline: whole-brain surface movie, all 60 stimuli

1. Run `RSA_fmri_eeg.ipynb` to generate subject-level whole-brain RSA fusion
   volumes (input: per-subject fMRI single-trial betas + EEG epochs; output:
   per-subject fusion arrays/volumes).
2. Run `rsa_4d.ipynb` to average across subjects into the group mean 4D
   fusion volume (`rsa_subject_mean.nii.gz`).
3. Remove NaNs from the volume with `nibabel`/`numpy`
   (`np.nan_to_num(data, nan=0.0)`), saving as float32
   (`rsa_subject_mean_nonan.nii.gz`).
4. Build the per-frame surface files with FreeSurfer under WSL: for each
   time frame, run `mri_vol2surf --mov <4D nifti> --frame <f> --mni152reg
   --hemi {lh,rh} --trgsubject fsaverage --projfrac 0.5 --interp trilinear
   --o <hemi>.tp<frame>.mgh`, then `mri_concat` the per-frame `.mgh` files
   into a single multi-frame file.
5. Convert the FreeSurfer pial (or inflated, for the inflated-surface
   variant) mesh and the per-frame functional data to GIFTI with
   `mris_convert` (`create_surface.sh` / `create_inflated_surface.sh` do
   this), then combine the per-frame `.func.gii` files into one multi-frame
   `.func.gii` per hemisphere (matching frames by the `tp###` index).
6. Open FSLeyes, load the 4D image, switch to the 3D view panel, open the
   FSLeyes Python console, and run:
   ```python
   panel = [p for p in frame.viewPanels if p.__class__.__name__ == 'Scene3DPanel'][0]
   from fsleyes.actions.moviegif import MovieGifAction
   movie_action = MovieGifAction(overlayList, displayCtx, panel)
   movie_action()
   ```
   to render the animated GIF.
7. Use `generate_video.ipynb` to break the GIF into PNG frames and
   `merge_frames_to_video.sh` to re-encode the frames into an MP4.

### Pipeline: fusion split by emotion category

This variant repeats the same fusion/rendering pipeline separately for each
emotion condition (pleasant / neutral / unpleasant) instead of pooling all
60 stimuli.

1. Run `RSA_fmri_eeg_wholebrain.ipynb` to generate subject-level whole-brain
   RSA fusion per condition, then again (or with an averaging step) to get
   the mean RSA across subjects, producing `rsa_subject_mean.nii.gz` per
   condition.
2. Create the surface mesh with `create_surface.sh` (requires FreeSurfer,
   a FreeSurfer license, and FSL).
3. Combine the per-frame GIFTI files into a multi-frame GIFTI (only needed
   for the pial-surface variant; not needed for the inflated-surface
   variant, via `create_inflated_surface.sh`).
4. Open FSLeyes with the fsaverage inflated surface
   (`$FREESURFER_HOME/subjects/fsaverage/surf/{lh,rh}.inflated`), load the
   surface definition GIFTIs and the per-condition vertex data, set the
   color map/min-max, and run the same `MovieGifAction` console snippet as
   above to render the GIF.
5. Run `generate_video_3emo.ipynb` to cut the GIF into PNG frames, then
   `merge_frames_to_video.sh` to build the final by-emotion comparison
   video.

### ROI ranking / dynamic-connectivity follow-up

Starting from the AAL3 ROI-wise fusion time series produced by
`AAL3_EEG_fMRI_representational_connectivity.ipynb` (input: subject fMRI
betas + EEG epochs + AAL3 atlas masks; output: per-subject/condition
ROI-by-time fusion arrays):

1. `append_step18_voxel_connectivity.py` / `top25_voxel_dynamic_connectivity.py`
   add voxel-level dynamic-connectivity summaries on top of the ROI arrays.
2. `select_top_fusion_rois.py` ranks ROIs by a robust, bootstrap-based
   fusion score and writes the top-N ROI list.
3. `build_roi_study_panel.py` assembles the confirmatory/exploratory ROI
   study panel from that ranking. Output/methodology write-up:
   `top_roi_selection_outputs/README.md`.
4. `create_harvard_oxford_4roi_mask.py` / `create_top25_peak_fusion_4roi_mask.py`
   build reduced 4-ROI masks from the ranking for targeted follow-up
   analyses.

### Onset-latency analysis (`Searchlight ROI EEG/`)

Given per-condition fusion RSA arrays (`gistevc_rsa`, `behotc_rsa`,
`clipvit_rsa`, each `(n_subjects, n_timepoints)`, e.g. saved as
`gistevc_rsa_20260603.npy` etc. under a `results/` folder, or an
`.npz` bundle with the shared time axis):

1. Run `analysis_main_clean_combined.ipynb` top to bottom. Set
   `LOAD_RSA_FROM_NPY = False` in the configuration cell if you need to
   recompute the fusion RSA arrays from raw EEG/fMRI data instead of
   loading cached arrays. Output: a combined publication figure (RSA time
   courses with onset markers, bootstrap onset confidence-interval
   boxplots, and pairwise onset-difference tests with permutation
   p-values, Bonferroni-corrected across comparisons) saved as PNG/PDF.
2. `onset_sensitivity_bootstrap_test.ipynb` re-runs the onset-detection
   step across a grid of smoothing/threshold/outlier-handling settings to
   check how sensitive the onset estimates are to those choices. Output:
   summary tables and per-configuration plots under
   `onset_sensitivity_outputs/`.
3. `sensitivity_top_percent_fusion.py` runs a top-percent-voxel sensitivity
   analysis on the searchlight (voxelwise, not ROI) fusion maps.

## Dependencies

These notebooks/scripts expect: a working FreeSurfer installation with a
valid license, FSL, FSLeyes (with the FSLeyes Python console and
`fsleyes.actions.moviegif`), Nilearn (for AAL3/Harvard-Oxford atlas
downloads — see `top_roi_selection_outputs/atlases/README.md`), and standard
neuroimaging/scientific Python packages (`nibabel`, `numpy`, `scipy`,
`nilearn`, `pandas`, `rsatoolbox`, `statsmodels`, `scikit-image`, `tqdm`).
See `Searchlight ROI EEG/environment.yml` and
`Searchlight ROI EEG/requirements.txt` for the pinned environment used for
the searchlight/ROI onset analyses.

## Excluded from this repository

Per data-safety policy, everything not listed under **Contents** above was
left out, including (non-exhaustive):

- All neuroimaging/array data: `.gii`, `.nii.gz`/`.gz`, `.npy`, `.npz`,
  `.mgh`/`.mgz` files (e.g. root `rsa_subject*.nii.gz`, `rsa_tp255.mgz`,
  `Searchlight ROI EEG/eeg_raw.npy`, `Searchlight ROI EEG/rsa_eeg_fmri_essentials.npz`).
- All rendered figures/movies: `.png`, `.gif`, `.mp4`, movie `.zip` archives
  (e.g. root `LH.gif`, `MID.gif`, `RH.gif`, `TOP.gif`, `movie.gif`,
  `fusion_movie_60stimuli.mp4`/`.zip`).
- Result tables and reports: `.csv`, `.html`, `.pdf`, `.xlsx` (e.g.
  `kastner_rsa_simple.xlsx`).
- `Thumbs.db`.
- Entire result-output directories: `searchlight_rsa_*/`, `fs_rsa_*/`,
  `FSLeyes/`, `FSLeyes_outputs/`, `aal3_fusion_connectivity_outputs*/`,
  `top_roi_selection_outputs/*` other than the two README.md files kept,
  `partial_rsa_outputs/`, and all result subfolders under
  `Searchlight ROI EEG/` (e.g. `results/`, `subject_onset_results/`,
  `onset_main_configurable_*/`, `onset_sensitivity_outputs/`,
  `baseline_corrected_*_analysis/`, `group_level_bootstrap_onset_analysis/`,
  `independent_onset_test/`, `preonset_2sigma_onset_test/`,
  `sensitivity_top_percent_tmaps_20260810/`, and `__pycache__/`).
- `readme.docx` and `readme_wholebrain_rsa_fusion.docx` themselves — their
  pipeline instructions are folded into this README instead (see above).
- Small metadata `.json` files sitting inside the excluded output folders
  were also left out by default, since it was not certain they contain no
  derived data values worth treating cautiously.
- `RSA_fmri_eeg og.ipynb` and four early/broken drafts of the
  `Searchlight ROI EEG/` onset-latency notebook — removed as
  unfinished/superseded code (not data); see the git history for the
  removal commit and rationale.
