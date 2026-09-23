"""Robust ROI selection for the 60-stimulus IAPS EEG-fMRI RSA fusion.

This script ranks AAL3 ROIs using the existing all-trials fusion and a
voxel-count-balanced reanalysis. It is intentionally a discovery analysis:
selected ROIs must be tested on independent data or inside nested CV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_img
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata, spearmanr, ttest_1samp


N_VOXELS = 100
N_VOXEL_REPEATS = 20
SMOOTH_SIGMA_MS = 20.0
N_PERMUTATIONS = 5000
N_BOOTSTRAPS = 2000
RANDOM_SEED = 20260723
ANALYSIS_VERSION = 2


def percentile_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 1:
        out[valid] = 1.0
    elif valid.sum() > 1:
        out[valid] = (rankdata(values[valid], method="average") - 1) / (valid.sum() - 1)
    return out


def baseline_correct(x: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
    baseline = times_ms < 0
    return x - np.nanmean(x[..., baseline], axis=-1, keepdims=True)


def normalize_for_correlation(values: np.ndarray, method: str = "spearman") -> np.ndarray:
    x = np.asarray(values, dtype=float)
    original_shape = x.shape
    if x.ndim == 1:
        x = x[None, :]
    if method == "spearman":
        x = rankdata(x, axis=-1, method="average")
    x = x - np.mean(x, axis=-1, keepdims=True)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = np.nan
    x = x / norm
    return x[0] if len(original_shape) == 1 else x


def load_resampled_masks(
    roi_names: list[str], atlas_dir: Path, template_img: nib.spatialimages.SpatialImage
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    masks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for roi in roi_names:
        path = atlas_dir / f"{roi}.nii.gz"
        if not path.exists():
            continue
        img = resample_img(
            nib.load(str(path)),
            target_affine=template_img.affine,
            target_shape=template_img.shape[:3],
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        mask = np.asarray(img.get_fdata() > 0).ravel()
        masks[roi] = mask
        counts[roi] = int(mask.sum())
    return masks, counts


def compute_balanced_fmri_rdms(
    allsub_avg: np.ndarray,
    roi_names: list[str],
    masks: dict[str, np.ndarray],
    cache_path: Path,
    n_voxels: int,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as z:
            cache_version = int(z["analysis_version"]) if "analysis_version" in z else 0
            if (
                cache_version == ANALYSIS_VERSION
                and int(z["n_voxels"]) == n_voxels
                and int(z["n_repeats"]) == n_repeats
            ):
                print("Loading balanced fMRI RDMs:", cache_path)
                return z["fmri_vecs_balanced"]

    n_subjects, n_stimuli, _ = allsub_avg.shape
    n_pairs = n_stimuli * (n_stimuli - 1) // 2
    rdms = np.full((n_subjects, len(roi_names), n_pairs), np.nan, dtype=np.float32)
    rng = np.random.default_rng(seed)

    print("Building global finite/nonconstant fMRI voxel mask")
    globally_eligible = np.ones(allsub_avg.shape[-1], dtype=bool)
    for subject in range(n_subjects):
        subj = np.asarray(allsub_avg[subject], dtype=np.float32)
        globally_eligible &= np.isfinite(subj).all(axis=0)
        globally_eligible &= np.std(subj, axis=0) > 1e-8

    subsets: dict[str, list[np.ndarray]] = {}
    eligible_counts: dict[str, int] = {}
    for roi in roi_names:
        idx = np.flatnonzero(masks[roi])
        eligible = idx[globally_eligible[idx]]
        eligible_counts[roi] = int(len(eligible))
        if len(eligible) >= n_voxels:
            subsets[roi] = [
                rng.choice(eligible, n_voxels, replace=False) for _ in range(n_repeats)
            ]

    for subject in range(n_subjects):
        print(f"Balanced fMRI RDMs: subject {subject + 1:02d}/{n_subjects}")
        subj = np.asarray(allsub_avg[subject], dtype=np.float32)
        for roi_idx, roi in enumerate(roi_names):
            if roi not in subsets:
                continue
            repeat_rdms = []
            for voxel_idx in subsets[roi]:
                patterns = subj[:, voxel_idx]
                valid = np.isfinite(patterns).all(axis=0) & (np.std(patterns, axis=0) > 1e-8)
                patterns = patterns[:, valid]
                if patterns.shape[1] < n_voxels * 0.9:
                    continue
                if np.any(np.std(patterns, axis=1) <= 1e-8):
                    continue
                rdm = pdist(patterns, metric="correlation")
                if np.isfinite(rdm).all():
                    repeat_rdms.append(rdm)
            if repeat_rdms:
                rdms[subject, roi_idx] = np.nanmean(repeat_rdms, axis=0)

    np.savez_compressed(
        cache_path,
        fmri_vecs_balanced=rdms,
        roi_names=np.asarray(roi_names),
        n_voxels=np.asarray(n_voxels),
        n_repeats=np.asarray(n_repeats),
        seed=np.asarray(seed),
        analysis_version=np.asarray(ANALYSIS_VERSION),
        eligible_voxel_counts=np.asarray([eligible_counts[r] for r in roi_names]),
    )
    print("Saved:", cache_path)
    return rdms


def fuse_rdms(fmri_rdms: np.ndarray, eeg_rdms: np.ndarray) -> np.ndarray:
    n_subjects, n_rois, _ = fmri_rdms.shape
    n_time = eeg_rdms.shape[1]
    fusion = np.full((n_subjects, n_rois, n_time), np.nan, dtype=np.float32)
    for subject in range(n_subjects):
        print(f"Balanced fusion: subject {subject + 1:02d}/{n_subjects}")
        eeg_norm = normalize_for_correlation(eeg_rdms[subject], method="spearman")
        for roi in range(n_rois):
            if not np.isfinite(fmri_rdms[subject, roi]).all():
                continue
            fmri_norm = normalize_for_correlation(fmri_rdms[subject, roi], method="spearman")
            fusion[subject, roi] = eeg_norm @ fmri_norm
    return fusion


def condition_fusion_from_all_trial_rdms(
    fmri_rdms: np.ndarray, eeg_condition_rdms: np.ndarray
) -> np.ndarray:
    n_subjects, n_rois, _ = fmri_rdms.shape
    n_conditions, n_time = eeg_condition_rdms.shape[1:3]
    out = np.full((n_subjects, n_conditions, n_rois, n_time), np.nan, dtype=np.float32)
    condition_indices = [np.arange(0, 20), np.arange(20, 40), np.arange(40, 60)]
    for subject in range(n_subjects):
        for roi in range(n_rois):
            if not np.isfinite(fmri_rdms[subject, roi]).all():
                continue
            square = squareform(fmri_rdms[subject, roi])
            for condition, idx in enumerate(condition_indices):
                roi_vec = squareform(square[np.ix_(idx, idx)], checks=False)
                roi_norm = normalize_for_correlation(roi_vec, method="spearman")
                eeg_norm = normalize_for_correlation(
                    eeg_condition_rdms[subject, condition], method="spearman"
                )
                out[subject, condition, roi] = eeg_norm @ roi_norm
    return out


def max_t_inference(
    smoothed_bc: np.ndarray,
    times_ms: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    post = times_ms >= 0
    x = smoothed_bc[..., post]
    valid_rois = np.isfinite(x).all(axis=(0, 2))
    xv = x[:, valid_rois, :]
    n = xv.shape[0]
    observed_t = np.nanmean(xv, axis=0) / (np.nanstd(xv, axis=0, ddof=1) / np.sqrt(n))
    observed_max = np.nanmax(observed_t, axis=1)

    rng = np.random.default_rng(seed)
    sumsq = np.sum(xv * xv, axis=0)
    perm_max = np.empty(n_permutations, dtype=float)
    batch_size = 100
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, n))
        signed_sum = np.tensordot(signs, xv, axes=(1, 0))
        mean = signed_sum / n
        variance = (sumsq[None, ...] - signed_sum * signed_sum / n) / (n - 1)
        t_perm = mean / np.sqrt(variance / n)
        perm_max[start:stop] = np.nanmax(t_perm, axis=(1, 2))

    critical_t = float(np.quantile(perm_max, 0.95))
    p_fwer_valid = (1 + np.sum(perm_max[:, None] >= observed_max[None, :], axis=0)) / (
        n_permutations + 1
    )
    p_fwer = np.full(x.shape[1], np.nan)
    max_t = np.full(x.shape[1], np.nan)
    p_fwer[valid_rois] = p_fwer_valid
    max_t[valid_rois] = observed_max
    return max_t, p_fwer, critical_t, observed_t


def first_consecutive_onset(
    curve: np.ndarray, threshold: float, times: np.ndarray, run_length: int = 5
) -> float:
    above = np.asarray(curve > threshold, dtype=np.int8)
    hits = np.convolve(above, np.ones(run_length, dtype=int), mode="valid")
    idx = np.flatnonzero(hits == run_length)
    return float(times[idx[0]]) if len(idx) else np.nan


def summarize_fusion(
    fusion: np.ndarray,
    condition_fusion: np.ndarray,
    times_ms: np.ndarray,
    roi_names: list[str],
    voxel_counts: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    valid_rois = np.isfinite(fusion).all(axis=(0, 2))
    bc = np.full_like(fusion, np.nan, dtype=float)
    bc[:, valid_rois] = baseline_correct(fusion[:, valid_rois], times_ms)
    sigma_samples = SMOOTH_SIGMA_MS / float(np.median(np.diff(times_ms)))
    smoothed = gaussian_filter1d(bc, sigma_samples, axis=-1, mode="nearest")
    post = times_ms >= 0
    post_times = times_ms[post]
    group = np.full((fusion.shape[1], fusion.shape[2]), np.nan, dtype=float)
    group[valid_rois] = np.mean(smoothed[:, valid_rois], axis=0)
    group_post = group[:, post]

    raw_group = np.full_like(group_post, np.nan)
    raw_group[valid_rois] = np.mean(bc[:, valid_rois][:, :, post], axis=0)
    raw_peak = np.full(fusion.shape[1], np.nan)
    peak = np.full(fusion.shape[1], np.nan)
    peak_idx = np.zeros(fusion.shape[1], dtype=int)
    peak_time = np.full(fusion.shape[1], np.nan)
    positive_mean = np.full(fusion.shape[1], np.nan)
    sustained_mean = np.full(fusion.shape[1], np.nan)
    raw_peak[valid_rois] = np.max(raw_group[valid_rois], axis=1)
    peak[valid_rois] = np.max(group_post[valid_rois], axis=1)
    peak_idx[valid_rois] = np.argmax(group_post[valid_rois], axis=1)
    peak_time[valid_rois] = post_times[peak_idx[valid_rois]]
    positive_mean[valid_rois] = np.trapezoid(
        np.maximum(group_post[valid_rois], 0), post_times, axis=1
    ) / (post_times[-1] - post_times[0])
    sustained_mean[valid_rois] = np.mean(group_post[valid_rois], axis=1)

    subject_peak = np.full((fusion.shape[0], fusion.shape[1]), np.nan)
    for roi in np.flatnonzero(valid_rois):
        lo = max(0, int(peak_idx[roi]) - 5)
        hi = min(group_post.shape[1], int(peak_idx[roi]) + 6)
        subject_peak[:, roi] = np.nanmean(smoothed[:, roi, post][:, lo:hi], axis=1)
    t_at_peak = np.full(fusion.shape[1], np.nan)
    subject_consistency = np.full(fusion.shape[1], np.nan)
    for roi in np.flatnonzero(valid_rois):
        t_at_peak[roi] = ttest_1samp(subject_peak[:, roi], 0).statistic
        subject_consistency[roi] = np.mean(subject_peak[:, roi] > 0)

    loo_reliability = np.full(fusion.shape[1], np.nan)
    for roi in range(fusion.shape[1]):
        if not np.isfinite(smoothed[:, roi]).all():
            continue
        correlations = []
        roi_post = smoothed[:, roi, post]
        total = roi_post.sum(axis=0)
        for subject in range(fusion.shape[0]):
            loo = (total - roi_post[subject]) / (fusion.shape[0] - 1)
            correlations.append(pearsonr(roi_post[subject], loo).statistic)
        loo_reliability[roi] = np.nanmedian(correlations)

    condition_bc = np.full_like(condition_fusion, np.nan, dtype=float)
    condition_bc[:, :, valid_rois] = baseline_correct(
        condition_fusion[:, :, valid_rois], times_ms
    )
    condition_smoothed = gaussian_filter1d(condition_bc, sigma_samples, axis=-1, mode="nearest")
    condition_group = np.full(condition_fusion.shape[1:], np.nan, dtype=float)
    condition_group[:, valid_rois] = np.mean(condition_smoothed[:, :, valid_rois], axis=0)
    condition_peak = np.full(condition_group.shape[:2], np.nan)
    condition_mean = np.full(condition_group.shape[:2], np.nan)
    condition_peak[:, valid_rois] = np.max(
        condition_group[:, valid_rois][:, :, post], axis=-1
    )
    condition_mean[:, valid_rois] = np.mean(
        condition_group[:, valid_rois][:, :, post], axis=-1
    )
    condition_score = np.mean(
        np.stack(
            [
                (percentile_score(condition_peak[c]) + percentile_score(condition_mean[c])) / 2
                for c in range(condition_group.shape[0])
            ]
        ),
        axis=0,
    )

    max_t, p_fwer, critical_t, observed_t = max_t_inference(
        smoothed, times_ms, n_permutations=n_permutations, seed=seed
    )

    onset_5sd = np.full(fusion.shape[1], np.nan)
    for roi in range(fusion.shape[1]):
        if not np.isfinite(group[roi]).all():
            continue
        baseline_values = group[roi, times_ms < 0]
        threshold = np.mean(baseline_values) + 5 * np.std(baseline_values, ddof=1)
        onset_5sd[roi] = first_consecutive_onset(
            group[roi, post], threshold, post_times, run_length=5
        )

    frame = pd.DataFrame(
        {
            "roi": roi_names,
            "voxel_count": voxel_counts,
            "raw_group_peak_r": raw_peak,
            "smoothed_group_peak_r": peak,
            "peak_time_ms": peak_time,
            "positive_mean_r_0_2000": positive_mean,
            "signed_mean_r_0_2000": sustained_mean,
            "t_at_group_peak": t_at_peak,
            "subjects_positive_at_peak_fraction": subject_consistency,
            "loo_temporal_reliability_median_r": loo_reliability,
            "condition_consistency_score": condition_score,
            "max_t_over_time": max_t,
            "max_t_fwer_p": p_fwer,
            "paper_style_5sd_onset_ms": onset_5sd,
            "max_t_fwer_critical_05": critical_t,
        }
    )

    score_columns = [
        "smoothed_group_peak_r",
        "positive_mean_r_0_2000",
        "signed_mean_r_0_2000",
        "t_at_group_peak",
        "subjects_positive_at_peak_fraction",
        "loo_temporal_reliability_median_r",
        "condition_consistency_score",
        "max_t_over_time",
    ]
    percentile_columns = []
    for column in score_columns:
        pct_column = f"pct_{column}"
        frame[pct_column] = percentile_score(frame[column].to_numpy())
        percentile_columns.append(pct_column)
    frame["robust_data_score"] = frame[percentile_columns].mean(axis=1, skipna=False)
    frame["robust_data_rank"] = frame["robust_data_score"].rank(ascending=False, method="min")
    return frame, smoothed, observed_t


def bootstrap_top10_frequency(
    smoothed_bc: np.ndarray, times_ms: np.ndarray, n_bootstraps: int, seed: int
) -> np.ndarray:
    post = times_ms >= 0
    x = smoothed_bc[..., post]
    valid = np.isfinite(x).all(axis=(0, 2))
    valid_idx = np.flatnonzero(valid)
    counts = np.zeros(x.shape[1], dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstraps):
        sample = rng.integers(0, x.shape[0], size=x.shape[0])
        xb = x[sample][:, valid, :]
        group = xb.mean(axis=0)
        peak = group.max(axis=1)
        positive_mean = np.maximum(group, 0).mean(axis=1)
        signed_mean = group.mean(axis=1)
        sd = xb.std(axis=0, ddof=1)
        max_t = np.nanmax(group / (sd / np.sqrt(x.shape[0])), axis=1)
        score = np.nanmean(
            np.stack(
                [
                    percentile_score(peak),
                    percentile_score(positive_mean),
                    percentile_score(signed_mean),
                    percentile_score(max_t),
                ]
            ),
            axis=0,
        )
        top = valid_idx[np.argsort(score)[-10:]]
        counts[top] += 1
    return counts / n_bootstraps


def create_figures(
    output_dir: Path,
    ranking: pd.DataFrame,
    smoothed_balanced: np.ndarray,
    times_ms: np.ndarray,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    top = ranking.sort_values("robust_data_rank").head(10)
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=True)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        idx = int(row["roi_index"])
        data = smoothed_balanced[:, idx]
        mean = np.nanmean(data, axis=0)
        sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.isfinite(data).sum(axis=0))
        ax.plot(times_ms, mean, color="#15616d", linewidth=2)
        ax.fill_between(times_ms, mean - sem, mean + sem, color="#15616d", alpha=0.2)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_title(
            f"#{int(row['robust_data_rank'])} {row['roi']} | peak {row['peak_time_ms']:.0f} ms",
            fontsize=11,
        )
        ax.set_ylabel("Fusion r")
    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    fig.suptitle("Top 10 AAL3 ROIs: 100-voxel-balanced EEG-fMRI fusion", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures / "top10_balanced_fusion_timecourses.png", dpi=180)
    plt.close(fig)

    metric_columns = [
        "pct_smoothed_group_peak_r",
        "pct_positive_mean_r_0_2000",
        "pct_t_at_group_peak",
        "pct_subjects_positive_at_peak_fraction",
        "pct_loo_temporal_reliability_median_r",
        "pct_condition_consistency_score",
        "pct_max_t_over_time",
    ]
    labels = ["Peak", "Positive mean", "Peak t", "Subject consistency", "LOO reliability", "3-condition consistency", "Max t"]
    top20 = ranking.sort_values("robust_data_rank").head(20)
    matrix = top20[metric_columns].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(top20)), [f"{int(r.robust_data_rank)}. {r.roi}" for _, r in top20.iterrows()])
    ax.set_title("ROI ranking components (percentile among 100-voxel-eligible ROIs)")
    fig.colorbar(im, ax=ax, label="Percentile")
    fig.tight_layout()
    fig.savefig(figures / "top20_ranking_metric_heatmap.png", dpi=180)
    plt.close(fig)

    valid = np.isfinite(ranking["robust_data_score"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        np.log10(ranking.loc[valid, "voxel_count"]),
        ranking.loc[valid, "unbalanced_smoothed_group_peak_r"],
        color="#d1495b",
        alpha=0.75,
        label="Original peak",
    )
    ax.scatter(
        np.log10(ranking.loc[valid, "voxel_count"]),
        ranking.loc[valid, "smoothed_group_peak_r"],
        color="#15616d",
        alpha=0.75,
        label="100-voxel-balanced peak",
    )
    ax.set_xlabel("log10(AAL3 voxel count)")
    ax.set_ylabel("Smoothed group peak fusion r")
    ax.legend(frameon=False)
    ax.set_title("Why voxel-count balancing matters")
    fig.tight_layout()
    fig.savefig(figures / "voxel_count_vs_peak_before_after_balancing.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_dir.resolve()
    parent = project.parent
    source_dir = project / "aal3_fusion_connectivity_outputs"
    output_dir = project / "top_roi_selection_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fusion_path = source_dir / "all_trials_aal3_roi_eeg_fmri_fusion_timeseries.npz"
    with np.load(all_fusion_path, allow_pickle=True) as z:
        unbalanced_fusion = z["fusion_all"].astype(np.float32)
        roi_names = list(z["roi_names"].astype(str))
        times_ms = z["times_ms"].astype(float)

    with np.load(source_dir / "all_trials_eeg_rdm_vectors.npz") as z:
        eeg_all = z["eeg_vecs_all"].astype(np.float32)
    with np.load(source_dir / "condition_eeg_rdm_vectors.npz") as z:
        eeg_condition = z["eeg_vecs"].astype(np.float32)

    voxel_table = pd.read_csv(source_dir / "tables" / "aal3_roi_voxel_counts.csv").set_index("roi")
    voxel_counts = voxel_table.loc[roi_names, "voxel_count"].to_numpy(dtype=int)

    template_path = parent / "IAPS_fMRI_RSA" / "fMRI_singletrial_betas" / "nifti" / "Nt1.img"
    fmri_path = parent / "IAPS_Searchlight" / "allsub_avg.npy"
    atlas_dir = parent / "data" / "masks" / "AAL3" / "new"
    template_img = nib.load(str(template_path))
    masks, mask_counts = load_resampled_masks(roi_names, atlas_dir, template_img)
    if int(np.prod(template_img.shape[:3])) != np.load(fmri_path, mmap_mode="r").shape[-1]:
        raise ValueError("Template grid does not match allsub_avg voxel dimension")
    if any(mask_counts[name] != voxel_counts[i] for i, name in enumerate(roi_names)):
        raise ValueError("Recomputed AAL3 mask counts do not match the existing notebook output")

    balanced_rdm_path = output_dir / "balanced100_all_trials_fmri_rdm_vectors.npz"
    balanced_rdms = compute_balanced_fmri_rdms(
        np.load(fmri_path, mmap_mode="r"),
        roi_names,
        masks,
        balanced_rdm_path,
        n_voxels=N_VOXELS,
        n_repeats=N_VOXEL_REPEATS,
        seed=RANDOM_SEED,
    )

    balanced_fusion_path = output_dir / "balanced100_all_trials_fusion_timeseries.npz"
    condition_fusion_path = output_dir / "balanced100_condition_fusion_timeseries.npz"
    fusion_cache_valid = False
    if balanced_fusion_path.exists() and condition_fusion_path.exists():
        with np.load(balanced_fusion_path) as z:
            version = int(z["analysis_version"]) if "analysis_version" in z else 0
            if version == ANALYSIS_VERSION:
                balanced_fusion = z["fusion_all_balanced"]
                fusion_cache_valid = True
        with np.load(condition_fusion_path) as z:
            version = int(z["analysis_version"]) if "analysis_version" in z else 0
            if version == ANALYSIS_VERSION and fusion_cache_valid:
                balanced_condition = z["fusion_condition_balanced"]
            else:
                fusion_cache_valid = False
    if not fusion_cache_valid:
        balanced_fusion = fuse_rdms(balanced_rdms, eeg_all)
        balanced_condition = condition_fusion_from_all_trial_rdms(balanced_rdms, eeg_condition)
        np.savez_compressed(
            balanced_fusion_path,
            fusion_all_balanced=balanced_fusion,
            roi_names=np.asarray(roi_names),
            times_ms=times_ms,
            n_voxels=np.asarray(N_VOXELS),
            n_repeats=np.asarray(N_VOXEL_REPEATS),
            analysis_version=np.asarray(ANALYSIS_VERSION),
        )
        np.savez_compressed(
            condition_fusion_path,
            fusion_condition_balanced=balanced_condition,
            roi_names=np.asarray(roi_names),
            condition_names=np.asarray(["pleasant", "neutral", "unpleasant"]),
            times_ms=times_ms,
            analysis_version=np.asarray(ANALYSIS_VERSION),
        )

    balanced_ranking, smoothed_balanced, _ = summarize_fusion(
        balanced_fusion,
        balanced_condition,
        times_ms,
        roi_names,
        voxel_counts,
        n_permutations=args.n_permutations,
        seed=RANDOM_SEED + 1,
    )
    with np.load(balanced_rdm_path) as z:
        balanced_ranking["eligible_voxel_count"] = z["eligible_voxel_counts"].astype(int)
    balanced_ranking["balanced_100voxel_eligible"] = np.isfinite(balanced_fusion).all(
        axis=(0, 2)
    )

    with np.load(source_dir / "aal3_roi_eeg_fmri_fusion_timeseries.npz") as z:
        unbalanced_condition = z["fusion_ts"].astype(np.float32)
    unbalanced_ranking, _, _ = summarize_fusion(
        unbalanced_fusion,
        unbalanced_condition,
        times_ms,
        roi_names,
        voxel_counts,
        n_permutations=max(500, args.n_permutations // 5),
        seed=RANDOM_SEED + 2,
    )
    unbalanced_keep = [
        "roi",
        "raw_group_peak_r",
        "smoothed_group_peak_r",
        "peak_time_ms",
        "robust_data_score",
        "robust_data_rank",
    ]
    unbalanced_rename = {c: f"unbalanced_{c}" for c in unbalanced_keep if c != "roi"}
    ranking = balanced_ranking.merge(
        unbalanced_ranking[unbalanced_keep].rename(columns=unbalanced_rename), on="roi", how="left"
    )

    bootstrap_frequency = bootstrap_top10_frequency(
        smoothed_balanced,
        times_ms,
        n_bootstraps=args.n_bootstraps,
        seed=RANDOM_SEED + 3,
    )
    ranking["bootstrap_top10_frequency"] = bootstrap_frequency
    ranking["roi_index"] = np.arange(len(ranking), dtype=int)
    ranking = ranking.sort_values(["robust_data_rank", "roi"], na_position="last").reset_index(drop=True)

    rho_original = spearmanr(
        ranking["voxel_count"], ranking["unbalanced_smoothed_group_peak_r"], nan_policy="omit"
    )
    rho_balanced = spearmanr(
        ranking["voxel_count"], ranking["smoothed_group_peak_r"], nan_policy="omit"
    )
    ranking.to_csv(output_dir / "aal3_roi_robust_ranking.csv", index=False)
    ranking.head(10).to_csv(output_dir / "top10_data_driven_aal3_rois.csv", index=False)
    create_figures(output_dir, ranking, smoothed_balanced, times_ms)

    metadata = {
        "input": str(all_fusion_path),
        "subjects": int(unbalanced_fusion.shape[0]),
        "stimuli": 60,
        "time_ms": [float(times_ms[0]), float(times_ms[-1])],
        "roi_count_requested": 65,
        "roi_count_analyzed": len(roi_names),
        "missing_after_resampling": ["ThalRe"],
        "voxel_balance": {"n_voxels": N_VOXELS, "repeats": N_VOXEL_REPEATS},
        "balanced_eligible_roi_count": int(ranking["balanced_100voxel_eligible"].sum()),
        "gaussian_sigma_ms": SMOOTH_SIGMA_MS,
        "permutations": args.n_permutations,
        "bootstraps": args.n_bootstraps,
        "voxel_count_peak_spearman_original": {
            "rho": float(rho_original.statistic),
            "p": float(rho_original.pvalue),
        },
        "voxel_count_peak_spearman_balanced": {
            "rho": float(rho_balanced.statistic),
            "p": float(rho_balanced.pvalue),
        },
        "interpretation": "Discovery ranking of EEG-fMRI representational alignment, not activation or conventional functional connectivity.",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nTop 10 data-driven ROIs")
    print(
        ranking.head(10)[
            [
                "robust_data_rank",
                "roi",
                "smoothed_group_peak_r",
                "peak_time_ms",
                "max_t_fwer_p",
                "bootstrap_top10_frequency",
            ]
        ].to_string(index=False)
    )
    print("\nSaved outputs to", output_dir)


if __name__ == "__main__":
    main()
