"""Top-percent ROI sensitivity analysis for IAPS EEG-fMRI RSA fusion.

This script follows the all-stimulus fusion logic in analysis_main_clean_combined.ipynb:
for each subject, compute an fMRI correlation-distance RDM inside an ROI, then
Pearson-correlate its upper triangle with the subject's time-resolved EEG RDM.

The sensitivity analysis keeps the raw-VA behavioral ROI fixed, and recomputes
CLIP and GIST fusion curves after keeping the top 100%, 90%, ..., 10% positive
t-valued voxels from their fdr01 t-maps.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_fMRI_EEG_fusion")
SEARCHLIGHT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight")
FUSION_DIR = PROJECT_DIR / "Searchlight ROI EEG"
OUT_DIR = FUSION_DIR / "sensitivity_top_percent_tmaps_20260810"

ALLSUB_AVG_PATH = SEARCHLIGHT_DIR / "allsub_avg.npy"
EEG_RDM_PATH = PROJECT_DIR / "aal3_fusion_connectivity_outputs" / "all_trials_eeg_rdm_vectors.npz"
TMAP_DIR = SEARCHLIGHT_DIR / "outputs" / "tBrainmap" / "final_results2"

ROI_PATHS = {
    "Behavioral raw VA": TMAP_DIR / "tmap_beh60_raw_euclidean_fdr01_cluster30_motor_removed.nii.gz",
    "CLIP": TMAP_DIR / "tmap_clip_euclidean_l2norm_fdr01.nii.gz",
    "GIST": TMAP_DIR / "tmap_gist_l2norm_euclidean_fdr01.nii.gz",
}

PERCENTAGES = tuple(range(100, 0, -10))
BASELINE_WINDOW = (-200, 0)
TEST_WINDOW = (0, 1200)
SMOOTH_SIGMA_SAMPLES = 10
THRESHOLD_SIGMA = 5
MIN_CONSECUTIVE_SAMPLES = 25  # 100 ms at 4 ms sampling
X_LIM = (-200, 1200)


def read_nifti_data(path: Path) -> np.ndarray:
    raw = gzip.open(path, "rb").read()
    if struct.unpack("<i", raw[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", raw[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"Not a NIfTI-1 image: {path}")

    dim = struct.unpack(endian + "8h", raw[40:56])
    shape = tuple(int(v) for v in dim[1 : 1 + dim[0]])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    bitpix = struct.unpack(endian + "h", raw[72:74])[0]
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    dtypes = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        256: np.int8,
        512: np.uint16,
        768: np.uint32,
    }
    dtype = np.dtype(dtypes[datatype]).newbyteorder(endian)
    count = int(np.prod(shape))
    data = np.frombuffer(
        raw[vox_offset : vox_offset + count * (bitpix // 8)],
        dtype=dtype,
        count=count,
    ).copy()
    return data.reshape(shape, order="F")


def ranked_positive_voxels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = read_nifti_data(path)
    flat = img.ravel(order="C")
    idx = np.where(np.isfinite(flat) & (flat > 0))[0]
    values = flat[idx]
    order = np.lexsort((idx, -values))
    return idx[order].astype(np.int64), values[order].astype(float)


def normalize_rows_for_pearson(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x, axis=-1, keepdims=True)
    norm = np.sqrt(np.nansum(x * x, axis=-1, keepdims=True))
    norm[norm == 0] = np.nan
    return x / norm


def gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_smooth(curve: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return curve.copy()
    pad = kernel.size // 2
    padded = np.pad(curve, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def upper_triangle_vector(square: np.ndarray) -> np.ndarray:
    r, c = np.triu_indices(square.shape[0], 1)
    return square[r, c]


def correlation_distance_from_prefix_stats(
    gram: np.ndarray, sums: np.ndarray, sums_sq: np.ndarray, n_voxels: int
) -> np.ndarray:
    cov = gram - np.outer(sums, sums) / float(n_voxels)
    var = sums_sq - (sums * sums) / float(n_voxels)
    var[var < 0] = 0
    denom = np.sqrt(np.outer(var, var))
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_fusion_for_ranked_prefixes(
    allsub_avg: np.ndarray,
    ranked_idx: np.ndarray,
    counts_by_pct: dict[int, int],
    eeg_norm: np.ndarray,
    label: str,
) -> np.ndarray:
    n_subjects = allsub_avg.shape[0]
    n_time = eeg_norm.shape[1]
    pct_ascending = sorted(counts_by_pct, key=lambda p: counts_by_pct[p])
    max_count = max(counts_by_pct.values())
    fusion = np.full((len(PERCENTAGES), n_subjects, n_time), np.nan, dtype=np.float32)
    pct_to_out_index = {pct: i for i, pct in enumerate(PERCENTAGES)}

    tri = np.triu_indices(60, 1)
    for subject in range(n_subjects):
        print(f"{label}: subject {subject + 1:02d}/{n_subjects}")
        # Slice subject first so advanced voxel indexing does not move the voxel
        # dimension ahead of the stimulus dimension.
        patterns = np.asarray(allsub_avg[subject][:, ranked_idx[:max_count]], dtype=np.float64)
        if not np.isfinite(patterns).all():
            patterns = np.nan_to_num(patterns, nan=0.0, posinf=0.0, neginf=0.0)

        gram = np.zeros((60, 60), dtype=np.float64)
        sums = np.zeros(60, dtype=np.float64)
        sums_sq = np.zeros(60, dtype=np.float64)
        prev = 0
        for pct in pct_ascending:
            count = counts_by_pct[pct]
            block = patterns[:, prev:count]
            gram += block @ block.T
            sums += block.sum(axis=1)
            sums_sq += np.sum(block * block, axis=1)
            prev = count

            dist = correlation_distance_from_prefix_stats(gram, sums, sums_sq, count)
            brain_vec = dist[tri]
            brain_vec_norm = normalize_rows_for_pearson(brain_vec)
            fusion[pct_to_out_index[pct], subject] = eeg_norm[subject] @ brain_vec_norm
    return fusion


def compute_group_curve(rsa_data: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray | float]:
    baseline = (times_ms >= BASELINE_WINDOW[0]) & (times_ms < BASELINE_WINDOW[1])
    test = (times_ms >= TEST_WINDOW[0]) & (times_ms <= TEST_WINDOW[1])
    subj_base = np.nanmean(rsa_data[:, baseline], axis=1, keepdims=True)
    rsa_bc = rsa_data - subj_base
    mean_curve = np.nanmean(rsa_bc, axis=0)
    sem_curve = np.nanstd(rsa_bc, axis=0, ddof=1) / math.sqrt(rsa_bc.shape[0])
    smoothed = gaussian_smooth(mean_curve, SMOOTH_SIGMA_SAMPLES)
    baseline_values = smoothed[baseline]
    threshold = float(np.nanmean(baseline_values) + THRESHOLD_SIGMA * np.nanstd(baseline_values, ddof=1))
    sig = (smoothed > threshold) & (smoothed > 0) & test
    onset = np.nan
    run = 0
    for i, is_sig in enumerate(sig):
        run = run + 1 if is_sig else 0
        if run >= MIN_CONSECUTIVE_SAMPLES:
            onset = float(times_ms[i - MIN_CONSECUTIVE_SAMPLES + 1])
            break
    return {
        "mean_bc": mean_curve,
        "sem_bc": sem_curve,
        "smoothed_bc": smoothed,
        "threshold": threshold,
        "onset_ms": onset,
    }


def blend(color1: tuple[int, int, int], color2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(color1, color2))


def draw_sensitivity_figure(
    out_png: Path,
    times_ms: np.ndarray,
    behavior_curve: np.ndarray,
    clip_curves: dict[int, np.ndarray],
    gist_curves: dict[int, np.ndarray],
    counts: dict[str, dict[int, int] | int],
) -> None:
    width, height = 2200, 2600
    margin_left, margin_right = 180, 520
    margin_top, margin_bottom = 230, 150
    panel_gap = 120
    plot_w = width - margin_left - margin_right
    panel_h = (height - margin_top - margin_bottom - 2 * panel_gap) // 3
    bg = "white"
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    bold_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 34)
    small = ImageFont.truetype(str(font_path), 26)
    tiny = ImageFont.truetype(str(font_path), 23)
    title_font = ImageFont.truetype(str(bold_path), 44)
    panel_font = ImageFont.truetype(str(bold_path), 34)

    draw.text((margin_left, 45), "EEG-fMRI fusion sensitivity to top-t ROI size", fill=(20, 20, 20), font=title_font)
    draw.text(
        (margin_left, 105),
        "CLIP and GIST curves keep the top percentage of positive fdr01 t-map voxels; behavioral raw-VA ROI is fixed.",
        fill=(80, 80, 80),
        font=small,
    )

    x_min, x_max = X_LIM
    visible = (times_ms >= x_min) & (times_ms <= x_max)
    xticks = [-200, 0, 200, 400, 600, 800, 1000, 1200]

    red_light, red_dark = (255, 205, 197), (155, 0, 0)
    blue_light, blue_dark = (198, 224, 255), (0, 63, 142)
    green = (28, 145, 77)

    panels = [
        ("Behavioral raw VA ROI", {100: behavior_curve}, {100: green}, {100: int(counts["behavior"])}),
        (
            "CLIP ROI top-t sensitivity",
            clip_curves,
            {pct: blend(red_light, red_dark, (100 - pct) / 90.0) for pct in PERCENTAGES},
            counts["clip"],
        ),
        (
            "GIST ROI top-t sensitivity",
            gist_curves,
            {pct: blend(blue_light, blue_dark, (100 - pct) / 90.0) for pct in PERCENTAGES},
            counts["gist"],
        ),
    ]

    global_y = []
    for _, curves, _, _ in panels:
        for curve in curves.values():
            global_y.extend(curve[visible])
    y_min = float(np.nanmin(global_y))
    y_max = float(np.nanmax(global_y))
    pad = (y_max - y_min) * 0.12
    y_min -= pad
    y_max += pad

    def sx(x: float) -> int:
        return int(margin_left + (x - x_min) / (x_max - x_min) * plot_w)

    def sy(y: float, top: int) -> int:
        return int(top + panel_h - (y - y_min) / (y_max - y_min) * panel_h)

    for pi, (panel_title, curves, colors, panel_counts) in enumerate(panels):
        top = margin_top + pi * (panel_h + panel_gap)
        bottom = top + panel_h
        draw.text((margin_left, top - 48), panel_title, fill=(25, 25, 25), font=panel_font)

        for xt in xticks:
            x = sx(xt)
            draw.line((x, top, x, bottom), fill=(230, 230, 230), width=1)
            draw.text((x - 30, bottom + 12), str(xt), fill=(80, 80, 80), font=tiny)
        for yt in np.linspace(y_min, y_max, 5):
            y = sy(float(yt), top)
            draw.line((margin_left, y, margin_left + plot_w, y), fill=(235, 235, 235), width=1)
            draw.text((40, y - 13), f"{yt:.3f}", fill=(80, 80, 80), font=tiny)
        draw.text((margin_left - 145, top + 10), "Fusion r", fill=(80, 80, 80), font=tiny)

        draw.rectangle((margin_left, top, margin_left + plot_w, bottom), outline=(70, 70, 70), width=2)
        draw.line((sx(0), top, sx(0), bottom), fill=(35, 35, 35), width=2)
        y0 = sy(0.0, top)
        draw.line((margin_left, y0, margin_left + plot_w, y0), fill=(90, 90, 90), width=1)

        for pct in sorted(curves, reverse=True):
            curve = curves[pct]
            pts = [(sx(float(x)), sy(float(y), top)) for x, y in zip(times_ms[visible], curve[visible]) if np.isfinite(y)]
            if len(pts) > 1:
                draw.line(pts, fill=colors[pct], width=5 if pi == 0 else 4)

        legend_x = margin_left + plot_w + 55
        legend_y = top + 10
        if pi == 0:
            draw.line((legend_x, legend_y + 18, legend_x + 62, legend_y + 18), fill=green, width=6)
            draw.text((legend_x + 78, legend_y), f"fixed ROI: {panel_counts[100]} voxels", fill=(30, 30, 30), font=small)
        else:
            for j, pct in enumerate(PERCENTAGES):
                y = legend_y + j * 38
                draw.line((legend_x, y + 15, legend_x + 55, y + 15), fill=colors[pct], width=5)
                draw.text((legend_x + 72, y - 2), f"top {pct:3d}%  ({panel_counts[pct]:5d} voxels)", fill=(35, 35, 35), font=tiny)

    draw.text((margin_left + plot_w // 2 - 100, height - 75), "Time (ms)", fill=(35, 35, 35), font=font)
    image.save(out_png, dpi=(220, 220))
    image.save(out_png.with_suffix(".pdf"), "PDF", resolution=220.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(EEG_RDM_PATH, allow_pickle=True) as z:
        eeg_vecs = z["eeg_vecs_all"].astype(np.float64)
        times_ms = z["times_ms"].astype(float)
        eeg_metric = str(z["metric"])
    eeg_norm = normalize_rows_for_pearson(eeg_vecs)

    allsub_avg = np.load(ALLSUB_AVG_PATH, mmap_mode="r")
    if allsub_avg.shape[:2] != (20, 60):
        raise ValueError(f"Unexpected allsub_avg shape: {allsub_avg.shape}")

    behavior_idx, behavior_t = ranked_positive_voxels(ROI_PATHS["Behavioral raw VA"])
    clip_idx, clip_t = ranked_positive_voxels(ROI_PATHS["CLIP"])
    gist_idx, gist_t = ranked_positive_voxels(ROI_PATHS["GIST"])

    behavior_count = int(len(behavior_idx))
    clip_counts = {pct: max(1, int(round(len(clip_idx) * pct / 100.0))) for pct in PERCENTAGES}
    gist_counts = {pct: max(1, int(round(len(gist_idx) * pct / 100.0))) for pct in PERCENTAGES}

    behavior_fusion = compute_fusion_for_ranked_prefixes(
        allsub_avg,
        behavior_idx,
        {100: behavior_count},
        eeg_norm,
        "Behavioral raw VA",
    )[0]
    clip_fusion = compute_fusion_for_ranked_prefixes(allsub_avg, clip_idx, clip_counts, eeg_norm, "CLIP")
    gist_fusion = compute_fusion_for_ranked_prefixes(allsub_avg, gist_idx, gist_counts, eeg_norm, "GIST")

    behavior_result = compute_group_curve(behavior_fusion, times_ms)
    clip_results = {pct: compute_group_curve(clip_fusion[i], times_ms) for i, pct in enumerate(PERCENTAGES)}
    gist_results = {pct: compute_group_curve(gist_fusion[i], times_ms) for i, pct in enumerate(PERCENTAGES)}

    np.savez_compressed(
        OUT_DIR / "top_percent_fusion_timeseries.npz",
        times_ms=times_ms,
        percentages=np.asarray(PERCENTAGES),
        behavior_fusion=behavior_fusion,
        clip_fusion=clip_fusion,
        gist_fusion=gist_fusion,
        behavior_count=np.asarray(behavior_count),
        clip_counts=np.asarray([clip_counts[p] for p in PERCENTAGES]),
        gist_counts=np.asarray([gist_counts[p] for p in PERCENTAGES]),
        eeg_metric=np.asarray(eeg_metric),
    )

    summary_rows = []
    summary_rows.append(
        {
            "family": "behavior_raw_va",
            "percent_kept": 100,
            "voxel_count": behavior_count,
            "source_positive_voxels": behavior_count,
            "min_kept_t": float(np.nanmin(behavior_t)),
            "max_kept_t": float(np.nanmax(behavior_t)),
            "onset_ms": behavior_result["onset_ms"],
        }
    )
    for pct in PERCENTAGES:
        k = clip_counts[pct]
        summary_rows.append(
            {
                "family": "clip_euclidean_l2norm",
                "percent_kept": pct,
                "voxel_count": k,
                "source_positive_voxels": int(len(clip_idx)),
                "min_kept_t": float(clip_t[k - 1]),
                "max_kept_t": float(clip_t[0]),
                "onset_ms": clip_results[pct]["onset_ms"],
            }
        )
    for pct in PERCENTAGES:
        k = gist_counts[pct]
        summary_rows.append(
            {
                "family": "gist_l2norm_euclidean",
                "percent_kept": pct,
                "voxel_count": k,
                "source_positive_voxels": int(len(gist_idx)),
                "min_kept_t": float(gist_t[k - 1]),
                "max_kept_t": float(gist_t[0]),
                "onset_ms": gist_results[pct]["onset_ms"],
            }
        )

    with (OUT_DIR / "top_percent_roi_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    curve_columns = ["time_ms", "behavior_raw_va"]
    curve_matrix = [times_ms, behavior_result["smoothed_bc"]]
    for pct in PERCENTAGES:
        curve_columns.append(f"clip_top{pct}_vox{clip_counts[pct]}")
        curve_matrix.append(clip_results[pct]["smoothed_bc"])
    for pct in PERCENTAGES:
        curve_columns.append(f"gist_top{pct}_vox{gist_counts[pct]}")
        curve_matrix.append(gist_results[pct]["smoothed_bc"])
    curve_data = np.column_stack(curve_matrix)
    with (OUT_DIR / "top_percent_group_smoothed_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(curve_columns)
        writer.writerows(curve_data.tolist())

    fig_path = OUT_DIR / "fusion_top_percent_sensitivity_three_panel.png"
    draw_sensitivity_figure(
        fig_path,
        times_ms,
        behavior_result["smoothed_bc"],
        {pct: clip_results[pct]["smoothed_bc"] for pct in PERCENTAGES},
        {pct: gist_results[pct]["smoothed_bc"] for pct in PERCENTAGES},
        {"behavior": behavior_count, "clip": clip_counts, "gist": gist_counts},
    )

    metadata = {
        "method": "All-stimulus fMRI ROI RDM to time-resolved EEG RDM fusion.",
        "fusion_correlation": "Pearson correlation between upper-triangle RDM vectors, matching analysis_main_clean_combined.ipynb.",
        "fmri_rdm_metric": "Correlation distance across voxel patterns.",
        "eeg_rdm_metric": eeg_metric,
        "behavior_roi": str(ROI_PATHS["Behavioral raw VA"]),
        "clip_roi_source": str(ROI_PATHS["CLIP"]),
        "gist_roi_source": str(ROI_PATHS["GIST"]),
        "percentages": list(PERCENTAGES),
        "behavior_voxels": behavior_count,
        "clip_source_positive_voxels": int(len(clip_idx)),
        "gist_source_positive_voxels": int(len(gist_idx)),
        "baseline_window_ms": list(BASELINE_WINDOW),
        "smoothing_sigma_samples": SMOOTH_SIGMA_SAMPLES,
        "onset_threshold": f"baseline mean + {THRESHOLD_SIGMA} baseline SD",
        "onset_min_consecutive_samples": MIN_CONSECUTIVE_SAMPLES,
        "outputs": {
            "figure_png": str(fig_path),
            "figure_pdf": str(fig_path.with_suffix(".pdf")),
            "summary_csv": str(OUT_DIR / "top_percent_roi_summary.csv"),
            "curves_csv": str(OUT_DIR / "top_percent_group_smoothed_curves.csv"),
            "timeseries_npz": str(OUT_DIR / "top_percent_fusion_timeseries.npz"),
        },
    }
    (OUT_DIR / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved outputs to", OUT_DIR)
    print("Behavior voxels:", behavior_count)
    print("CLIP source voxels:", len(clip_idx), clip_counts)
    print("GIST source voxels:", len(gist_idx), gist_counts)


if __name__ == "__main__":
    main()
