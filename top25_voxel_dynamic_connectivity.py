"""Dynamic voxel-to-voxel fusion connectivity for the four top-25% visual ROIs."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


PROJECT = Path(__file__).resolve().parent
MASK_DIR = (
    PROJECT
    / "top_roi_selection_outputs"
    / "harvard_oxford_4roi_top25_peak_fusion_masks"
)
LABEL_MASK = MASK_DIR / "harvard_oxford_4roi_top25_peak_fusion_labels_mni152_3mm.nii.gz"
LABEL_TABLE = MASK_DIR / "harvard_oxford_4roi_top25_peak_fusion_labels.csv"
OUTPUT_DIR = (
    PROJECT
    / "top_roi_selection_outputs"
    / "harvard_oxford_4roi_top25_peak_fusion_connectivity"
)
RESULT_PATH = OUTPUT_DIR / "top25_voxel_dynamic_roi_connectivity.npz"
TIMES_MS = np.arange(1, 576, dtype=float) * 4.0 - 300.0
SHORT_ROI_NAMES = {
    "Surface early visual cortex": "EVC",
    "Dorsal visual stream": "Dorsal",
    "Ventral visual stream": "Ventral",
    "Lateral occipitotemporal cortex": "Lateral OTC",
}


def _window_definition(times_ms: np.ndarray, window_ms: float, step_ms: float):
    dt = float(np.median(np.diff(times_ms)))
    half_width = max(1, int(round(window_ms / (2.0 * dt))))
    step_samples = max(1, int(round(step_ms / dt)))
    centers = np.arange(half_width, len(times_ms) - half_width, step_samples)
    bounds = np.column_stack((centers - half_width, centers + half_width + 1))
    return centers, bounds


def _standardize_voxel_windows(values: np.ndarray) -> np.ndarray:
    """Center and L2-normalize voxel rows so row products equal Pearson r."""
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite].astype(np.float64, copy=True)
    if not len(values):
        return values
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    valid = norms > np.finfo(float).eps
    return values[valid] / norms[valid, None]


def mean_cross_roi_voxel_correlation(
    roi_a: np.ndarray, roi_b: np.ndarray, start: int, stop: int
) -> float:
    """Return the exact raw-r mean over every voxel pair across two ROIs.

    Averaging all entries of Z_a @ Z_b.T is algebraically equivalent to the
    dot product of the two ROIs' mean standardized temporal profiles. This
    avoids materializing a potentially large voxel-by-voxel matrix.
    """
    za = _standardize_voxel_windows(roi_a[:, start:stop])
    zb = _standardize_voxel_windows(roi_b[:, start:stop])
    if not len(za) or not len(zb):
        return np.nan
    return float(za.mean(axis=0) @ zb.mean(axis=0))


def _group_fisher_summary(subject_r: np.ndarray):
    clipped = np.clip(subject_r, -0.999999, 0.999999)
    z = np.arctanh(clipped)
    n = np.sum(np.isfinite(z), axis=0)
    mean_z = np.nanmean(z, axis=0)
    sd_z = np.nanstd(z, axis=0, ddof=1)
    sem_z = np.divide(sd_z, np.sqrt(n), out=np.full_like(sd_z, np.nan), where=n > 1)
    return (
        np.tanh(mean_z),
        np.tanh(mean_z - 1.96 * sem_z),
        np.tanh(mean_z + 1.96 * sem_z),
        np.nanmean(subject_r, axis=0),
        n,
    )


def _save_tables(result: dict, output_dir: Path) -> None:
    group_rows = []
    subject_rows = []
    for edge_idx, edge_name in enumerate(result["edge_names"]):
        for time_idx, time_ms in enumerate(result["window_centers_ms"]):
            group_rows.append(
                {
                    "edge": str(edge_name),
                    "time_ms": float(time_ms),
                    "group_fisher_mean_r": float(result["group_edge_mean_r"][edge_idx, time_idx]),
                    "ci95_low_r": float(result["group_edge_ci95_low_r"][edge_idx, time_idx]),
                    "ci95_high_r": float(result["group_edge_ci95_high_r"][edge_idx, time_idx]),
                    "group_arithmetic_mean_r": float(result["group_edge_raw_mean_r"][edge_idx, time_idx]),
                    "n_subjects": int(result["group_edge_n"][edge_idx, time_idx]),
                }
            )
            for subject_idx, subject_id in enumerate(result["subject_ids"]):
                subject_rows.append(
                    {
                        "subject": str(subject_id),
                        "edge": str(edge_name),
                        "time_ms": float(time_ms),
                        "mean_voxel_pair_r": float(
                            result["subject_edge_time"][subject_idx, edge_idx, time_idx]
                        ),
                    }
                )
    pd.DataFrame(group_rows).to_csv(
        output_dir / "group_roi_pair_dynamic_connectivity.csv", index=False
    )
    pd.DataFrame(subject_rows).to_csv(
        output_dir / "subject_roi_pair_dynamic_connectivity.csv", index=False
    )


def compute_voxel_dynamic_connectivity(
    output_dir: Path = OUTPUT_DIR,
    window_ms: float = 160.0,
    step_ms: float = 20.0,
    force: bool = False,
) -> dict:
    """Compute or load subject-level and group dynamic ROI connectivity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / RESULT_PATH.name
    if result_path.exists() and not force:
        cached = load_connectivity_result(result_path)
        cached_window = float(cached["window_ms"])
        cached_step = float(cached["step_ms"])
        if np.isclose(cached_window, window_ms) and np.isclose(cached_step, step_ms):
            print("Loaded cached voxel connectivity:", result_path)
            return cached

    mask_img = nib.load(str(LABEL_MASK))
    labels = np.asarray(mask_img.dataobj).astype(np.int16)
    table = pd.read_csv(LABEL_TABLE).sort_values("mask_value")
    roi_labels = table["mask_value"].to_numpy(dtype=int)
    roi_names = table["visual_system_role"].to_numpy(dtype=str)
    roi_atlas_names = table["harvard_oxford_label"].to_numpy(dtype=str)
    roi_masks = [labels == label for label in roi_labels]
    if any(mask.sum() == 0 for mask in roi_masks):
        raise ValueError("At least one selected ROI is empty")

    subject_paths = sorted(PROJECT.glob("rsa_subject[0-9][0-9].nii.gz"))
    if not subject_paths:
        raise FileNotFoundError("No rsa_subjectXX.nii.gz fusion files found")
    subject_ids = np.asarray(
        [path.stem.replace(".nii", "") for path in subject_paths], dtype=str
    )
    centers, bounds = _window_definition(TIMES_MS, window_ms, step_ms)
    edge_pairs = np.asarray(list(combinations(range(len(roi_labels)), 2)), dtype=int)
    edge_names = np.asarray(
        [f"{roi_names[a]} -- {roi_names[b]}" for a, b in edge_pairs], dtype=str
    )

    subject_roi_time = np.full(
        (len(subject_paths), len(roi_labels), len(TIMES_MS)), np.nan, dtype=np.float32
    )
    subject_edge_time = np.full(
        (len(subject_paths), len(edge_pairs), len(centers)), np.nan, dtype=np.float32
    )
    baseline = TIMES_MS < 0

    for subject_idx, subject_path in enumerate(subject_paths):
        print(f"Subject {subject_idx + 1:02d}/{len(subject_paths)}: {subject_path.name}")
        image = nib.load(str(subject_path))
        if image.shape != (*labels.shape, len(TIMES_MS)):
            raise ValueError(f"Unexpected shape for {subject_path.name}: {image.shape}")
        if not np.allclose(image.affine, mask_img.affine):
            raise ValueError(f"Grid mismatch for {subject_path.name}")
        volume = np.asarray(image.dataobj, dtype=np.float32)
        roi_voxels = []
        for roi_idx, mask in enumerate(roi_masks):
            values = volume[mask, :].astype(np.float32, copy=True)
            values -= np.nanmean(values[:, baseline], axis=1, keepdims=True)
            roi_voxels.append(values)
            subject_roi_time[subject_idx, roi_idx] = np.nanmean(values, axis=0)
        del volume

        for edge_idx, (roi_a, roi_b) in enumerate(edge_pairs):
            for window_idx, (start, stop) in enumerate(bounds):
                subject_edge_time[subject_idx, edge_idx, window_idx] = (
                    mean_cross_roi_voxel_correlation(
                        roi_voxels[roi_a], roi_voxels[roi_b], int(start), int(stop)
                    )
                )

    group_edge_mean_r, ci_low, ci_high, raw_mean, group_n = _group_fisher_summary(
        subject_edge_time
    )
    roi_n = np.sum(np.isfinite(subject_roi_time), axis=0)
    group_roi_mean = np.nanmean(subject_roi_time, axis=0)
    group_roi_sem = np.divide(
        np.nanstd(subject_roi_time, axis=0, ddof=1),
        np.sqrt(roi_n),
        out=np.full_like(group_roi_mean, np.nan),
        where=roi_n > 1,
    )

    result = {
        "subject_ids": subject_ids,
        "times_ms": TIMES_MS,
        "window_centers_ms": TIMES_MS[centers],
        "window_bounds_indices": bounds,
        "window_ms": np.asarray(window_ms),
        "step_ms": np.asarray(step_ms),
        "roi_labels": roi_labels,
        "roi_names": roi_names,
        "roi_atlas_names": roi_atlas_names,
        "roi_voxel_counts": np.asarray([mask.sum() for mask in roi_masks]),
        "edge_pairs": edge_pairs,
        "edge_names": edge_names,
        "subject_roi_time": subject_roi_time,
        "group_roi_mean": group_roi_mean,
        "group_roi_sem": group_roi_sem,
        "subject_edge_time": subject_edge_time,
        "group_edge_mean_r": group_edge_mean_r.astype(np.float32),
        "group_edge_ci95_low_r": ci_low.astype(np.float32),
        "group_edge_ci95_high_r": ci_high.astype(np.float32),
        "group_edge_raw_mean_r": raw_mean.astype(np.float32),
        "group_edge_n": group_n.astype(np.int16),
    }
    np.savez_compressed(result_path, **result)
    _save_tables(result, output_dir)
    metadata = {
        "method": "Sliding-window mean cross-ROI voxel-pair Pearson correlation of fusion trajectories",
        "within_subject_aggregation": "Exact arithmetic mean of raw Pearson r across every cross-ROI voxel pair",
        "group_aggregation": "Fisher z mean across subjects, transformed back to r",
        "window_ms": window_ms,
        "step_ms": step_ms,
        "baseline_ms": [float(TIMES_MS[0]), -4.0],
        "subjects": subject_ids.tolist(),
        "mask": str(LABEL_MASK),
        "warning": "The top-25% masks were selected from the same group fusion dataset; confirmatory inference requires independent or cross-validated ROI selection.",
    }
    (output_dir / "top25_voxel_dynamic_roi_connectivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Saved:", result_path)
    return result


def load_connectivity_result(path: Path = RESULT_PATH) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _smooth_curve(values: np.ndarray, times_ms: np.ndarray, sigma_ms: float) -> np.ndarray:
    if sigma_ms <= 0:
        return np.asarray(values)
    dt = float(np.median(np.diff(times_ms)))
    return gaussian_filter1d(values, sigma=sigma_ms / dt, axis=-1, mode="nearest")


def _resolve_roi_indices(result: dict, selected_labels=(1, 2, 3)) -> list[int]:
    label_to_index = {int(label): idx for idx, label in enumerate(result["roi_labels"])}
    if len(selected_labels) != 3 or 1 not in selected_labels:
        raise ValueError("Choose exactly three ROI labels and include label 1 (surface EVC)")
    try:
        return [label_to_index[int(label)] for label in selected_labels]
    except KeyError as exc:
        raise ValueError(f"Unknown ROI label: {exc.args[0]}") from exc


def plot_three_roi_state_space(
    result: dict,
    selected_labels=(1, 2, 3),
    gaussian_sigma_ms: float = 20.0,
    output_path: Path | None = None,
):
    """Plot ROI responses, edge time courses, and two complementary 3D states."""
    selected = _resolve_roi_indices(result, selected_labels)
    full_names = [str(result["roi_names"][idx]) for idx in selected]
    names = [SHORT_ROI_NAMES.get(name, name) for name in full_names]
    times = result["times_ms"]
    centers = result["window_centers_ms"]
    roi_mean = _smooth_curve(result["group_roi_mean"], times, gaussian_sigma_ms)
    roi_sem = result["group_roi_sem"]
    edge_mean = _smooth_curve(result["group_edge_mean_r"], centers, gaussian_sigma_ms)
    edge_low = _smooth_curve(result["group_edge_ci95_low_r"], centers, gaussian_sigma_ms)
    edge_high = _smooth_curve(result["group_edge_ci95_high_r"], centers, gaussian_sigma_ms)

    edge_lookup = {
        tuple(sorted((int(a), int(b)))): edge_idx
        for edge_idx, (a, b) in enumerate(result["edge_pairs"])
    }
    selected_edge_indices = [
        edge_lookup[tuple(sorted(pair))] for pair in combinations(selected, 2)
    ]
    selected_edge_names = [f"{names[a]} -- {names[b]}" for a, b in combinations(range(3), 2)]

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax_roi = fig.add_subplot(grid[0, 0])
    ax_edge = fig.add_subplot(grid[0, 1])
    ax_state = fig.add_subplot(grid[1, 0], projection="3d")
    ax_edge_state = fig.add_subplot(grid[1, 1], projection="3d")
    colors = ["#1677b8", "#e68613", "#2b9b59"]

    post = times >= 0
    for color, roi_idx, name in zip(colors, selected, names):
        ax_roi.plot(times[post], roi_mean[roi_idx, post], color=color, lw=2, label=name)
        ax_roi.fill_between(
            times[post],
            roi_mean[roi_idx, post] - roi_sem[roi_idx, post],
            roi_mean[roi_idx, post] + roi_sem[roi_idx, post],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax_roi.axhline(0, color="0.4", lw=0.8)
    ax_roi.set(title="Group ROI fusion responses", xlabel="Time (ms)", ylabel="Baseline-corrected fusion r")
    ax_roi.legend(frameon=False, fontsize=9)

    edge_post = centers >= 0
    for color, edge_idx, edge_name in zip(colors, selected_edge_indices, selected_edge_names):
        ax_edge.plot(centers[edge_post], edge_mean[edge_idx, edge_post], color=color, lw=2, label=edge_name)
        ax_edge.fill_between(
            centers[edge_post],
            edge_low[edge_idx, edge_post],
            edge_high[edge_idx, edge_post],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax_edge.axhline(0, color="0.4", lw=0.8)
    ax_edge.set(title="Dynamic voxel-pair ROI connectivity", xlabel="Window center (ms)", ylabel="Group mean r")
    ax_edge.legend(frameon=False, fontsize=9, loc="upper right")

    state_indices = np.flatnonzero(post)[::5]
    x, y, z = [roi_mean[idx, state_indices] for idx in selected]
    state_time = times[state_indices]
    ax_state.plot(x, y, z, color="0.55", lw=1.0, alpha=0.8)
    state_scatter = ax_state.scatter(x, y, z, c=state_time, cmap="viridis", s=16)
    ax_state.scatter(x[0], y[0], z[0], color="black", marker="o", s=45, label="0 ms")
    ax_state.scatter(x[-1], y[-1], z[-1], color="black", marker="X", s=55, label="2000 ms")
    ax_state.set(
        title="Three-ROI representational state trajectory",
        xlabel=names[0],
        ylabel=names[1],
        zlabel=names[2],
    )
    ax_state.legend(frameon=False, fontsize=8)
    fig.colorbar(state_scatter, ax=ax_state, shrink=0.62, pad=0.12, label="Time (ms)")

    edge_state_indices = np.flatnonzero(edge_post)
    ex, ey, ez = [edge_mean[idx, edge_state_indices] for idx in selected_edge_indices]
    edge_time = centers[edge_state_indices]
    ax_edge_state.plot(ex, ey, ez, color="0.55", lw=1.0, alpha=0.8)
    edge_scatter = ax_edge_state.scatter(ex, ey, ez, c=edge_time, cmap="plasma", s=18)
    ax_edge_state.set(
        title="Three-edge connectivity state trajectory",
        xlabel=selected_edge_names[0],
        ylabel=selected_edge_names[1],
        zlabel=selected_edge_names[2],
    )
    fig.colorbar(edge_scatter, ax=ax_edge_state, shrink=0.62, pad=0.12, label="Window center (ms)")
    fig.suptitle(
        f"Top-25% fusion ROIs: {names[0]} + {names[1]} + {names[2]}", fontsize=17
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    analysis = compute_voxel_dynamic_connectivity()
    figure = plot_three_roi_state_space(
        analysis,
        selected_labels=(1, 2, 3),
        gaussian_sigma_ms=20.0,
        output_path=OUTPUT_DIR / "state_space_evc_dorsal_ventral.png",
    )
    plt.close(figure)
