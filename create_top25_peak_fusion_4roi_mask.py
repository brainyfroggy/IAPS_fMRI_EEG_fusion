"""Select the top 25% peak-fusion voxels within four Harvard-Oxford ROIs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.plotting import plot_roi
from scipy.ndimage import gaussian_filter1d


PROJECT = Path(__file__).resolve().parent
SOURCE_MASK_DIR = (
    PROJECT
    / "top_roi_selection_outputs"
    / "harvard_oxford_4roi_masks_surface_otc"
)
OUTPUT_DIR = (
    PROJECT
    / "top_roi_selection_outputs"
    / "harvard_oxford_4roi_top25_peak_fusion_masks"
)
FUSION_PATH = PROJECT / "rsa_subject_mean_nonan.nii.gz"
BRAIN_MASK_PATH = PROJECT.parent / "IAPS_Searchlight" / "mask.npy"
SOURCE_LABEL_TABLE = SOURCE_MASK_DIR / "harvard_oxford_4roi_labels.csv"
SOURCE_2MM_REFERENCE = SOURCE_MASK_DIR / "harvard_oxford_4roi_surface_otc_labels_mni152_2mm.nii.gz"

TIMES_MS = np.arange(1, 576) * 4 - 300
SMOOTH_SIGMA_MS = 20.0
TOP_FRACTION = 0.25


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_nifti(data: np.ndarray, reference: nib.Nifti1Image, path: Path, description: str) -> None:
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    header["descrip"] = description[:79]
    image = nib.Nifti1Image(data, reference.affine, header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    image.set_qform(qform if qform is not None else reference.affine, int(qcode or 1))
    image.set_sform(sform if sform is not None else reference.affine, int(scode or 1))
    nib.save(image, str(path))


def compute_peak_map(fusion_img: nib.Nifti1Image) -> np.ndarray:
    if fusion_img.shape != (53, 63, 46, 575):
        raise ValueError(f"Unexpected fusion shape: {fusion_img.shape}")
    data = np.asarray(fusion_img.dataobj, dtype=np.float32)
    baseline = TIMES_MS < 0
    post = TIMES_MS >= 0
    data -= np.mean(data[..., baseline], axis=-1, keepdims=True)
    sigma_samples = SMOOTH_SIGMA_MS / float(np.median(np.diff(TIMES_MS)))
    gaussian_filter1d(data, sigma=sigma_samples, axis=-1, output=data, mode="nearest")
    peak = np.max(data[..., post], axis=-1)
    del data
    return peak.astype(np.float32)


def exact_top_fraction(
    peak_map: np.ndarray, roi_mask: np.ndarray, fraction: float
) -> tuple[np.ndarray, float, int]:
    flat_indices = np.flatnonzero(roi_mask.ravel())
    values = peak_map.ravel()[flat_indices]
    valid = np.isfinite(values)
    flat_indices = flat_indices[valid]
    values = values[valid]
    if not len(values):
        raise ValueError("ROI has no finite fusion voxels")
    n_keep = max(1, int(np.ceil(len(values) * fraction)))
    order = np.argsort(values, kind="stable")[::-1]
    selected_flat = flat_indices[order[:n_keep]]
    selected = np.zeros(roi_mask.size, dtype=bool)
    selected[selected_flat] = True
    selected = selected.reshape(roi_mask.shape)
    threshold = float(values[order[n_keep - 1]])
    return selected, threshold, n_keep


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fusion_img = nib.load(str(FUSION_PATH))
    reference_2mm = nib.load(str(SOURCE_2MM_REFERENCE))
    brain_mask = np.asarray(np.load(BRAIN_MASK_PATH), dtype=bool)
    if brain_mask.shape != fusion_img.shape[:3]:
        raise ValueError("Brain mask and fusion grid do not match")

    print("Computing baseline-corrected, smoothed post-stimulus peak map")
    peak_map = compute_peak_map(fusion_img)
    peak_map[~brain_mask] = np.nan
    save_nifti(
        np.nan_to_num(peak_map, nan=0.0).astype(np.float32),
        fusion_img,
        OUTPUT_DIR / "group_fusion_peak_baseline_corrected_smoothed20ms_mni152_3mm.nii.gz",
        "Group fusion peak; baseline corrected; Gaussian sigma 20ms; poststimulus",
    )

    source_table = pd.read_csv(SOURCE_LABEL_TABLE).sort_values("mask_value")
    selected_labels_3mm = np.zeros(fusion_img.shape[:3], dtype=np.int16)
    selected_binary_3mm = np.zeros(fusion_img.shape[:3], dtype=np.uint8)
    rows = []

    for row in source_table.itertuples():
        source_path = SOURCE_MASK_DIR / row.individual_mask_file
        source_img = nib.load(str(source_path))
        roi_native_img = resample_to_img(
            source_img,
            fusion_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        roi_native = (np.asarray(roi_native_img.dataobj) > 0) & brain_mask
        selected, threshold, n_keep = exact_top_fraction(peak_map, roi_native, TOP_FRACTION)
        if np.any(selected_labels_3mm[selected] != 0):
            raise ValueError(f"Selected ROI overlap: {row.harvard_oxford_label}")
        selected_labels_3mm[selected] = int(row.mask_value)
        selected_binary_3mm[selected] = 1

        individual_3mm_name = f"{int(row.mask_value):02d}_{slugify(row.visual_system_role)}_top25_mni152_3mm.nii.gz"
        save_nifti(
            selected.astype(np.uint8),
            fusion_img,
            OUTPUT_DIR / individual_3mm_name,
            f"Top 25pct peak fusion: {row.harvard_oxford_label}; MNI152 3mm",
        )
        rows.append(
            {
                "mask_value": int(row.mask_value),
                "visual_system_role": row.visual_system_role,
                "harvard_oxford_label": row.harvard_oxford_label,
                "native_roi_voxel_count_3mm": int(roi_native.sum()),
                "selected_voxel_count_3mm": n_keep,
                "selected_fraction_3mm": n_keep / int(roi_native.sum()),
                "minimum_selected_peak_r": threshold,
                "maximum_selected_peak_r": float(np.nanmax(peak_map[selected])),
                "individual_top25_mask_3mm": individual_3mm_name,
            }
        )

    labels_3mm_path = OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_labels_mni152_3mm.nii.gz"
    binary_3mm_path = OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_binary_mni152_3mm.nii.gz"
    save_nifti(
        selected_labels_3mm,
        fusion_img,
        labels_3mm_path,
        "Four-label top 25pct peak EEG-fMRI fusion mask; MNI152 3mm",
    )
    save_nifti(
        selected_binary_3mm,
        fusion_img,
        binary_3mm_path,
        "Binary union top 25pct peak EEG-fMRI fusion mask; MNI152 3mm",
    )

    labels_2mm_img = resample_to_img(
        nib.load(str(labels_3mm_path)),
        reference_2mm,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    labels_2mm = np.rint(np.asarray(labels_2mm_img.dataobj)).astype(np.int16)
    binary_2mm = (labels_2mm > 0).astype(np.uint8)
    labels_2mm_path = OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_labels_mni152_2mm.nii.gz"
    binary_2mm_path = OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_binary_mni152_2mm.nii.gz"
    save_nifti(
        labels_2mm,
        reference_2mm,
        labels_2mm_path,
        "Four-label top 25pct peak EEG-fMRI fusion mask; resampled MNI152 2mm",
    )
    save_nifti(
        binary_2mm,
        reference_2mm,
        binary_2mm_path,
        "Binary union top 25pct peak EEG-fMRI fusion mask; resampled MNI152 2mm",
    )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_labels.csv", index=False)
    metadata = {
        "source_fusion": str(FUSION_PATH),
        "source_fusion_definition": "Group mean, all 60 averaged IAPS stimulus RDM fusion",
        "peak_definition": "Maximum from 0 to 2000 ms after subtracting -296 to -4 ms baseline and Gaussian smoothing with sigma 20 ms",
        "selection": "Exact top ceil(25%) of finite native 3 mm fusion voxels separately within each ROI",
        "top_fraction": TOP_FRACTION,
        "times_ms": [int(TIMES_MS[0]), int(TIMES_MS[-1])],
        "native_output": str(labels_3mm_path),
        "resampled_output": str(labels_2mm_path),
        "warning": "Data-driven discovery mask. Testing peak fusion on the same data used for selection is circular.",
        "labels": rows,
    }
    (OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    qc_colors = ["#1677b8", "#e68613", "#2b9b59", "#c24c9a"]
    qc_titles = [
        "Surface EVC: Occipital Pole",
        "Dorsal: Superior Parietal Lobule",
        "Ventral: Temporal Occipital Fusiform",
        "Lateral OTC: Lateral Occipital (inferior)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Top 25% peak EEG-fMRI fusion voxels within four Harvard-Oxford ROIs",
        fontsize=19,
    )
    for label, (ax, color, title) in enumerate(
        zip(axes.ravel(), qc_colors, qc_titles), start=1
    ):
        roi_data = (labels_2mm == label).astype(np.uint8)
        ijk_center = np.argwhere(roi_data).mean(axis=0)
        xyz_center = nib.affines.apply_affine(reference_2mm.affine, ijk_center)
        roi_img = nib.Nifti1Image(roi_data, reference_2mm.affine, reference_2mm.header)
        plot_roi(
            roi_img,
            display_mode="ortho",
            cut_coords=tuple(xyz_center),
            cmap=ListedColormap([color]),
            alpha=0.9,
            colorbar=False,
            draw_cross=False,
            annotate=True,
            title=title,
            axes=ax,
            figure=fig,
        )
    fig.subplots_adjust(top=0.91, hspace=0.18, wspace=0.05)
    fig.savefig(
        OUTPUT_DIR / "harvard_oxford_4roi_top25_peak_fusion_qc.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    print(table.to_string(index=False))
    print("Saved native label mask:", labels_3mm_path)
    print("Saved MNI152 2 mm label mask:", labels_2mm_path)


if __name__ == "__main__":
    main()
