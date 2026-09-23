"""Create revised bilateral Harvard-Oxford visual-stream masks in MNI152 space."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.plotting import plot_roi


PROJECT = Path(__file__).resolve().parent
OUTPUT_DIR = (
    PROJECT
    / "top_roi_selection_outputs"
    / "harvard_oxford_4roi_masks_surface_otc"
)
ATLAS_CACHE = PROJECT / "top_roi_selection_outputs" / "atlases"
ATLAS_NAME = "cort-maxprob-thr25-2mm"

ROI_DEFINITIONS = [
    (1, "Surface early visual cortex", "Occipital Pole"),
    (2, "Dorsal visual stream", "Superior Parietal Lobule"),
    (3, "Ventral visual stream", "Temporal Occipital Fusiform Cortex"),
    (4, "Lateral occipitotemporal cortex", "Lateral Occipital Cortex, inferior division"),
]


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = fetch_atlas_harvard_oxford(
        ATLAS_NAME,
        data_dir=str(ATLAS_CACHE),
        symmetric_split=False,
    )
    atlas_img = atlas.maps
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int16)
    label_to_index = {str(label): index for index, label in enumerate(atlas.labels)}

    missing = [name for _, _, name in ROI_DEFINITIONS if name not in label_to_index]
    if missing:
        raise ValueError(f"Missing Harvard-Oxford labels: {missing}")

    combined_labels = np.zeros(atlas_data.shape, dtype=np.int16)
    binary_union = np.zeros(atlas_data.shape, dtype=np.uint8)
    rows = []

    for output_value, role, atlas_label in ROI_DEFINITIONS:
        atlas_value = label_to_index[atlas_label]
        roi_mask = atlas_data == atlas_value
        voxel_count = int(roi_mask.sum())
        if voxel_count == 0:
            raise ValueError(f"ROI has no voxels: {atlas_label}")
        if np.any(combined_labels[roi_mask] != 0):
            raise ValueError(f"Unexpected overlap in max-probability atlas: {atlas_label}")

        combined_labels[roi_mask] = output_value
        binary_union[roi_mask] = 1
        individual_name = f"{output_value:02d}_{slugify(role)}.nii.gz"
        save_nifti(
            roi_mask.astype(np.uint8),
            atlas_img,
            OUTPUT_DIR / individual_name,
            f"Harvard-Oxford {atlas_label}; bilateral; MNI152 2mm",
        )
        rows.append(
            {
                "mask_value": output_value,
                "visual_system_role": role,
                "harvard_oxford_label": atlas_label,
                "source_atlas_value": atlas_value,
                "voxel_count_2mm": voxel_count,
                "volume_mm3": voxel_count * 8,
                "individual_mask_file": individual_name,
            }
        )

    label_path = OUTPUT_DIR / "harvard_oxford_4roi_surface_otc_labels_mni152_2mm.nii.gz"
    binary_path = OUTPUT_DIR / "harvard_oxford_4roi_surface_otc_binary_mni152_2mm.nii.gz"
    save_nifti(
        combined_labels,
        atlas_img,
        label_path,
        "4-label Harvard-Oxford surface-EVC/visual-stream mask; MNI152 2mm",
    )
    save_nifti(
        binary_union,
        atlas_img,
        binary_path,
        "Binary union of revised Harvard-Oxford visual-stream ROIs; MNI152 2mm",
    )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "harvard_oxford_4roi_labels.csv", index=False)
    metadata = {
        "space": "MNI152",
        "resolution_mm": [2.0, 2.0, 2.0],
        "shape": list(atlas_data.shape),
        "affine": np.asarray(atlas_img.affine).tolist(),
        "source_atlas": ATLAS_NAME,
        "source_atlas_symmetric_split": False,
        "laterality": "bilateral",
        "label_mask": label_path.name,
        "binary_union_mask": binary_path.name,
        "label_values": rows,
    }
    (OUTPUT_DIR / "harvard_oxford_4roi_mask_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    display = plot_roi(
        nib.load(str(label_path)),
        display_mode="ortho",
        cut_coords=(0, -70, 10),
        cmap="tab10",
        alpha=0.75,
        colorbar=True,
        title="Harvard-Oxford surface-EVC and visual-stream mask (MNI152 2 mm)",
    )
    display.savefig(str(OUTPUT_DIR / "harvard_oxford_4roi_surface_otc_qc.png"), dpi=180)
    display.close()
    plt.close("all")

    print(table.to_string(index=False))
    print("Saved label mask:", label_path)
    print("Saved binary mask:", binary_path)


if __name__ == "__main__":
    main()
