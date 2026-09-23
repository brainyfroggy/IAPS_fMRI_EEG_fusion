"""Build atlas crosswalks and two ROI panels from the robust fusion ranking."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "top_roi_selection_outputs"
AAL3_DIR = PROJECT.parent / "data" / "masks" / "AAL3" / "new"

# HCP-MMP1.0 is surface-based and much finer than the bilateral AAL3 masks.
# These are anatomical correspondences, not one-to-one parcel equivalences.
HCP_APPROX = {
    "Cerebellum": "Not represented in cortical HCP-MMP1.0",
    "OccipitalMid": "V3A, V3B, V3CD, LO1, LO2, V4t",
    "SupraMarginal": "PF, PFm, PFop, PFt, PFcm",
    "Postcentral": "3a, 3b, 1, 2",
    "Fusiform": "FFC, V8, VMV1, VMV2, VMV3",
    "OccipitalInf": "V4, LO1, LO2, PIT",
    "Lingual": "V1, V2, V3, V4, VMV1, VMV2, VMV3",
    "Calcarine": "V1, V2",
    "TemporalMid": "MT, MST, FST, TE1p, TE2p, PHT",
    "ParietalSup": "7AL, 7Am, 7PC, 7PL, IPS1",
    "Insula": "AVI, AAIC, MI, PI, PoI1, PoI2, Igd",
    "Amygdala": "Not represented in cortical HCP-MMP1.0",
    "ACCsup": "24dd, 24dv, a24pr, p24pr, p32pr",
}

CONFIRMATORY_PANEL = [
    ("OccipitalMid", "Data-strong visual association cortex"),
    ("Fusiform", "Data-strong ventral visual cortex; tested in Bo et al. 2022"),
    ("OccipitalInf", "Data-strong lateral/ventral occipital cortex"),
    ("Lingual", "Data-strong ventromedial visual cortex"),
    ("Calcarine", "Early visual cortex anchor"),
    ("TemporalMid", "Data-ranked temporal association node and cascade target"),
    ("Cerebellum", "Strongest discovery result; retain as exploratory replication target"),
    ("Insula", "Independent emotion-network prior; tested in Bo et al. 2022"),
    ("Amygdala", "Independent emotion/reentry prior; tested in Bo et al. 2022"),
    ("ACCsup", "AAL3 supracallosal/dorsal ACC prior; tested in Bo et al. 2022"),
]

VISUAL_STREAM_PANEL = [
    ("Calcarine", "V1 / early visual cortex"),
    ("ParietalSup", "Dorsal visual stream"),
    ("Fusiform", "Ventral visual stream"),
    ("OccipitalMid", "Lateral visual cortex"),
]

HARVARD_OXFORD_VISUAL_PANEL = [
    {
        "panel_order": 1,
        "visual_system_role": "Surface early visual cortex",
        "harvard_oxford_roi": "Occipital Pole",
        "source_aal3_proxy": "Calcarine",
        "selection_note": "Surface-visible early visual proxy; broader and less V1-specific than intracalcarine cortex",
    },
    {
        "panel_order": 2,
        "visual_system_role": "Dorsal visual stream",
        "harvard_oxford_roi": "Superior Parietal Lobule",
        "source_aal3_proxy": "ParietalSup",
        "selection_note": "Parietal endpoint of the dorsal visual pathway",
    },
    {
        "panel_order": 3,
        "visual_system_role": "Ventral visual stream",
        "harvard_oxford_roi": "Temporal Occipital Fusiform Cortex",
        "source_aal3_proxy": "Fusiform",
        "selection_note": "Ventral occipitotemporal object and affective-scene representation",
    },
    {
        "panel_order": 4,
        "visual_system_role": "Lateral occipitotemporal cortex",
        "harvard_oxford_roi": "Lateral Occipital Cortex, inferior division",
        "source_aal3_proxy": "OccipitalMid",
        "selection_note": "Closest single Harvard-Oxford parcel to lateral occipitotemporal cortex",
    },
]


def overlap_string(mask_img: nib.spatialimages.SpatialImage, atlas) -> str:
    mask = resample_to_img(
        mask_img,
        atlas.maps,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    ).get_fdata() > 0
    labels = np.asarray(atlas.maps.get_fdata(), dtype=int)
    ids, counts = np.unique(labels[mask & (labels > 0)], return_counts=True)
    if not len(ids):
        return "No overlap"
    order = np.argsort(counts)[::-1][:3]
    total = int(mask.sum())
    return "; ".join(
        f"{atlas.labels[int(ids[i])]} ({100 * counts[i] / total:.1f}%)" for i in order
    )


def main() -> None:
    ranking = pd.read_csv(OUTPUT / "aal3_roi_robust_ranking.csv")
    atlas_cache = OUTPUT / "atlases"
    cortical = fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm", data_dir=str(atlas_cache), symmetric_split=False
    )
    subcortical = fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-2mm", data_dir=str(atlas_cache), symmetric_split=False
    )

    union_rois = list(
        dict.fromkeys(
            ranking.sort_values("robust_data_rank").head(10)["roi"].tolist()
            + [roi for roi, _ in CONFIRMATORY_PANEL]
            + [roi for roi, _ in VISUAL_STREAM_PANEL]
        )
    )
    crosswalk_rows = []
    for roi in union_rois:
        mask_img = nib.load(str(AAL3_DIR / f"{roi}.nii.gz"))
        cortical_overlap = overlap_string(mask_img, cortical)
        subcortical_overlap = overlap_string(mask_img, subcortical)
        if roi == "Cerebellum":
            cortical_overlap = "Not represented; use a dedicated probabilistic cerebellar atlas"
            subcortical_overlap = "Not represented; use a dedicated probabilistic cerebellar atlas"
        elif roi == "Amygdala":
            cortical_overlap = "Not applicable; see Harvard-Oxford subcortical overlap"
        crosswalk_rows.append(
            {
                "aal3_roi": roi,
                "harvard_oxford_cortical_top_overlap": cortical_overlap,
                "harvard_oxford_subcortical_top_overlap": subcortical_overlap,
                "hcp_mmp1_approximate_cortical_parcels": HCP_APPROX[roi],
                "crosswalk_status": "Spatial overlap for Harvard-Oxford; anatomical approximation for HCP-MMP1.0",
            }
        )
    crosswalk = pd.DataFrame(crosswalk_rows)
    crosswalk.to_csv(OUTPUT / "aal3_harvardoxford_hcpmmp_crosswalk.csv", index=False)

    literal = ranking.sort_values(["robust_data_rank", "roi"]).head(10).copy()
    literal.insert(0, "panel_order", np.arange(1, len(literal) + 1))
    literal = literal.merge(crosswalk, left_on="roi", right_on="aal3_roi", how="left")
    literal.to_csv(OUTPUT / "literal_top10_data_driven_with_atlas_crosswalk.csv", index=False)

    rationale = dict(CONFIRMATORY_PANEL)
    panel_order = {roi: i + 1 for i, (roi, _) in enumerate(CONFIRMATORY_PANEL)}
    confirmatory = ranking[ranking["roi"].isin(panel_order)].copy()
    confirmatory["panel_order"] = confirmatory["roi"].map(panel_order)
    confirmatory["selection_basis"] = confirmatory["roi"].map(rationale)
    confirmatory["panel_tier"] = np.where(
        confirmatory["robust_data_rank"] <= 10,
        "Observed top-10 fusion",
        "Independent literature prior",
    )
    confirmatory = confirmatory.sort_values("panel_order").merge(
        crosswalk, left_on="roi", right_on="aal3_roi", how="left"
    )
    confirmatory.to_csv(OUTPUT / "recommended_10_roi_confirmatory_panel.csv", index=False)

    visual_role = dict(VISUAL_STREAM_PANEL)
    visual_order = {roi: i + 1 for i, (roi, _) in enumerate(VISUAL_STREAM_PANEL)}
    visual_panel = ranking[ranking["roi"].isin(visual_order)].copy()
    visual_panel["panel_order"] = visual_panel["roi"].map(visual_order)
    visual_panel["visual_system_role"] = visual_panel["roi"].map(visual_role)
    visual_panel = visual_panel.sort_values("panel_order").merge(
        crosswalk, left_on="roi", right_on="aal3_roi", how="left"
    )
    visual_panel.to_csv(OUTPUT / "recommended_4_roi_visual_stream_panel.csv", index=False)

    ho_panel = pd.DataFrame(HARVARD_OXFORD_VISUAL_PANEL)
    rank_lookup = ranking.set_index("roi")
    ho_panel["source_aal3_balanced_rank"] = ho_panel["source_aal3_proxy"].map(
        rank_lookup["robust_data_rank"]
    )
    ho_panel["harvard_oxford_atlas"] = "cort-maxprob-thr25-2mm"
    ho_panel.to_csv(OUTPUT / "recommended_4_roi_harvard_oxford_panel_surface_otc.csv", index=False)

    print("Literal top 10:")
    print(literal[["panel_order", "roi", "robust_data_rank"]].to_string(index=False))
    print("\nRecommended confirmatory panel:")
    print(
        confirmatory[
            ["panel_order", "roi", "robust_data_rank", "panel_tier", "selection_basis"]
        ].to_string(index=False)
    )
    print("\nFour-ROI visual-stream panel:")
    print(
        visual_panel[
            ["panel_order", "visual_system_role", "roi", "robust_data_rank"]
        ].to_string(index=False)
    )
    print("\nFour-ROI Harvard-Oxford panel:")
    print(
        ho_panel[
            ["panel_order", "visual_system_role", "harvard_oxford_roi"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
