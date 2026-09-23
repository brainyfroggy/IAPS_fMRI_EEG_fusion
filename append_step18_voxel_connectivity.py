"""Append the tested top-25% voxel connectivity workflow to the project notebook."""

from pathlib import Path

import nbformat


PROJECT = Path(__file__).resolve().parent
NOTEBOOK = PROJECT / "AAL3_EEG_fMRI_representational_connectivity.ipynb"
STEP_HEADING = "## Step 18: Top-25% Voxel-to-Voxel Dynamic ROI Connectivity and 3D State Space"


markdown = f"""{STEP_HEADING}

This step uses the four Harvard-Oxford masks containing the peak top 25% of fusion voxels. For each subject and each sliding time window, every voxel's EEG-fMRI fusion trajectory is standardized within the window. Pearson correlations are calculated for every cross-ROI voxel pair and averaged to one subject-level ROI-pair value. Subject values are Fisher-z averaged and transformed back to `r`, producing six ROI-to-ROI correlation time courses.

The default 160 ms window and 20 ms step estimate local synchrony while retaining useful temporal resolution. This is **dynamic connectivity of the fused representational signal**, not conventional BOLD functional connectivity and not evidence of directionality or causal influence.

Two 3D views are provided for EVC plus any two other ROIs:

1. **Representational state space:** the three group-average ROI fusion responses are the axes.
2. **Connectivity state space:** the three pairwise dynamic correlations are the axes.

Because the top-25% masks were selected using the same group fusion data, inferential testing on these trajectories would be circular. Use independent ROI definitions, held-out subjects, or nested cross-validation for confirmatory statistics.
"""


compute_code = """import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from top25_voxel_dynamic_connectivity import (
    OUTPUT_DIR as TOP25_CONNECTIVITY_OUT_DIR,
    compute_voxel_dynamic_connectivity,
    plot_three_roi_state_space,
)

# A 160 ms window spans 41 samples at the 4 ms EEG-fusion sampling interval.
VOXEL_CONNECTIVITY_WINDOW_MS = 160.0
VOXEL_CONNECTIVITY_STEP_MS = 20.0
VOXEL_CONNECTIVITY_FORCE_RECOMPUTE = False

top25_voxel_connectivity = compute_voxel_dynamic_connectivity(
    window_ms=VOXEL_CONNECTIVITY_WINDOW_MS,
    step_ms=VOXEL_CONNECTIVITY_STEP_MS,
    force=VOXEL_CONNECTIVITY_FORCE_RECOMPUTE,
)

top25_roi_table = pd.DataFrame({
    "mask_value": top25_voxel_connectivity["roi_labels"],
    "short_name": ["EVC", "Dorsal", "Ventral", "Lateral OTC"],
    "role": top25_voxel_connectivity["roi_names"],
    "Harvard-Oxford ROI": top25_voxel_connectivity["roi_atlas_names"],
    "selected_voxels_3mm": top25_voxel_connectivity["roi_voxel_counts"],
})
display(top25_roi_table)
print("Subject ROI trajectories:", top25_voxel_connectivity["subject_roi_time"].shape)
print("Subject edge trajectories:", top25_voxel_connectivity["subject_edge_time"].shape)
print("Saved outputs:", TOP25_CONNECTIVITY_OUT_DIR)
"""


static_plot_code = """# Label 1 must remain EVC. Choose any two distinct labels from 2, 3, and 4.
STATE_SPACE_ROI_LABELS = (1, 2, 3)  # EVC + dorsal + ventral
STATE_SPACE_GAUSSIAN_SIGMA_MS = 20.0  # 0 shows unsmoothed curves

state_space_name = "_".join(map(str, STATE_SPACE_ROI_LABELS))
fig = plot_three_roi_state_space(
    top25_voxel_connectivity,
    selected_labels=STATE_SPACE_ROI_LABELS,
    gaussian_sigma_ms=STATE_SPACE_GAUSSIAN_SIGMA_MS,
    output_path=TOP25_CONNECTIVITY_OUT_DIR / f"state_space_labels_{state_space_name}.png",
)
plt.show()
"""


interactive_code = """# Interactive EVC + two-ROI state-space explorer.
try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    secondary_roi_options = [
        ("Dorsal visual stream", 2),
        ("Ventral visual stream", 3),
        ("Lateral occipitotemporal cortex", 4),
    ]
    roi_b_dropdown = widgets.Dropdown(
        options=secondary_roi_options, value=2, description="ROI 2:"
    )
    roi_c_dropdown = widgets.Dropdown(
        options=secondary_roi_options, value=3, description="ROI 3:"
    )
    smoothing_toggle = widgets.Checkbox(
        value=True, description="Gaussian smoothing"
    )
    smoothing_slider = widgets.FloatSlider(
        value=20.0, min=0.0, max=100.0, step=4.0,
        description="Sigma (ms):", continuous_update=False,
    )
    state_output = widgets.Output()

    def refresh_top25_state_space(*_):
        with state_output:
            clear_output(wait=True)
            if roi_b_dropdown.value == roi_c_dropdown.value:
                print("Choose two different non-EVC ROIs.")
                return
            sigma_ms = smoothing_slider.value if smoothing_toggle.value else 0.0
            interactive_fig = plot_three_roi_state_space(
                top25_voxel_connectivity,
                selected_labels=(1, roi_b_dropdown.value, roi_c_dropdown.value),
                gaussian_sigma_ms=sigma_ms,
            )
            display(interactive_fig)
            plt.close(interactive_fig)

    for control in (roi_b_dropdown, roi_c_dropdown, smoothing_toggle, smoothing_slider):
        control.observe(refresh_top25_state_space, names="value")
    display(
        widgets.HBox([roi_b_dropdown, roi_c_dropdown]),
        widgets.HBox([smoothing_toggle, smoothing_slider]),
        state_output,
    )
    refresh_top25_state_space()
except ImportError:
    print("ipywidgets is unavailable; use STATE_SPACE_ROI_LABELS in the preceding cell.")
"""


notebook = nbformat.read(NOTEBOOK, as_version=4)
existing = next(
    (
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and "".join(cell.source).startswith(STEP_HEADING)
    ),
    None,
)
if existing is not None:
    notebook.cells = notebook.cells[:existing]
while notebook.cells and not "".join(notebook.cells[-1].source).strip():
    notebook.cells.pop()

notebook.cells.extend(
    [
        nbformat.v4.new_markdown_cell(markdown),
        nbformat.v4.new_code_cell(compute_code),
        nbformat.v4.new_code_cell(static_plot_code),
        nbformat.v4.new_code_cell(interactive_code),
    ]
)
nbformat.write(notebook, NOTEBOOK)
print(f"Appended Step 18 to {NOTEBOOK}")
