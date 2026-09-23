#!/bin/bash

# ============================================
# FreeSurfer setup
# ============================================
export FREESURFER_HOME="/usr/local/freesurfer/7.4.1"
export FS_LICENSE="/mnt/n/Experimental_Data/yujunchen/softwares/fs_license.txt"
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

# ============================================
# Root paths
# ============================================
BASE="/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_EEG_fusion"

declare -A INPUTS
INPUTS[neutral]="$BASE/searchlight_rsa_neutral/rsa_neutral_subject_mean_nonan.nii.gz"
INPUTS[pleasant]="$BASE/searchlight_rsa_pleasant/rsa_pleasant_subject_mean_nonan.nii.gz"
INPUTS[unpleasant]="$BASE/searchlight_rsa_unpleasant/rsa_unpleasant_subject_mean_nonan.nii.gz"

LHSURF="$FREESURFER_HOME/subjects/fsaverage/surf/lh.pial"
RHSURF="$FREESURFER_HOME/subjects/fsaverage/surf/rh.pial"

# ============================================
# Loop over emotions
# ============================================
for emo in neutral pleasant unpleasant; do
    MOV4D="${INPUTS[$emo]}"
    WORK="$BASE/fs_rsa_${emo}"
    mkdir -p "$WORK"
    cd "$WORK" || exit 1

    echo "===================================="
    echo "Processing $emo"
    echo "Input: $MOV4D"
    echo "Output folder: $WORK"
    echo "===================================="

    # number of frames
    N=$(mri_info --nframes "$MOV4D")
    echo "Frames: $N"

    # ----------------------------------------
    # Project each frame to fsaverage surface
    # ----------------------------------------
    for H in lh rh; do
        for f in $(seq 0 $((N-1))); do
            out="$H.tp$(printf %03d $f).mgh"
            [ -f "$out" ] && continue

            echo "$emo | $H | frame $f / $((N-1))"

            mri_vol2surf \
                --mov "$MOV4D" \
                --frame "$f" \
                --mni152reg \
                --hemi "$H" \
                --trgsubject fsaverage \
                --projfrac 0.5 \
                --interp trilinear \
                --o "$out"
        done
    done

    # ----------------------------------------
    # Convert per-frame MGH to per-frame func.gii
    # ----------------------------------------
    for f in "$WORK"/lh.tp*.mgh; do
        [ -e "$f" ] || continue
        base="${f%.mgh}"
        mris_convert -c "$f" "$LHSURF" "${base}.func.gii"
    done

    for f in "$WORK"/rh.tp*.mgh; do
        [ -e "$f" ] || continue
        base="${f%.mgh}"
        mris_convert -c "$f" "$RHSURF" "${base}.func.gii"
    done

    echo "Finished per-frame surface conversion for $emo"
done

echo "All emotions finished."