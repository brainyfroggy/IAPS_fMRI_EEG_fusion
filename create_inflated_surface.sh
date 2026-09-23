#!/usr/bin/env bash
# Robust per-frame conversion to inflated GIFTI for all emotion folders
# Starts at frame 001

set -o pipefail   # keep -e off so one bad file doesn't kill the whole run

# ============================================
# FreeSurfer setup
# ============================================
export FREESURFER_HOME="/usr/local/freesurfer/7.4.1"
export FS_LICENSE="/mnt/n/Experimental_Data/yujunchen/softwares/fs_license.txt"
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

# ============================================
# Paths
# ============================================
BASE="/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_EEG_fusion"
LHSURF_INFL="$FREESURFER_HOME/subjects/fsaverage/surf/lh.inflated"
RHSURF_INFL="$FREESURFER_HOME/subjects/fsaverage/surf/rh.inflated"

EMOTIONS=("neutral" "pleasant" "unpleasant")
START=1   # start from 001, skip 000

# ============================================
# Helper: get last existing frame index
# ============================================
last_idx() {
  local work="$1"
  local side="$2"
  ls -1v "$work/${side}.tp"*.mg? 2>/dev/null | \
    sed -n 's/.*\.tp\([0-9][0-9][0-9]\)\.mg[hz]$/\1/p' | \
    tail -n1
}

# ============================================
# Loop over emotions
# ============================================
for emo in "${EMOTIONS[@]}"; do
  WORK="$BASE/fs_rsa_${emo}"

  if [[ ! -d "$WORK" ]]; then
    echo "Missing folder: $WORK ; skipping"
    continue
  fi

  cd "$WORK" || { echo "Cannot cd to $WORK"; continue; }

  echo "======================================"
  echo "Processing emotion: $emo"
  echo "Working folder: $WORK"
  echo "======================================"

  END_LH=$(last_idx "$WORK" lh)
  END_RH=$(last_idx "$WORK" rh)

  : "${END_LH:=999}"
  : "${END_RH:=999}"

  echo "LH frames: ${START}..${END_LH}"
  for i in $(seq -w $START $END_LH); do
    in_mgh="$WORK/lh.tp${i}.mgh"
    in_mgz="$WORK/lh.tp${i}.mgz"

    if [[ -f "$in_mgh" ]]; then
      in_file="$in_mgh"
    elif [[ -f "$in_mgz" ]]; then
      in_file="$in_mgz"
    else
      echo "$emo | LH frame ${i}: no input; skipping"
      continue
    fi

    out_file="${in_file%.*}.inflated.func.gii"
    if [[ -f "$out_file" ]]; then
      echo "$emo | LH frame ${i}: already exists -> $(basename "$out_file"); skipping"
      continue
    fi

    echo "$emo | LH frame ${i}: $(basename "$in_file") -> $(basename "$out_file")"
    mris_convert -c "$in_file" "$LHSURF_INFL" "$out_file" || {
      echo "$emo | LH frame ${i}: mris_convert failed"
      continue
    }
  done

  echo "RH frames: ${START}..${END_RH}"
  for i in $(seq -w $START $END_RH); do
    in_mgh="$WORK/rh.tp${i}.mgh"
    in_mgz="$WORK/rh.tp${i}.mgz"

    if [[ -f "$in_mgh" ]]; then
      in_file="$in_mgh"
    elif [[ -f "$in_mgz" ]]; then
      in_file="$in_mgz"
    else
      echo "$emo | RH frame ${i}: no input; skipping"
      continue
    fi

    out_file="${in_file%.*}.inflated.func.gii"
    if [[ -f "$out_file" ]]; then
      echo "$emo | RH frame ${i}: already exists -> $(basename "$out_file"); skipping"
      continue
    fi

    echo "$emo | RH frame ${i}: $(basename "$in_file") -> $(basename "$out_file")"
    mris_convert -c "$in_file" "$RHSURF_INFL" "$out_file" || {
      echo "$emo | RH frame ${i}: mris_convert failed"
      continue
    }
  done

  echo "Finished emotion: $emo"
  echo
done

echo "All emotions done."