#!/usr/bin/env bash

ffmpeg -framerate 4 \
-i "/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_EEG_fusion/FSLeyes_outputs/merged_frames_12views/merged_%03d.png" \
-c:v libx264 -crf 18 -pix_fmt yuv420p -r 4 \
"/mnt/n/Experimental_Data/yujunchen/projects/IAPS_fMRI_EEG_fusion/FSLeyes_outputs/merged_RSA_movie.mp4"
