#!/bin/bash
# @file crop_source.sh
# @brief

set -x

scale_video(){
  local filename=$1
  local BITRATE=$2
  local SIZE=$3

  ffmpeg -i $filename.mp4\
         -b:v $BITRATE\
         -s $SIZE\
         ${filename}_${SIZE}.mp4
}

main(){
  local file=$1
  local filename="${file%.*}"

  # scale
  scale_video $filename "200k" 640x360
}

main $@
