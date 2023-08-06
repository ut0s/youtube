#!/bin/bash
# @file crop_source.sh
# @brief

set -x

crop_video(){
  local filename=$1
  local post=$2
  local out_w=$3
  local out_h=$4
  local x=$5
  local y=$6
  local BITRATE=$7

  ffmpeg -i $filename.mp4 -b:v $BITRATE -filter:v "crop=${out_w}:${out_h}:${x}:${y}" -an ${filename}_${post}.mp4
}

crop_audio(){
  local filename=$1
  local extention="aac"

  ffmpeg -i $filename.mp4 \
         -map 0:1 -vn -acodec copy ${filename}_f3.${extention}\
         -map 0:2 -vn -acodec copy ${filename}_Anker.${extention}\
         -map 0:3 -vn -acodec copy ${filename}_iPhone.${extention}
}


main(){
  local file=$1
  local filename="${file%.*}"

  # Audio
  crop_audio $filename

  # AnkerC200
  crop_video $filename AnkerC200 1920 1080 0 1080 "300k"

  # iPhone
  crop_video $filename iPhone 1920 1080 1920 1080 "500k"

  # iPhone Sim
  crop_video $filename sim_iPhone 510 1080 0 0 "1M"

  # Android Sim
  crop_video $filename sim_android 633 1080 550 0 "1M"

  # Spotify
  crop_video $filename spotify 556 1080 3284 0 "1M"

  # Chrome Dev
  crop_video $filename chrome_debug 1920 1080 1255 0 "1M"
}

main $@
