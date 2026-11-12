#!/bin/bash
# Assemble the captured frame sequences and the score into the trailer.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-trailer.mp4}
rm -rf seq && mkdir seq
i=0
for scene in open title s1a s1b s2a s2b s3 s4 why end; do
  for f in frames/$scene/*.jpg; do
    ln -s "../$f" "seq/$(printf %05d $i).jpg"; i=$((i + 1))
  done
done
echo "$i frames"
ffmpeg -v error -y -framerate 30 -i seq/%05d.jpg -i score.wav \
  -vf "tpad=stop_mode=add:stop_duration=2.2:color=black,format=yuv420p" \
  -c:v libx264 -preset slow -crf 17 -c:a aac -b:a 192k -movflags +faststart -shortest "$OUT"
ffprobe -v error -show_entries format=duration,size -of csv=p=0 "$OUT"
