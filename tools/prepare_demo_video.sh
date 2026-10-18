#!/bin/sh
# Prepares an iPhone recording (often HDR/HLG, 4K 60 fps) as the app's demo face video:
#   tools/prepare_demo_video.sh ~/Downloads/IMG_1234.MOV
# 1. HDR → SDR 1080p with Apple's avconvert (proper tone mapping; otherwise the video looks washed out),
# 2. 30 fps H.264 → Resources/Demo/demo_face.mp4 (git-ignored: your face never goes in the repo),
# 3. Mac face track (blinks/yawns/gaze for the Simulator) → Resources/Demo/demo_face.facetrack.csv.
# Then rebuild the app. In the Simulator, sessions use this video automatically.
set -e
[ -f "$1" ] || { echo "usage: $0 <video>"; exit 1; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP="${TMPDIR:-/tmp}/focusloop-sdr.mov"
mkdir -p "$ROOT/Resources/Demo"
avconvert -s "$1" -p Preset1920x1080 -o "$TMP" --replace
ffmpeg -v error -y -i "$TMP" -map 0:v:0 -an -vf fps=30 -c:v libx264 -profile:v high -crf 20 -preset slow \
  -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -movflags +faststart \
  "$ROOT/Resources/Demo/demo_face.mp4"
rm -f "$TMP"
"$ROOT/tools/facetrack.sh" "$ROOT/Resources/Demo/demo_face.mp4"
echo "Ready: Resources/Demo/demo_face.mp4 (+ face track). Rebuild the app (xcodegen generate, then Run)."
