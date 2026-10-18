#!/bin/sh
# Composites the app's screen recording on top of your real camera footage (picture-in-picture),
# for the demo video:
#
#   tools/compose_demo_video.sh <camera footage> <screen recording> <output.mp4> [offset_s] [side]
#
#   camera footage    the real video of you studying (iPhone .MOV is fine; HDR is tone-mapped)
#   screen recording  the Simulator/iPhone screen capture, e.g. made with
#                     xcrun simctl io booted recordVideo --codec=h264 app.mov   (Ctrl-C to stop)
#   offset_s          seconds to delay the screen recording so both line up (default 0; negative trims it)
#   side              right (default) or left
#
# The phone screen is scaled to ~78% of the frame height, given rounded corners and a soft shadow,
# and placed with a margin. Output: 1080p H.264, the camera's audio (if any), ready to edit or upload.
set -e
[ -f "$1" ] && [ -f "$2" ] && [ -n "$3" ] || { sed -n '2,15p' "$0"; exit 1; }
CAM="$1"; SCR="$2"; OUT="$3"; OFF="${4:-0}"; SIDE="${5:-right}"
TMP="${TMPDIR:-/tmp}/focusloop-compose"; mkdir -p "$TMP"

# 1. Camera footage → SDR 1080p (Apple's converter tone-maps iPhone HDR correctly).
avconvert -s "$CAM" -p Preset1920x1080 -o "$TMP/cam.mov" --replace >/dev/null

# 2. Screen-recording delay/trim.
if [ "$(echo "$OFF < 0" | bc)" -eq 1 ]; then SCR_IN="-ss $(echo "-1 * $OFF" | bc) -i"; DELAY=0; else SCR_IN="-i"; DELAY="$OFF"; fi

# 3. Layout: 1920x1080 canvas. Portrait camera footage is centred on a blurred fill of itself;
#    landscape footage fills the frame. Phone screen height = 840px, rounded 48px corners.
if [ "$SIDE" = "left" ]; then X="80"; else X="W-w-80"; fi
ffmpeg -v error -y -i "$TMP/cam.mov" $SCR_IN "$SCR" -filter_complex "
  [0:v]split=2[c1][c2];
  [c1]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=30:2,eq=brightness=-0.15[bg];
  [c2]scale=-2:1080:force_original_aspect_ratio=decrease[fg];
  [bg][fg]overlay=(W-w)/2:(H-h)/2[cam];
  [1:v]setpts=PTS-STARTPTS+${DELAY}/TB,scale=-2:840,format=yuva420p,
       geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':
           a='if(gt(abs(W/2-X),W/2-48)*gt(abs(H/2-Y),H/2-48),if(lte(hypot(48-(W/2-abs(W/2-X)),48-(H/2-abs(H/2-Y))),48),255,0),255)'[phone];
  [phone]split=2[p1][p2];
  [p2]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.45,boxblur=18:1[shadow];
  [cam][shadow]overlay=x='${X}+12':y='(H-h)/2+16':eof_action=pass[withshadow];
  [withshadow][p1]overlay=x='${X}':y='(H-h)/2':eof_action=pass,format=yuv420p[v]" \
  -map "[v]" -map 0:a? -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 160k -shortest -movflags +faststart "$OUT"
rm -rf "$TMP"
echo "Wrote $OUT"
