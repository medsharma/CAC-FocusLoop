#!/bin/sh
# Builds and runs the FocusLoop face-track tool on the Mac (see tools/facetrack/main.swift):
#   tools/facetrack.sh Resources/Demo/demo_face.mov
# writes Resources/Demo/demo_face.facetrack.csv (numbers only). Needed for blinks / yawns in the Simulator.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="${TMPDIR:-/tmp}/focusloop-facetrack"
swiftc -O -parse-as-library -module-name FocusLoopFaceTrack -o "$BIN" \
  "$ROOT/tools/facetrack/main.swift" \
  "$ROOT/Modules/Contracts/Types.swift" "$ROOT/Modules/Contracts/FeatureFrame.swift" \
  "$ROOT/Modules/Contracts/Protocols.swift" "$ROOT/Modules/Contracts/ReportAndLectures.swift" \
  "$ROOT/Modules/Contracts/AppConfig.swift" \
  "$ROOT/Modules/Perception/VideoFaceAnalyzer.swift" "$ROOT/Modules/Perception/FaceTrack.swift" \
  "$ROOT/Modules/Perception/AttentionFeatureExtractor.swift" "$ROOT/Modules/Perception/PerceptionEventDetectors.swift" \
  "$ROOT/Modules/Perception/WorkZoneMath.swift" "$ROOT/Modules/Perception/PerceptionPoseMath.swift" \
  "$ROOT/Modules/RPPG/POSHeartRateEstimator.swift" "$ROOT/Modules/RPPG/RPPGDSP.swift" \
  "$ROOT/Modules/RPPG/RPPGSkinPixelAverager.swift"
"$BIN" "$@"
