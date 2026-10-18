# Demo recording (optional)

Drop a video of yourself studying here as **`demo_face.mov`** (or **`demo_face.mp4`**), then run
`xcodegen generate` and rebuild. The app bundles it and can use it as the camera for demos (Settings →
Debug → Use my recorded video), which is how the real perception pipeline runs in the Simulator.


* The file is git-ignored (it is a video of your face), so it stays on your Mac.
* This README is not bundled. If no video is here, the build simply skips it.
* Alternatives that need no rebuild: import the video in Settings (Photos or Files), or launch with
  `-recordedVideo /absolute/path/to/video.mov`.
