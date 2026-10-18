#!/usr/bin/env python3
"""Step 1 of the offline brain-content pipeline: split a lecture video into model inputs.

For one video this writes, under ml/data/brain/<id>/ (gitignored, Mac only):
  frames/frame_00001.jpg ...   video frames at --fps (default 1 fps), scaled to <= --frame-width px wide
  audio.wav                    16 kHz mono PCM
  transcript.json              [{"start": s, "end": s, "text": "..."}]  (may be [])
  stimulus.json                manifest (paths, duration, transcript source)

Transcript priority: --transcript PATH > ml/brain/transcripts/<id>.{vtt,srt} > <video>.{vtt,srt}
> local Whisper (if the `whisper` package is importable) > none (clear message, pipeline continues).

Usage:
  python ml/brain/prepare_stimulus.py Resources/Lectures/black-holes-01.mp4 --id black-holes-01
  python ml/brain/prepare_stimulus.py VIDEO --id ID --write-vtt   # also save a Whisper transcript as
                                                                   # ml/brain/transcripts/<id>.vtt
Requires ffmpeg/ffprobe on PATH (brew install ffmpeg).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain_common import LECTURES_DIR, TRANSCRIPTS_DIR, WORK_DIR, find_transcript, load_catalog  # noqa: E402


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH. Install it with: brew install ffmpeg")


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


# ---------------------------------------------------------------- transcripts

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _secs(ts: str) -> float:
    m = _TS.search(ts)
    if not m:
        raise ValueError(f"bad timestamp {ts!r}")
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_subtitles(path: str | Path) -> list[dict]:
    """Minimal WebVTT / SRT parser -> [{start, end, text}] sorted by start."""
    text = Path(path).read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->", 1)
                body = " ".join(lines[i + 1:])
                body = re.sub(r"<[^>]+>", "", body).strip()
                if body:
                    cues.append({"start": round(_secs(a), 3), "end": round(_secs(b.split()[0]), 3), "text": body})
                break
    return sorted(cues, key=lambda c: c["start"])


def write_vtt(segments: list[dict], path: str | Path) -> None:
    def ts(x: float) -> str:
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    lines = ["WEBVTT", ""]
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s["text"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def whisper_transcript(audio_wav: str | Path, model_name: str = "base") -> list[dict] | None:
    """Local Whisper (openai-whisper) if installed; None (with a message) otherwise."""
    try:
        import whisper  # type: ignore
    except ImportError:
        print("  [transcript] no sidecar .vtt/.srt and local Whisper is not installed "
              "(pip install openai-whisper) -> continuing WITHOUT a transcript.", file=sys.stderr)
        return None
    print(f"  [transcript] running local Whisper ({model_name}) ...", file=sys.stderr)
    model = whisper.load_model(model_name)
    res = model.transcribe(str(audio_wav), fp16=False, language="en")
    return [{"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "text": s["text"].strip()}
            for s in res["segments"] if s["text"].strip()]


# ---------------------------------------------------------------- main step

def prepare(video: str | Path, lecture_id: str, out_root: str | Path = WORK_DIR, fps: float = 1.0,
            frame_width: int = 320, transcript: str | Path | None = None, whisper_model: str = "base",
            write_vtt_sidecar: bool = False, extract_frames: bool = True) -> dict:
    require_ffmpeg()
    video = Path(video)
    out = Path(out_root) / lecture_id
    out.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video)

    frames_dir = out / "frames"
    if extract_frames:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir()
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                        "-vf", f"fps={fps},scale='min({frame_width},iw)':-2", "-q:v", "4",
                        str(frames_dir / "frame_%05d.jpg")], check=True)
    n_frames = len(list(frames_dir.glob("frame_*.jpg"))) if frames_dir.exists() else 0

    wav = out / "audio.wav"
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                     "stream=index", "-of", "csv=p=0", str(video)],
                                    capture_output=True, text=True).stdout.strip())
    if has_audio:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    else:  # silent track of the right length so downstream code has one shape
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "anullsrc=r=16000:cl=mono", "-t", f"{duration:.3f}", "-c:a", "pcm_s16le", str(wav)],
                       check=True)

    src = Path(transcript) if transcript else find_transcript(lecture_id, video)
    if src is not None:
        segs, tsrc = parse_subtitles(src), f"sidecar:{src.name}"
    else:
        segs = whisper_transcript(wav, whisper_model)
        tsrc = f"whisper:{whisper_model}" if segs is not None else "none"
        segs = segs or []
        if segs and write_vtt_sidecar:
            TRANSCRIPTS_DIR.mkdir(exist_ok=True)
            write_vtt(segs, TRANSCRIPTS_DIR / f"{lecture_id}.vtt")
            print(f"  [transcript] saved {TRANSCRIPTS_DIR / f'{lecture_id}.vtt'}", file=sys.stderr)
    (out / "transcript.json").write_text(json.dumps(segs, indent=1))

    manifest = {"lecture_id": lecture_id, "video": str(video.resolve()), "duration": round(duration, 3),
                "fps": fps, "frames_dir": str(frames_dir), "n_frames": n_frames, "audio_wav": str(wav),
                "has_audio": has_audio, "transcript": str(out / "transcript.json"),
                "transcript_source": tsrc, "n_transcript_segments": len(segs)}
    (out / "stimulus.json").write_text(json.dumps(manifest, indent=1))
    print(f"prepared {lecture_id}: {duration:.1f}s, {n_frames} frames, transcript={tsrc} ({len(segs)} segs)")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file (or use --all)")
    ap.add_argument("--all", action="store_true",
                    help="every lecture in Resources/Lectures/lectures.json that has a videoFile")
    ap.add_argument("--id", help="lecture id (default: video file stem)")
    ap.add_argument("--out", default=str(WORK_DIR))
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--frame-width", type=int, default=320)
    ap.add_argument("--transcript", help="explicit .vtt/.srt path")
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--write-vtt", action="store_true", help="save a Whisper transcript to ml/brain/transcripts/")
    a = ap.parse_args(argv)
    if a.all:
        if a.video or a.id or a.transcript:
            ap.error("--all cannot be combined with a video, --id or --transcript")
        jobs = [(x["id"], LECTURES_DIR / x["videoFile"]) for x in load_catalog() if x.get("videoFile")]
        if not jobs:
            ap.error("no lecture in lectures.json has a videoFile")
        for lid, video in jobs:
            prepare(video, lid, a.out, a.fps, a.frame_width, None, a.whisper_model, a.write_vtt)
        return
    if not a.video:
        ap.error("give a video path or --all")
    prepare(a.video, a.id or Path(a.video).stem, a.out, a.fps, a.frame_width, a.transcript, a.whisper_model,
            a.write_vtt)


if __name__ == "__main__":
    main()
