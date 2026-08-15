from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path


def require_media_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing required media tools: {', '.join(missing)}")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe(path: Path) -> dict[str, object]:
    require_media_tools()
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration,size,format_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_video(raw_video: Path, audio_source: Path, output: Path) -> None:
    require_media_tools()
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_video),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def make_gif(video: Path, output: Path, caption_png: Path | None, fps: int = 12) -> None:
    require_media_tools()
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = output.with_suffix(".palette.png")
    if caption_png:
        base_filter = (
            f"[0:v]scale=720:405:flags=lanczos[base];[base][1:v]overlay=0:0:format=auto,fps={fps}[gifbase]"
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-i",
                str(caption_png),
                "-filter_complex",
                base_filter + ";[gifbase]palettegen=max_colors=256:stats_mode=diff[palette]",
                "-map",
                "[palette]",
                str(palette),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-i",
                str(caption_png),
                "-i",
                str(palette),
                "-filter_complex",
                base_filter + ";[gifbase][2:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle[gif]",
                "-map",
                "[gif]",
                "-loop",
                "0",
                str(output),
            ]
        )
    else:
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                f"fps={fps},scale=720:405:flags=lanczos,palettegen=max_colors=256:stats_mode=diff",
                str(palette),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-i",
                str(palette),
                "-lavfi",
                f"fps={fps},scale=720:405:flags=lanczos[x];"
                "[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                "-loop",
                "0",
                str(output),
            ]
        )
    palette.unlink(missing_ok=True)


def make_contact_sheet(video: Path, output: Path, samples: int = 12) -> None:
    metadata = probe(video)
    duration = float(metadata["format"]["duration"])
    columns = 4
    rows = math.ceil(samples / columns)
    sample_fps = samples / max(duration, 0.001)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={sample_fps:.8f},scale=480:-2:flags=lanczos,tile={columns}x{rows}",
            "-frames:v",
            "1",
            str(output),
        ]
    )
