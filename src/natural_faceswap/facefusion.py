from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RECOMMENDED_FACEFUSION_COMMIT = "4b1dedb853e4838ca7f3cf70b572be241aee2497"


@dataclass(frozen=True)
class FaceFusionConfig:
    home: Path
    references: list[Path]
    target: Path
    output: Path
    provider: str = "cuda"
    model: str = "hyperswap_1a_256"
    selector_mode: str = "one"
    reference_frame_number: int = 0
    reference_face_position: int = 0
    reference_face_distance: float = 0.6


def build_command(config: FaceFusionConfig) -> list[str]:
    command = [
        sys.executable,
        str(config.home / "facefusion.py"),
        "headless-run",
        "--source-paths",
        *[str(reference) for reference in config.references],
        "--target-path",
        str(config.target),
        "--output-path",
        str(config.output),
        "--processors",
        "face_swapper",
        "--face-swapper-model",
        config.model,
        "--face-swapper-pixel-boost",
        "512x512",
        "--face-selector-mode",
        config.selector_mode,
        "--execution-providers",
        config.provider,
        "--output-video-encoder",
        "libx264",
        "--output-video-quality",
        "100",
        "--output-video-preset",
        "slow",
    ]
    if config.selector_mode == "reference":
        command.extend(
            [
                "--reference-frame-number",
                str(config.reference_frame_number),
                "--reference-face-position",
                str(config.reference_face_position),
                "--reference-face-distance",
                str(config.reference_face_distance),
            ]
        )
    return command


def printable_command(command: list[str]) -> str:
    return shlex.join(command)


def current_commit(home: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(home), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()
