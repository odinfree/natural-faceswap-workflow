from __future__ import annotations

import argparse
import json
import platform
import subprocess
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from natural_faceswap.expression import ExpressionSettings, transfer_expressions
from natural_faceswap.facefusion import (
    RECOMMENDED_FACEFUSION_COMMIT,
    FaceFusionConfig,
    build_command,
    current_commit,
    printable_command,
)
from natural_faceswap.landmarks import parse_point
from natural_faceswap.media import finalize_video, make_contact_sheet, make_gif, probe, sha256
from natural_faceswap.policy import RightsContext

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def add_rights_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--usage", choices=["internal-evaluation", "commercial"], required=True)
    parser.add_argument("--source-rights", choices=["unknown", "confirmed", "denied"], required=True)
    parser.add_argument("--model-rights", choices=["unknown", "confirmed", "denied"], required=True)
    parser.add_argument("--identity-authorized", action="store_true")


def rights_from_args(args) -> RightsContext:
    rights = RightsContext(
        usage=args.usage,
        source_rights=args.source_rights,
        model_rights=args.model_rights,
        identity_authorized=args.identity_authorized,
    )
    rights.validate()
    return rights


def command_inspect(args) -> None:
    print(json.dumps(probe(args.path), indent=2))


def command_download_model(args) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Model already exists: {args.output}. Use --force to replace it.")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, args.output)
    print(args.output)


def command_facefusion(args) -> None:
    rights_from_args(args)
    config = FaceFusionConfig(
        home=args.facefusion_home,
        references=args.reference,
        target=args.target,
        output=args.output,
        provider=args.provider,
        model=args.face_swapper_model,
        selector_mode=args.selector_mode,
        reference_frame_number=args.reference_frame_number,
        reference_face_position=args.reference_face_position,
        reference_face_distance=args.reference_face_distance,
    )
    command = build_command(config)
    found_commit = current_commit(config.home)
    print(f"recommended_facefusion_commit={RECOMMENDED_FACEFUSION_COMMIT}")
    print(f"current_facefusion_commit={found_commit or 'unknown'}")
    print(printable_command(command))
    if args.run:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


def _manifest(
    args,
    rights: RightsContext,
    settings: ExpressionSettings,
    failures: list[int],
    outputs: list[Path],
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rights": rights.to_dict(),
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "swapped": {"path": str(args.swapped), "sha256": sha256(args.swapped)},
            "source_url": args.source_url,
        },
        "settings": asdict(settings),
        "base_swap": {
            "tool": args.base_tool,
            "model": args.base_model,
            "commit": args.facefusion_commit,
        },
        "expected_face": list(args.expected_face),
        "landmark_failures": failures,
        "tools": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "mediapipe": mp.__version__,
            "numpy": np.__version__,
        },
        "outputs": [{"path": str(path), "sha256": sha256(path), "probe": probe(path)} for path in outputs],
    }


def command_transfer(args) -> None:
    rights = rights_from_args(args)
    if not args.model.exists():
        raise FileNotFoundError(f"Missing face-landmarker model: {args.model}")
    settings = ExpressionSettings(
        mode=args.mode,
        eye_core_alpha=args.eye_core_alpha,
        brow_alpha=args.brow_alpha,
        mouth_core_alpha=args.mouth_core_alpha,
        lip_alpha=args.lip_alpha,
    )
    settings.validate()
    if args.caption_png:
        caption = cv2.imread(str(args.caption_png), cv2.IMREAD_UNCHANGED)
        if caption is None:
            raise FileNotFoundError(f"Could not read caption PNG: {args.caption_png}")
        if caption.shape[:2] != (405, 720):
            raise ValueError(f"Caption PNG must be 720x405, got {caption.shape[1]}x{caption.shape[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = args.output_dir / "expression-raw.mp4"
    final_output = args.output_dir / "final.mp4"
    contact_sheet = args.output_dir / "contact-sheet.jpg"
    failures = transfer_expressions(
        args.source,
        args.swapped,
        raw_output,
        args.model,
        args.expected_face,
        settings,
    )
    finalize_video(raw_output, args.source, final_output)
    make_contact_sheet(final_output, contact_sheet)
    outputs = [final_output, contact_sheet]
    if args.gif:
        gif_output = args.output_dir / "telegram.gif"
        make_gif(final_output, gif_output, args.caption_png, args.gif_fps)
        outputs.append(gif_output)
    manifest_path = args.output_dir / "manifest.json"
    manifest = _manifest(args, rights, settings, failures, outputs)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="natural-faceswap",
        description="Restore natural eye and mouth performance in an authorized face swap.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a media file with FFprobe.")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.set_defaults(handler=command_inspect)

    model_parser = subparsers.add_parser("download-model", help="Download Google's face-landmarker model.")
    model_parser.add_argument("--output", type=Path, default=Path("models/face_landmarker.task"))
    model_parser.add_argument("--force", action="store_true")
    model_parser.set_defaults(handler=command_download_model)

    facefusion_parser = subparsers.add_parser(
        "facefusion",
        help="Print or run the FaceFusion base-swap command.",
    )
    facefusion_parser.add_argument("--facefusion-home", type=Path, required=True)
    facefusion_parser.add_argument("--reference", type=Path, nargs="+", required=True)
    facefusion_parser.add_argument("--target", type=Path, required=True)
    facefusion_parser.add_argument("--output", type=Path, required=True)
    facefusion_parser.add_argument("--provider", choices=["cpu", "coreml", "cuda"], default="cuda")
    facefusion_parser.add_argument("--face-swapper-model", default="hyperswap_1a_256")
    facefusion_parser.add_argument("--selector-mode", choices=["one", "reference"], default="one")
    facefusion_parser.add_argument("--reference-frame-number", type=int, default=0)
    facefusion_parser.add_argument("--reference-face-position", type=int, default=0)
    facefusion_parser.add_argument("--reference-face-distance", type=float, default=0.6)
    facefusion_parser.add_argument("--run", action="store_true")
    add_rights_arguments(facefusion_parser)
    facefusion_parser.set_defaults(handler=command_facefusion)

    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Transfer natural expressions and package outputs.",
    )
    transfer_parser.add_argument("--source", type=Path, required=True)
    transfer_parser.add_argument("--swapped", type=Path, required=True)
    transfer_parser.add_argument("--output-dir", type=Path, required=True)
    transfer_parser.add_argument("--model", type=Path, required=True)
    transfer_parser.add_argument("--expected-face", type=parse_point, default=(0.5, 0.35))
    transfer_parser.add_argument("--mode", choices=["eyes", "mouth", "both"], default="both")
    transfer_parser.add_argument("--eye-core-alpha", type=float, default=0.78)
    transfer_parser.add_argument("--brow-alpha", type=float, default=0.28)
    transfer_parser.add_argument("--mouth-core-alpha", type=float, default=0.98)
    transfer_parser.add_argument("--lip-alpha", type=float, default=0.48)
    transfer_parser.add_argument("--gif", action="store_true")
    transfer_parser.add_argument("--gif-fps", type=int, default=12)
    transfer_parser.add_argument("--caption-png", type=Path)
    transfer_parser.add_argument("--source-url")
    transfer_parser.add_argument("--base-tool", default="FaceFusion")
    transfer_parser.add_argument("--base-model", default="hyperswap_1a_256")
    transfer_parser.add_argument("--facefusion-commit", default=RECOMMENDED_FACEFUSION_COMMIT)
    add_rights_arguments(transfer_parser)
    transfer_parser.set_defaults(handler=command_transfer)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
