from pathlib import Path

from natural_faceswap.facefusion import FaceFusionConfig, build_command, printable_command


def test_facefusion_command_is_explicit_and_shell_safe() -> None:
    config = FaceFusionConfig(
        home=Path("/tmp/face fusion"),
        references=[Path("refs/front face.jpg"), Path("refs/side.jpg")],
        target=Path("inputs/source.mp4"),
        output=Path("outputs/base.mp4"),
        provider="cuda",
    )
    command = build_command(config)
    assert "hyperswap_1a_256" in command
    assert "cuda" in command
    assert "refs/front face.jpg" in command
    rendered = printable_command(command)
    assert "'refs/front face.jpg'" in rendered


def test_reference_selector_includes_inspected_target() -> None:
    config = FaceFusionConfig(
        home=Path("/tmp/facefusion"),
        references=[Path("refs/front.jpg")],
        target=Path("inputs/source.mp4"),
        output=Path("outputs/base.mp4"),
        selector_mode="reference",
        reference_frame_number=42,
        reference_face_position=1,
        reference_face_distance=0.45,
    )
    command = build_command(config)
    assert command[command.index("--reference-frame-number") + 1] == "42"
    assert command[command.index("--reference-face-position") + 1] == "1"
    assert command[command.index("--reference-face-distance") + 1] == "0.45"
