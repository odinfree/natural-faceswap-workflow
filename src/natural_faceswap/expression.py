from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from natural_faceswap.landmarks import (
    EYES,
    FACE_ANCHORS,
    INNER_LIPS,
    OUTER_LIPS,
    detect,
    make_landmarker,
)


@dataclass(frozen=True)
class ExpressionSettings:
    mode: str = "both"
    eye_core_alpha: float = 0.78
    brow_alpha: float = 0.28
    mouth_core_alpha: float = 0.98
    lip_alpha: float = 0.48

    def validate(self) -> None:
        if self.mode not in {"eyes", "mouth", "both"}:
            raise ValueError(f"Unsupported expression mode: {self.mode}")
        for name, value in asdict(self).items():
            if name == "mode":
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


def _color_match(warped: np.ndarray, destination: np.ndarray, mask: np.ndarray) -> np.ndarray:
    active = mask > 0.25
    if active.sum() < 30:
        return warped
    source_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    destination_lab = cv2.cvtColor(destination, cv2.COLOR_BGR2LAB).astype(np.float32)
    adjusted = source_lab.copy()
    for channel in range(3):
        source_values = source_lab[..., channel][active]
        destination_values = destination_lab[..., channel][active]
        source_std = max(float(source_values.std()), 1.0)
        destination_std = max(float(destination_values.std()), 1.0)
        scale = np.clip(destination_std / source_std, 0.72, 1.38)
        adjusted[..., channel] = (source_lab[..., channel] - float(source_values.mean())) * scale + float(
            destination_values.mean()
        )
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)


def _hull_mask(shape, points: np.ndarray, dilation: int, feather: float) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(points.round().astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if dilation > 0:
        size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask.astype(np.float32) / 255.0


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)],
        axis=1,
    )
    return homogeneous @ matrix.T


def _blend_eye(source, destination, source_points, destination_points, definition, settings):
    indices = definition["eye"] + definition["brow"]
    matrix, _ = cv2.estimateAffinePartial2D(
        source_points[indices],
        destination_points[indices],
        method=cv2.LMEDS,
    )
    if matrix is None:
        return destination
    warped = cv2.warpAffine(
        source,
        matrix,
        (destination.shape[1], destination.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    eye_points = destination_points[definition["eye"] + definition["iris"]]
    brow_points = destination_points[definition["eye"] + definition["brow"]]
    core_mask = _hull_mask(destination.shape, eye_points, dilation=6, feather=5.5)
    brow_mask = _hull_mask(destination.shape, brow_points, dilation=5, feather=13.0)
    brow_mask = np.clip(brow_mask - core_mask * 0.72, 0.0, 1.0)
    combined = np.clip(
        core_mask * settings.eye_core_alpha + brow_mask * settings.brow_alpha,
        0.0,
        1.0,
    )
    corrected = _color_match(warped, destination, combined)
    alpha = combined[..., None]
    result = destination.astype(np.float32) * (1.0 - alpha) + corrected.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _blend_mouth(source, destination, source_points, destination_points, settings):
    matrix, _ = cv2.estimateAffinePartial2D(
        source_points[FACE_ANCHORS],
        destination_points[FACE_ANCHORS],
        method=cv2.LMEDS,
    )
    if matrix is None:
        return destination
    warped = cv2.warpAffine(
        source,
        matrix,
        (destination.shape[1], destination.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed = _transform_points(source_points, matrix)
    inner_mask = _hull_mask(destination.shape, transformed[INNER_LIPS], dilation=5, feather=4.0)
    outer_mask = _hull_mask(destination.shape, transformed[OUTER_LIPS], dilation=5, feather=8.0)
    lip_ring = np.clip(outer_mask - inner_mask * 0.74, 0.0, 1.0)
    combined = np.clip(
        inner_mask * settings.mouth_core_alpha + lip_ring * settings.lip_alpha,
        0.0,
        1.0,
    )
    corrected = _color_match(warped, destination, outer_mask)
    alpha = combined[..., None]
    result = destination.astype(np.float32) * (1.0 - alpha) + corrected.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def transfer_expressions(
    source_path: Path,
    swapped_path: Path,
    output_path: Path,
    model_path: Path,
    expected_face: tuple[float, float],
    settings: ExpressionSettings,
) -> list[int]:
    settings.validate()
    source_capture = cv2.VideoCapture(str(source_path))
    swapped_capture = cv2.VideoCapture(str(swapped_path))
    if not source_capture.isOpened() or not swapped_capture.isOpened():
        raise RuntimeError("Could not open one of the input clips.")

    source_count = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    swapped_count = int(swapped_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_count != swapped_count:
        raise ValueError(f"Frame count differs: source={source_count}, swapped={swapped_count}")

    width = int(swapped_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(swapped_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(swapped_capture.get(cv2.CAP_PROP_FPS))
    source_width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (source_width, source_height):
        raise ValueError(
            f"Dimensions differ: source={source_width}x{source_height}, swapped={width}x{height}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")

    source_previous = None
    swapped_previous = None
    failures: list[int] = []
    with make_landmarker(model_path) as source_landmarker, make_landmarker(model_path) as swapped_landmarker:
        for frame_index in range(swapped_count):
            source_ok, source_frame = source_capture.read()
            swapped_ok, swapped_frame = swapped_capture.read()
            if not source_ok or not swapped_ok:
                raise RuntimeError(f"Decode failed at frame {frame_index}")
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            source_points = detect(
                source_landmarker,
                source_frame,
                timestamp_ms,
                expected_face,
                source_previous,
            )
            swapped_points = detect(
                swapped_landmarker,
                swapped_frame,
                timestamp_ms,
                expected_face,
                swapped_previous,
            )
            if source_points is None or swapped_points is None:
                failures.append(frame_index)
                writer.write(swapped_frame)
                continue
            source_previous = source_points
            swapped_previous = swapped_points
            result = swapped_frame
            if settings.mode in {"eyes", "both"}:
                for definition in EYES.values():
                    result = _blend_eye(
                        source_frame,
                        result,
                        source_points,
                        swapped_points,
                        definition,
                        settings,
                    )
            if settings.mode in {"mouth", "both"}:
                result = _blend_mouth(
                    source_frame,
                    result,
                    source_points,
                    swapped_points,
                    settings,
                )
            writer.write(result)

    writer.release()
    source_capture.release()
    swapped_capture.release()
    return failures
