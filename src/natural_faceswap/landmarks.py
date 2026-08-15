from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

EYES = {
    "right": {
        "eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
        "brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "iris": [468, 469, 470, 471, 472],
    },
    "left": {
        "eye": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
        "brow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
        "iris": [473, 474, 475, 476, 477],
    },
}

OUTER_LIPS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]
INNER_LIPS = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
]
FACE_ANCHORS = [33, 263, 1, 98, 327, 152]


def parse_point(value: str) -> tuple[float, float]:
    try:
        x_raw, y_raw = value.split(",", 1)
        point = float(x_raw), float(y_raw)
    except (ValueError, TypeError) as error:
        raise ValueError("Expected a normalized point in x,y form.") from error
    if not all(0.0 <= component <= 1.0 for component in point):
        raise ValueError("Expected-face coordinates must be between 0 and 1.")
    return point


def make_landmarker(model_path: Path):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def select_face(face_landmarks, width: int, height: int, expected: tuple[float, float], previous):
    if not face_landmarks:
        return None
    candidates = [
        np.array([(point.x * width, point.y * height) for point in face], dtype=np.float32)
        for face in face_landmarks
    ]
    target = np.array([expected[0] * width, expected[1] * height], dtype=np.float32)
    if previous is not None:
        target = previous.mean(axis=0) * 0.82 + target * 0.18
    return min(candidates, key=lambda points: float(np.linalg.norm(points.mean(axis=0) - target)))


def detect(landmarker, frame: np.ndarray, timestamp_ms: int, expected, previous):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    return select_face(result.face_landmarks, frame.shape[1], frame.shape[0], expected, previous)
