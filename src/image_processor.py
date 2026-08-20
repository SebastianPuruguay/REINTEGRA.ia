from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.detector import SurgicalInstrumentDetector
from src.video_processor import draw_detection


def process_image(
    input_path: Path,
    output_path: Path,
    detector: SurgicalInstrumentDetector,
    confidence: float,
) -> dict[str, Any]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("No se pudo abrir la imagen subida.")

    detections = detector.detect_frame(image, confidence)
    records = []
    for detection in detections:
        records.append(
            {
                "frame": 0,
                "timestamp_seconds": 0.0,
                "timestamp": "00:00",
                **detection,
            }
        )
        draw_detection(image, detection)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError("No se pudo guardar la imagen procesada.")

    return {"detections": records, "width": image.shape[1], "height": image.shape[0]}
