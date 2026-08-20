from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Callable

import cv2

from src.detector import SurgicalInstrumentDetector
from src.utils import format_timestamp

ProgressCallback = Callable[[float], None]
MAX_EVIDENCE_IMAGES = 30


def process_video(
    input_path: Path,
    output_path: Path,
    detector: SurgicalInstrumentDetector,
    confidence: float,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video subido.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError("El video no contiene una velocidad de cuadros válida.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("No se pudo crear el video procesado.")

    detector.reset_tracking()
    all_detections: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    evidence_directory = output_path.parent / "evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    for previous_image in evidence_directory.glob("*.jpg"):
        previous_image.unlink(missing_ok=True)
    saved_evidence_keys: set[tuple[str, int | None]] = set()
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp_seconds = frame_number / fps
            detections = detector.track_frame(frame, confidence)
            contains_new_evidence = False
            for detection in detections:
                all_detections.append({
                    "frame": frame_number,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "timestamp": format_timestamp(timestamp_seconds),
                    **detection,
                })
                draw_detection(frame, detection)
                evidence_key = (detection["class_name"], detection.get("track_id"))
                if (
                    evidence_key not in saved_evidence_keys
                    and len(evidence) < MAX_EVIDENCE_IMAGES
                ):
                    saved_evidence_keys.add(evidence_key)
                    contains_new_evidence = True

            if contains_new_evidence:
                evidence_path = evidence_directory / f"hallazgo_{len(evidence) + 1:03d}.jpg"
                cv2.imwrite(str(evidence_path), frame)
                evidence.append(
                    {
                        "path": str(evidence_path),
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "timestamp": format_timestamp(timestamp_seconds, True),
                        "instruments": sorted(
                            {detection["label_es"] for detection in detections}
                        ),
                    }
                )
            writer.write(frame)
            frame_number += 1
            if progress_callback and frame_count > 0:
                progress_callback(min(frame_number / frame_count, 1.0))
    finally:
        capture.release()
        writer.release()

    if progress_callback:
        progress_callback(1.0)
    return {
        "detections": all_detections,
        "fps": fps,
        "frames_processed": frame_number,
        "duration_seconds": frame_number / fps,
        "evidence": evidence,
    }


def draw_detection(frame: Any, detection: dict[str, Any]) -> None:
    bbox = detection["bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    track_id = detection.get("track_id")
    id_text = f" | ID {track_id}" if track_id is not None else ""
    label = unicodedata.normalize("NFKD", detection["label_es"]).encode(
        "ascii", "ignore"
    ).decode("ascii")
    text = f"{label}{id_text} | {detection['confidence']:.0%}"
    color = (46, 204, 113)
    font_scale = max(0.65, min(1.05, frame.shape[1] / 1280 * 0.9))
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text_y = max(32, y1 - 8)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        frame, (x1, text_y - text_height - baseline - 4),
        (x1 + text_width + 6, text_y + baseline), color, -1
    )
    cv2.putText(
        frame, text, (x1 + 3, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (15, 23, 42), thickness, cv2.LINE_AA
    )
