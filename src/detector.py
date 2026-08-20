from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.labels_es import get_spanish_label


@lru_cache(maxsize=1)
def load_model(model_path: str) -> YOLO:
    """Carga y conserva una única instancia del modelo preentrenado."""
    return YOLO(model_path)


class SurgicalInstrumentDetector:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        self.model = load_model(str(model_path.resolve()))

    def reset_tracking(self) -> None:
        """Reinicia IDs entre videos, si el predictor ya creó trackers."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def track_frame(self, frame: Any, confidence: float) -> list[dict[str, Any]]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            verbose=False,
        )
        return self._extract_detections(results[0])

    def detect_frame(self, frame: Any, confidence: float) -> list[dict[str, Any]]:
        """Detecta instrumentos en una imagen sin iniciar seguimiento."""
        results = self.model.predict(source=frame, conf=confidence, verbose=False)
        return self._extract_detections(results[0])

    def _extract_detections(self, result: Any) -> list[dict[str, Any]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.int().cpu().tolist()
        track_values = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(xyxy_values)
        )

        detections = []
        for xyxy, score, class_id, track_id in zip(
            xyxy_values, confidence_values, class_values, track_values
        ):
            class_name = str(self.model.names[int(class_id)])
            detections.append(
                {
                    "track_id": int(track_id) if track_id is not None else None,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "label_es": get_spanish_label(class_name),
                    "confidence": float(score),
                    "bbox": {
                        "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                    },
                }
            )
        return detections
