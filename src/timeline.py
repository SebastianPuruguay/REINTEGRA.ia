from __future__ import annotations

from collections import defaultdict
from typing import Any

TRACK_GAP_SECONDS = 1.0


def group_appearances(
    detections: list[dict[str, Any]], gap_seconds: float = TRACK_GAP_SECONDS
) -> list[dict[str, Any]]:
    """Agrupa detecciones contiguas por clase e ID; sin ID agrupa por clase."""
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        groups[(detection["class_name"], detection.get("track_id"))].append(detection)

    appearances: list[dict[str, Any]] = []
    for (_, track_id), items in groups.items():
        items.sort(key=lambda item: item["timestamp_seconds"])
        segment = [items[0]]
        for item in items[1:]:
            if item["timestamp_seconds"] - segment[-1]["timestamp_seconds"] <= gap_seconds:
                segment.append(item)
            else:
                appearances.append(_make_appearance(segment, track_id))
                segment = [item]
        appearances.append(_make_appearance(segment, track_id))

    appearances.sort(key=lambda item: item["start"])
    return appearances


def _make_appearance(
    segment: list[dict[str, Any]], track_id: int | None
) -> dict[str, Any]:
    start = float(segment[0]["timestamp_seconds"])
    end = float(segment[-1]["timestamp_seconds"])
    return {
        "track_id": track_id,
        "class_name": segment[0]["class_name"],
        "instrument": segment[0]["label_es"],
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(max(0.0, end - start), 2),
        "max_confidence": round(max(item["confidence"] for item in segment), 4),
    }


def build_inventory(appearances: list[dict[str, Any]]) -> dict[str, Any]:
    """Cuenta IDs únicos; si faltan, reporta apariciones sin afirmar objetos físicos."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for appearance in appearances:
        by_class[appearance["instrument"]].append(appearance)

    classes: dict[str, int] = {}
    methods: dict[str, str] = {}
    definitive = True
    for instrument, items in sorted(by_class.items()):
        track_ids = {item["track_id"] for item in items if item["track_id"] is not None}
        without_id = [item for item in items if item["track_id"] is None]
        if track_ids:
            classes[instrument] = len(track_ids)
            methods[instrument] = (
                "IDs únicos; detecciones sin ID omitidas"
                if without_id else "IDs únicos"
            )
            if without_id:
                definitive = False
        else:
            classes[instrument] = len(items)
            methods[instrument] = "Apariciones detectadas"
            definitive = False

    return {
        "total_objects": sum(classes.values()),
        "classes": classes,
        "count_method": methods,
        "is_physical_count": definitive,
    }
