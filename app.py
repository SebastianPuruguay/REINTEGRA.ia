from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from src.detector import SurgicalInstrumentDetector
from src.image_processor import process_image
from src.labels_es import INSTRUMENT_DESCRIPTIONS_ES, LABELS_ES
from src.timeline import TRACK_GAP_SECONDS, build_inventory, group_appearances
from src.utils import ensure_directories, format_timestamp, save_json
from src.video_processor import process_video

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"
OUTPUT_DIR = APP_DIR / "outputs"
TEMP_DIR = APP_DIR / "temp"
PROCESSED_VIDEO_PATH = OUTPUT_DIR / "processed_video.mp4"
PROCESSED_IMAGE_PATH = OUTPUT_DIR / "processed_image.jpg"
DETECTIONS_PATH = OUTPUT_DIR / "detections.json"
INVENTORY_PATH = OUTPUT_DIR / "inventory.json"
DEFAULT_CONFIDENCE = 0.50

ensure_directories(OUTPUT_DIR, TEMP_DIR)

st.set_page_config(page_title="REINTEGRA.ai", page_icon="🔬", layout="wide")
st.title("REINTEGRA.ai")
st.subheader("Inteligencia para Inventario Quirúrgico")
st.caption("Detección y seguimiento local con el modelo preentrenado del proyecto.")

with st.expander("Ver los 14 instrumentos que reconoce el modelo"):
    st.caption(
        "La detección se limita a estas categorías. La descripción es una referencia breve."
    )
    instrument_columns = st.columns(2)
    for index, (class_name, label) in enumerate(LABELS_ES.items()):
        with instrument_columns[index % 2]:
            st.markdown(
                f"**{label}**  \n{INSTRUMENT_DESCRIPTIONS_ES[class_name]}"
            )

uploaded_file = st.file_uploader(
    "Subir imagen o video", type=["mp4", "jpg", "jpeg", "png"]
)
confidence = st.slider(
    "Confianza mínima", min_value=0.10, max_value=0.95,
    value=DEFAULT_CONFIDENCE, step=0.05
)

if st.button("Analizar archivo", type="primary", disabled=uploaded_file is None):
    temporary_path = None
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as temporary:
            temporary.write(uploaded_file.getbuffer())
            temporary_path = Path(temporary.name)

        detector = SurgicalInstrumentDetector(MODEL_PATH)
        if suffix == ".mp4":
            progress = st.progress(0, text="Analizando video... 0%")

            def update_progress(value: float) -> None:
                percent = int(value * 100)
                progress.progress(percent, text=f"Analizando video... {percent}%")

            result = process_video(
                temporary_path, PROCESSED_VIDEO_PATH,
                detector, confidence, update_progress
            )
            appearances = group_appearances(result["detections"], TRACK_GAP_SECONDS)
            inventory = build_inventory(appearances)
            media_type = "video"
            progress.empty()
        else:
            with st.spinner("Analizando imagen..."):
                result = process_image(
                    temporary_path, PROCESSED_IMAGE_PATH, detector, confidence
                )
            appearances = []
            class_counts = dict(sorted(Counter(
                item["label_es"] for item in result["detections"]
            ).items()))
            inventory = {
                "total_objects": len(result["detections"]),
                "classes": class_counts,
                "count_method": {
                    instrument: "Detecciones en imagen" for instrument in class_counts
                },
                "is_physical_count": False,
            }
            media_type = "image"

        save_json(DETECTIONS_PATH, result["detections"])
        save_json(INVENTORY_PATH, inventory)
        st.session_state.analysis = {
            **result, "appearances": appearances, "inventory": inventory,
            "media_type": media_type,
        }
        st.success("Análisis completado.")
    except Exception as error:
        st.error(f"No se pudo completar el análisis: {error}")
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

analysis = st.session_state.get("analysis")
if analysis:
    detections = analysis["detections"]
    appearances = analysis["appearances"]
    inventory = analysis["inventory"]
    average_confidence = (
        sum(item["confidence"] for item in detections) / len(detections)
        if detections else 0.0
    )

    st.divider()
    st.header("Resumen del análisis")
    col1, col2, col3, col4 = st.columns(4)
    if analysis["media_type"] == "video":
        col1.metric("Duración del video", format_timestamp(analysis["duration_seconds"]))
        col2.metric("Instrumentos detectados", len(inventory["classes"]))
        col3.metric(
            "Objetos únicos" if inventory["is_physical_count"] else "Apariciones estimadas",
            inventory["total_objects"],
        )
    else:
        col1.metric("Dimensiones", f"{analysis['width']} × {analysis['height']}")
        col2.metric("Tipos de instrumento", len(inventory["classes"]))
        col3.metric("Instrumentos detectados", inventory["total_objects"])
    col4.metric("Confianza promedio", f"{average_confidence:.0%}")

    if analysis["media_type"] == "video":
        st.header("Video analizado")
        st.video(str(PROCESSED_VIDEO_PATH), format="video/mp4")
        st.download_button(
            "Descargar video analizado",
            data=PROCESSED_VIDEO_PATH.read_bytes(),
            file_name="reintegra_video_analizado.mp4",
            mime="video/mp4",
        )
        evidence = analysis.get("evidence", [])
        if evidence:
            st.subheader("Hallazgos en el video")
            st.caption(
                "Estos fotogramas muestran las cajas y etiquetas dibujadas en el momento de la detección."
            )
            evidence_columns = st.columns(3)
            for index, item in enumerate(evidence):
                with evidence_columns[index % 3]:
                    st.image(
                        item["path"],
                        caption=(
                            f"{item['timestamp']} · "
                            f"{', '.join(item['instruments'])}"
                        ),
                        use_container_width=True,
                    )
    else:
        st.header("Imagen analizada")
        _, image_column, _ = st.columns([1, 2, 1])
        with image_column:
            st.image(str(PROCESSED_IMAGE_PATH), use_container_width=True)

    st.header("Inventario detectado")
    if analysis["media_type"] == "video" and not inventory["is_physical_count"]:
        st.info(
            "El video sí contiene las cajas y etiquetas de todas las detecciones. "
            "ByteTrack no asignó ID a algunas apariciones, por lo que esas cantidades "
            "no deben interpretarse como un conteo físico definitivo."
        )
    inventory_rows = [
        {"Instrumento": instrument, "Cantidad": quantity,
         "Método": inventory["count_method"][instrument]}
        for instrument, quantity in inventory["classes"].items()
    ]
    st.dataframe(pd.DataFrame(inventory_rows), use_container_width=True, hide_index=True)

    if analysis["media_type"] == "video":
        st.header("Apariciones detectadas")
        appearance_rows = [
            {
                "Instrumento": item["instrument"],
                "ID": item["track_id"] if item["track_id"] is not None else "Sin ID",
                "Primera aparición": format_timestamp(item["start"], True),
                "Última aparición": format_timestamp(item["end"], True),
                "Duración visible": f"{item['duration']:.1f} s",
                "Confianza máxima": f"{item['max_confidence']:.0%}",
            }
            for item in appearances
        ]
        st.dataframe(
            pd.DataFrame(appearance_rows), use_container_width=True, hide_index=True
        )

        st.header("Buscar instrumento")
        instruments = sorted({item["instrument"] for item in appearances})
        if instruments:
            selected = st.selectbox("Seleccionar instrumento", instruments)
            matches = [item for item in appearances if item["instrument"] == selected]
            st.write(f"**{selected}: apariciones encontradas**")
            for item in matches:
                id_text = (
                    f" · ID {item['track_id']}" if item["track_id"] is not None else ""
                )
                st.write(
                    f"{format_timestamp(item['start'], True)} - "
                    f"{format_timestamp(item['end'], True)}{id_text}"
                )
        else:
            st.info("No se encontraron instrumentos con la confianza seleccionada.")
    else:
        st.header("Detecciones en la imagen")
        detection_rows = [
            {
                "Instrumento": item["label_es"],
                "Confianza": f"{item['confidence']:.0%}",
                "Ubicación": (
                    f"({item['bbox']['x1']}, {item['bbox']['y1']}) – "
                    f"({item['bbox']['x2']}, {item['bbox']['y2']})"
                ),
            }
            for item in detections
        ]
        if detection_rows:
            st.dataframe(
                pd.DataFrame(detection_rows), use_container_width=True, hide_index=True
            )
        else:
            st.info("No se encontraron instrumentos con la confianza seleccionada.")
