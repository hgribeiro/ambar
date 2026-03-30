"""
Vehicle Classification Frontend
--------------------------------
Streamlit application that lets the user upload or capture a photo and
displays the YOLOv8 vehicle classification results returned by the backend.
"""

from __future__ import annotations

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Allow the API URL to be overridden via environment variable (e.g. in Docker).
API_URL = os.getenv("API_URL", "http://localhost:8000")
CLASSIFY_ENDPOINT = f"{API_URL}/classify"
HEALTH_ENDPOINT = f"{API_URL}/health"

# Portuguese labels for supported input methods.
INPUT_METHODS = {
    "📁 Enviar arquivo": "upload",
    "📷 Usar câmera": "camera",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Classificador de Veículos",
    page_icon="🚗",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def check_backend_health() -> bool:
    """Return True if the backend is reachable and the model is loaded."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ready"
    except (requests.RequestException, ValueError):
        return False


def call_classify_api(image_bytes: bytes, filename: str, mime_type: str) -> dict:
    """
    Send the image to the /classify endpoint.

    Returns:
        Parsed JSON response from the backend.

    Raises:
        RuntimeError: on HTTP errors or connection failures.
    """
    try:
        response = requests.post(
            CLASSIFY_ENDPOINT,
            files={"file": (filename, image_bytes, mime_type)},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Não foi possível conectar ao backend. "
            "Certifique-se de que o servidor FastAPI está rodando em "
            f"{API_URL}."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        detail = "Erro desconhecido."
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"Erro do servidor: {detail}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("O servidor demorou demais para responder. Tente novamente.") from exc


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64 JPEG string back into a PIL Image."""
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))


def get_image_bytes_and_meta(source) -> tuple[bytes, str, str]:
    """
    Extract raw bytes, filename, and MIME type from a Streamlit file-like object.
    Works for both UploadedFile (file uploader) and CaptureFile (camera input).
    """
    raw_bytes = source.getvalue()
    # Camera input always returns PNG; uploaded files preserve their type.
    mime = getattr(source, "type", "image/jpeg") or "image/jpeg"
    name = getattr(source, "name", "photo.jpg") or "photo.jpg"
    return raw_bytes, name, mime


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.title("🚗 Classificador de Veículos")
    st.markdown(
        "Envie ou capture uma foto e descubra o tipo de veículo presente na imagem."
    )
    st.divider()


def render_backend_status() -> None:
    with st.spinner("Verificando status do backend …"):
        healthy = check_backend_health()

    if healthy:
        st.success("✅ Backend conectado e modelo pronto.", icon="✅")
    else:
        st.warning(
            "⚠️ Backend não detectado ou modelo ainda carregando. "
            "Inicie o servidor antes de enviar imagens.",
            icon="⚠️",
        )


def render_input_section() -> tuple[bytes | None, str, str]:
    """Render the image input section and return (bytes, filename, mime) or (None, …)."""
    method_label = st.radio(
        "Selecione o método de entrada:",
        options=list(INPUT_METHODS.keys()),
        horizontal=True,
    )
    method = INPUT_METHODS[method_label]

    raw_bytes, filename, mime = None, "photo.jpg", "image/jpeg"

    if method == "upload":
        uploaded = st.file_uploader(
            "Selecione uma imagem",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Formatos aceitos: JPEG, PNG, WEBP, BMP (máx 10 MB).",
        )
        if uploaded:
            raw_bytes, filename, mime = get_image_bytes_and_meta(uploaded)

    else:  # camera
        captured = st.camera_input("Tire uma foto")
        if captured:
            raw_bytes = captured.getvalue()
            mime = "image/png"
            filename = "capture.png"

    return raw_bytes, filename, mime


def render_original_preview(raw_bytes: bytes) -> None:
    """Display the original uploaded image, converting to a PIL Image for compatibility."""
    st.subheader("📸 Imagem original")
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        st.image(image, use_column_width="always")
    except Exception:
        st.image(raw_bytes, use_column_width="always")


def render_results(result: dict) -> None:
    """Display the annotated image and classification results."""
    detections: list[dict] = result.get("detections", [])
    annotated_b64: str = result.get("annotated_image", "")
    vehicles_found: bool = result.get("vehicles_found", False)

    st.subheader("🔍 Resultado da classificação")

    # Show annotated image (with bounding boxes).
    if annotated_b64:
        annotated_image = decode_base64_image(annotated_b64)
        st.image(annotated_image, caption="Imagem anotada com bounding boxes", use_column_width="always")

    if not vehicles_found:
        st.info(
            "ℹ️ Nenhum veículo foi identificado na imagem com confiança suficiente. "
            "Tente com uma foto mais nítida ou com veículo mais visível.",
        )
        return

    st.success(f"✅ {len(detections)} veículo(s) encontrado(s)!")

    for i, det in enumerate(detections, start=1):
        label = det.get("label", "Desconhecido")
        confidence = det.get("confidence", 0.0)
        box = det.get("box", [])

        with st.container(border=True):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"### Veículo #{i}: **{label}**")
                st.progress(confidence, text=f"Confiança: **{confidence * 100:.1f}%**")
            with col2:
                if box:
                    st.markdown(
                        f"**Posição na imagem**  \n"
                        f"x₁={box[0]}, y₁={box[1]}  \n"
                        f"x₂={box[2]}, y₂={box[3]}"
                    )


# ---------------------------------------------------------------------------
# Main application flow
# ---------------------------------------------------------------------------


def main() -> None:
    render_header()
    render_backend_status()
    st.divider()

    raw_bytes, filename, mime = render_input_section()

    if raw_bytes is None:
        st.info("👆 Selecione ou capture uma imagem para iniciar a classificação.")
        return

    render_original_preview(raw_bytes)

    st.divider()
    if st.button("🚀 Classificar Veículo", type="primary", use_container_width=True):
        with st.spinner("Enviando imagem para análise …"):
            try:
                result = call_classify_api(raw_bytes, filename, mime)
                render_results(result)
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
            except Exception as exc:
                st.error(f"❌ Erro inesperado: {exc}")


if __name__ == "__main__":
    main()
