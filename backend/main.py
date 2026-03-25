"""
Vehicle Classification API
--------------------------
FastAPI backend that receives an image, runs YOLOv8 inference to detect
and classify vehicles, and returns the annotated image with classification
results (label + confidence score).

Supported vehicle classes (COCO subset used by YOLOv8):
    - Car          (COCO class 2)
    - Motorcycle   (COCO class 3)
    - Bus          (COCO class 5)
    - Truck        (COCO class 7)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO class IDs that correspond to vehicles we want to detect.
VEHICLE_CLASS_IDS: set[int] = {2, 3, 5, 7}

# Human-readable labels for the vehicle classes we care about.
VEHICLE_LABELS: dict[int, str] = {
    2: "Carro",
    3: "Motocicleta",
    5: "Ônibus",
    7: "Caminhão",
}

# Bounding-box colour (BGR) used when drawing results onto the image.
BOX_COLOR = (0, 200, 50)   # green
TEXT_COLOR = (255, 255, 255)  # white
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Minimum confidence threshold; detections below this are ignored.
CONFIDENCE_THRESHOLD = 0.25

# Maximum file size accepted (10 MB).
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Maximum image pixel count (25 MP). Images whose decompressed size would
# exceed this limit are rejected before they reach the model, protecting
# the server against decompression-bomb attacks.
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Allowed MIME types.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------

# Path (or name) of the YOLOv8 weights file.
# Override via MODEL_WEIGHTS env var to point at a local file or a different
# variant (e.g. yolov8s.pt) without touching the source code.
MODEL_WEIGHTS: str = os.getenv("MODEL_WEIGHTS", "yolov8n.pt")

# Comma-separated list of allowed CORS origins.
# Defaults to "*" for local development; restrict in production, e.g.:
#   ALLOWED_ORIGINS=http://localhost:8501,https://myapp.example.com
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: list[str] = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

# ---------------------------------------------------------------------------
# Global model container
# ---------------------------------------------------------------------------

class ModelContainer:
    """Holds the loaded YOLOv8 model so it is initialised only once."""

    model: YOLO | None = None


model_container = ModelContainer()


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLOv8 model at startup and release it on shutdown."""
    logger.info("Carregando modelo YOLOv8 (%s) …", MODEL_WEIGHTS)
    start = time.perf_counter()
    # The weights path/name is read from the MODEL_WEIGHTS env var (default:
    # "yolov8n.pt"). On first run the file is downloaded automatically; in
    # offline environments set MODEL_WEIGHTS to the absolute path of a
    # pre-downloaded file.
    model_container.model = YOLO(MODEL_WEIGHTS)
    elapsed = time.perf_counter() - start
    logger.info("Modelo carregado em %.2f s", elapsed)
    yield
    logger.info("Encerrando aplicação …")
    model_container.model = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vehicle Classification API",
    description="Classifica veículos em imagens usando YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Origins are read from the ALLOWED_ORIGINS env var (default: "*").
    # In production set ALLOWED_ORIGINS to a comma-separated list, e.g.:
    #   ALLOWED_ORIGINS=http://localhost:8501,https://myapp.example.com
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_upload(file: UploadFile, content: bytes) -> None:
    """
    Validate that the uploaded file is an acceptable image.

    Raises:
        HTTPException: if validation fails.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Tipo de arquivo não suportado: '{file.content_type}'. "
                f"Use JPEG, PNG, WEBP ou BMP."
            ),
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Arquivo muito grande. O tamanho máximo permitido é 10 MB.",
        )


def decode_image(content: bytes) -> np.ndarray:
    """
    Decode raw bytes into an OpenCV BGR image array.

    Raises:
        HTTPException 413: if the image exceeds MAX_IMAGE_PIXELS (decompression
            bomb protection).
        HTTPException 422: if the bytes cannot be decoded as a valid image.
    """
    try:
        pil_image = Image.open(io.BytesIO(content))
        # Image.open() is lazy — the header is read but pixel data is not yet
        # decompressed. .size returns the declared dimensions without loading
        # pixels, letting us enforce our limit before any decompression occurs.
        width, height = pil_image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Imagem muito grande após descompressão. "
                    f"O limite é {MAX_IMAGE_PIXELS:,} pixels."
                ),
            )
        pil_image = pil_image.convert("RGB")
        # Convert PIL RGB → OpenCV BGR
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise  # propagate our own HTTP errors unchanged
    except Image.DecompressionBombError as exc:
        logger.warning("Imagem rejeitada por decompression bomb: %s", exc)
        raise HTTPException(
            status_code=413,
            detail=(
                f"Imagem muito grande após descompressão. "
                f"O limite é {MAX_IMAGE_PIXELS:,} pixels."
            ),
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.error("Falha ao decodificar imagem: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Não foi possível decodificar a imagem. Verifique se o arquivo não está corrompido.",
        ) from exc
    except Exception as exc:
        logger.error("Erro inesperado ao decodificar imagem: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Não foi possível decodificar a imagem. Verifique se o arquivo não está corrompido.",
        ) from exc


def run_inference(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """
    Run YOLOv8 inference and return only vehicle detections.

    Each result dict contains:
        - label       (str)   – Portuguese vehicle name
        - confidence  (float) – 0–1 confidence score
        - box         (list)  – [x1, y1, x2, y2] pixel coordinates
        - class_id    (int)   – COCO class index

    Raises:
        HTTPException 503: if the model has not yet been loaded.
        HTTPException 500: if the model prediction raises an unexpected error.
    """
    if model_container.model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado. Tente novamente em instantes.")

    try:
        results = model_container.model.predict(
            source=image_bgr,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
    except Exception as exc:
        logger.error("Falha na inferência: %s", exc)
        raise HTTPException(status_code=500, detail="Erro durante a inferência do modelo.") from exc

    detections: list[dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASS_IDS:
                continue  # Skip non-vehicle detections

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

            detections.append(
                {
                    "label": VEHICLE_LABELS[class_id],
                    "confidence": round(confidence, 4),
                    "box": [x1, y1, x2, y2],
                    "class_id": class_id,
                }
            )

    # Sort by confidence descending so the best detection comes first.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_detections(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """
    Draw bounding boxes and labels onto the image and return the annotated copy.
    """
    annotated = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]
        text = f"{label} {conf * 100:.1f}%"

        # Draw filled rectangle as background for the text label.
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - baseline - 4), (x1 + text_w + 4, y1), BOX_COLOR, -1)

        # Draw bounding box.
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Draw label text.
        cv2.putText(annotated, text, (x1 + 2, y1 - baseline - 2), FONT, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

    return annotated


def encode_image_base64(image_bgr: np.ndarray) -> str:
    """Encode an OpenCV BGR image to a base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem de resultado.")
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Returns 200 OK when the service is ready."""
    status = "ready" if model_container.model is not None else "loading"
    return {"status": status}


@app.post("/classify", summary="Classify vehicles in an image")
async def classify_vehicle(file: UploadFile = File(...)) -> JSONResponse:
    """
    Receive an image file, detect vehicles using YOLOv8, annotate the image
    with bounding boxes, and return the results.

    Response JSON schema:
    ```json
    {
        "detections": [
            {
                "label":      "Carro",
                "confidence": 0.92,
                "box":        [x1, y1, x2, y2],
                "class_id":   2
            }
        ],
        "annotated_image": "<base64-encoded JPEG>",
        "vehicles_found":  true
    }
    ```
    """
    # 1. Read raw bytes.
    content = await file.read()

    # 2. Validate format and size.
    validate_upload(file, content)

    # 3. Decode to OpenCV array.
    image_bgr = decode_image(content)

    # 4. Run model inference.
    detections = run_inference(image_bgr)

    # 5. Annotate the image (even if no vehicles were found, return the clean image).
    annotated = draw_detections(image_bgr, detections)

    # 6. Encode annotated image to base64.
    annotated_b64 = encode_image_base64(annotated)

    return JSONResponse(
        content={
            "detections": detections,
            "annotated_image": annotated_b64,
            "vehicles_found": len(detections) > 0,
        }
    )
