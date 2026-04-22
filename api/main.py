from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import Settings
from src.detector import TransportDetector
from src.schemas import DetectionRequest, DetectionSummary, HealthResponse


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_roi_json("config/roi.json")


@lru_cache(maxsize=1)
def get_detector() -> TransportDetector:
    return TransportDetector(get_settings())


app = FastAPI(
    title="Transport Presence Detection API",
    version="1.0.0",
    description="Detect car/motorcycle presence inside a fixed ROI on a video stream/file.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model_path=settings.model_path,
        device=settings.device,
        roi=settings.roi,
        classes=list(settings.classes),
        frame_stride=settings.frame_stride,
    )


@app.post("/detect/path", response_model=DetectionSummary)
def detect_by_path(payload: DetectionRequest) -> DetectionSummary:
    detector = get_detector()
    output_path = None
    if payload.save_output:
        output_name = payload.output_name or f"{Path(payload.video_path).stem}_annotated.mp4"
        output_path = str(Path(get_settings().output_dir) / output_name)

    try:
        summary = detector.process_video(
            input_path=payload.video_path,
            output_path=output_path,
            max_frames=payload.max_frames,
            frame_stride=payload.frame_stride,
        )
        return DetectionSummary(**summary)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detect/upload", response_model=DetectionSummary)
def detect_by_upload(
    file: UploadFile = File(...),
) -> DetectionSummary:
    detector = get_detector()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        with input_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        output_name = f"{Path(file.filename or 'video').stem}_{uuid4().hex[:8]}.mp4"
        output_path = str(Path(get_settings().output_dir) / output_name)

        try:
            summary = detector.process_video(
                input_path=input_path,
                output_path=output_path,
            )
            return DetectionSummary(**summary)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/outputs/{filename}")
def get_output_file(filename: str):
    path = Path(get_settings().output_dir) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(path)
