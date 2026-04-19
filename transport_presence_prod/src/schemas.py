from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_path: str
    device: str
    roi: tuple[int, int, int, int]
    classes: list[int]
    frame_stride: int


class DetectionRequest(BaseModel):
    video_path: str = Field(..., description="Path to a video file accessible inside the container/server")
    output_name: Optional[str] = Field(default=None, description="Optional name for annotated output file")
    save_output: bool = True
    max_frames: Optional[int] = None
    frame_stride: Optional[int] = None


class DetectionSummary(BaseModel):
    input_path: str
    output_path: Optional[str]
    frames_total: int
    frames_processed: int
    frames_with_transport: int
    transport_present_any: bool
    first_detected_frame: Optional[int]
    first_detected_second: Optional[float]
    fps: float
    roi: tuple[int, int, int, int]
    classes: list[int]
    model_path: str
