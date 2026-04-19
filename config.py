from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "yolov8n.pt")
    device: str = os.getenv("DEVICE", "cpu")
    confidence: float = float(os.getenv("CONFIDENCE", "0.25"))
    iou: float = float(os.getenv("IOU", "0.45"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "640"))
    classes: tuple[int, ...] = tuple(
        int(v.strip()) for v in os.getenv("CLASSES", "2,3").split(",") if v.strip()
    )
    # ROI from the user's notebook: x1, y1, x2, y2
    roi: Tuple[int, int, int, int] = tuple(
        int(v.strip()) for v in os.getenv("ROI", "100,300,600,450").split(",") if v.strip()
    )  # type: ignore[assignment]
    frame_stride: int = int(os.getenv("FRAME_STRIDE", "2"))
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")

    @staticmethod
    def from_roi_json(path: str | Path | None) -> "Settings":
        settings = Settings()
        if not path:
            return settings

        config_path = Path(path)
        if not config_path.exists():
            return settings

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        roi = payload.get("roi")
        if not roi or len(roi) != 4:
            return settings

        return Settings(
            model_path=settings.model_path,
            device=settings.device,
            confidence=settings.confidence,
            iou=settings.iou,
            image_size=settings.image_size,
            classes=settings.classes,
            roi=tuple(int(v) for v in roi),
            frame_stride=settings.frame_stride,
            output_dir=settings.output_dir,
        )
