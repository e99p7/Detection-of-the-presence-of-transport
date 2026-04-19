from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO

from src.config import Settings


@dataclass
class FrameDetection:
    transport_detected: bool
    boxes_in_roi: list[tuple[int, int, int, int, float, int]]
    boxes_all: list[tuple[int, int, int, int, float, int]]


class TransportDetector:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.model = YOLO(self.settings.model_path)
        self.roi = self.settings.roi
        self.classes = list(self.settings.classes)

    def detect_frame(self, frame) -> FrameDetection:
        results = self.model.predict(
            source=frame,
            conf=self.settings.confidence,
            iou=self.settings.iou,
            imgsz=self.settings.image_size,
            classes=self.classes,
            device=self.settings.device,
            verbose=False,
        )
        result = results[0]

        boxes_all: list[tuple[int, int, int, int, float, int]] = []
        boxes_in_roi: list[tuple[int, int, int, int, float, int]] = []
        x_min, y_min, x_max, y_max = self.roi

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                boxes_all.append((x1, y1, x2, y2, conf, cls_id))

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if x_min < center_x < x_max and y_min < center_y < y_max:
                    boxes_in_roi.append((x1, y1, x2, y2, conf, cls_id))

        return FrameDetection(
            transport_detected=bool(boxes_in_roi),
            boxes_in_roi=boxes_in_roi,
            boxes_all=boxes_all,
        )

    def annotate_frame(self, frame, detection: FrameDetection):
        annotated = frame.copy()
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for bx1, by1, bx2, by2, conf, cls_id in detection.boxes_all:
            in_roi = any(
                bx1 == rx1 and by1 == ry1 and bx2 == rx2 and by2 == ry2
                for rx1, ry1, rx2, ry2, _, _ in detection.boxes_in_roi
            )
            color = (0, 255, 0) if in_roi else (0, 200, 255)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
            label = f"cls={cls_id} conf={conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (bx1, max(20, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        status = "TRANSPORT DETECTED" if detection.transport_detected else "NO TRANSPORT"
        status_color = (0, 255, 0) if detection.transport_detected else (0, 0, 255)
        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
        )
        return annotated

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
        frame_stride: int | None = None,
    ) -> dict:
        input_path = str(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        stride = max(1, frame_stride or self.settings.frame_stride)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        writer = None
        if output_path:
            output_path = str(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25.0, (width, height))

        frames_total = 0
        frames_processed = 0
        frames_with_transport = 0
        first_detected_frame = None
        last_detection = FrameDetection(False, [], [])

        try:
            while True:
                if max_frames is not None and frames_total >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                frames_total += 1

                should_run = ((frames_total - 1) % stride == 0)
                if should_run:
                    last_detection = self.detect_frame(frame)
                    frames_processed += 1
                    if last_detection.transport_detected:
                        frames_with_transport += 1
                        if first_detected_frame is None:
                            first_detected_frame = frames_total

                if writer is not None:
                    writer.write(self.annotate_frame(frame, last_detection))
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        first_second = None
        if first_detected_frame is not None and fps > 0:
            first_second = round(first_detected_frame / fps, 3)

        return {
            "input_path": input_path,
            "output_path": str(output_path) if output_path else None,
            "frames_total": frames_total,
            "frames_processed": frames_processed,
            "frames_with_transport": frames_with_transport,
            "transport_present_any": first_detected_frame is not None,
            "first_detected_frame": first_detected_frame,
            "first_detected_second": first_second,
            "fps": fps,
            "roi": self.roi,
            "classes": self.classes,
            "model_path": self.settings.model_path,
        }
