from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO model to ONNX.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    exported_path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"Exported ONNX model to: {exported_path}")


if __name__ == "__main__":
    main()
