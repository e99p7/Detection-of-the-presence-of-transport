from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import Settings
from src.detector import TransportDetector


def build_output_path(input_path: str, output_dir: str) -> str:
    input_name = Path(input_path).stem
    return str(Path(output_dir) / f"{input_name}_annotated.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect transport presence in a video using ROI logic.")
    parser.add_argument("--input", default="data/cvtest.avi", help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to annotated output video")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--roi-config", default="config/roi.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_roi_json(args.roi_config)
    detector = TransportDetector(settings)

    output_path = args.output or build_output_path(args.input, settings.output_dir)
    summary = detector.process_video(
        input_path=args.input,
        output_path=output_path,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
