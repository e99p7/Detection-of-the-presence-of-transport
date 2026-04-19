# Transport Presence Detection Production Project

Production-friendly version of the notebook `Detection-of-the-presence-of-transport.ipynb`.

## What it does

- reads a video file or uploaded video
- detects **car** and **motorcycle** objects
- checks whether the **center of a detected object** falls inside a fixed ROI
- returns whether transport was detected in the ROI
- can save an annotated output video

## Project structure

```text
transport_presence_prod/
├── api/
│   └── main.py
├── config/
│   ├── roi.example.json
│   └── roi.json
├── data/
│   └── put_cvtest_avi_here.txt
├── outputs/
├── src/
│   ├── config.py
│   ├── detector.py
│   └── schemas.py
├── process_video.py
├── export_onnx.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Minimal local run

Place your video from the notebook here:

```text
data/cvtest.avi
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run on the video:

```bash
python process_video.py --input data/cvtest.avi
```

Annotated video will be saved to:

```text
outputs/cvtest_annotated.mp4
```

## Start API locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## REST examples

### 1) Detect by server-side path

```bash
curl -X POST "http://localhost:8000/detect/path" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "data/cvtest.avi",
    "save_output": true,
    "output_name": "cvtest_annotated.mp4",
    "max_frames": 300,
    "frame_stride": 2
  }'
```

### 2) Detect by upload

```bash
curl -X POST "http://localhost:8000/detect/upload" \
  -F "file=@data/cvtest.avi"
```

### 3) Download annotated output video

```text
GET /outputs/<filename>
```

## Docker

Build:

```bash
docker build -t transport-detector:latest .
```

Run:

```bash
docker compose up --build
```

## ROI configuration

Default ROI comes from the original notebook:

```json
{
  "roi": [100, 300, 600, 450]
}
```

Change it in `config/roi.json`.

## Optional ONNX export

Once everything works with the default model, you can export the detector model:

```bash
python export_onnx.py --model yolov8n.pt --imgsz 640
```

## Practical deployment advice for low resources

Start with this stack only:

- Python
- Ultralytics YOLO
- FastAPI
- Docker

Do **not** add Spark, Hadoop, Kafka, Kubernetes, Airflow, Oracle, or MLflow in the first version.
They add operational complexity and are unnecessary for one small video inference service.

A good upgrade path is:

1. baseline `.pt` model
2. stable REST + Docker
3. ONNX export if you need lighter CPU inference
4. custom fine-tuning only if your camera angle differs a lot from COCO-style data
