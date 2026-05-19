# Document AI Pipeline

> End-to-end extraction pipeline for handwritten form images containing barcodes, Korean/Latin names, user IDs, and category indicators.

**[Project Overview & Architecture →](https://arslankazmi.github.io/doc-ai-pipeline/)**

## Features
- Three swappable extraction backends: Qwen2-VL-2B, Donut, PaddleOCR + TrOCR
- Confidence-based routing — low-confidence images queued for human review
- HITL feedback loop — human corrections flow back into retraining
- Docker Compose deployment (two isolated services)
- Built-in benchmark suite comparing all three models

## Quick Start

### Local (uv)
```bash
uv sync
# Terminal 1 — Inference API
PYTHONPATH=src uv run uvicorn pipeline.api:app --port 8000 --reload
# Terminal 2 — HITL Review App
PYTHONPATH=src uv run uvicorn hitl.app:app --port 8001 --reload
```
Open: http://localhost:8000/demo

### Docker
```bash
docker compose up --build
```
- Inference demo: http://localhost:8000/demo
- HITL queue: http://localhost:8001

## Architecture
See [docs/demo.html](docs/demo.html) for the full interactive walkthrough.

## Pipeline Flow
```
Image -> Barcode Reader -> Layout Detector -> Extractor(s) -> Confidence Scorer
                                                                    |
                                                    >=0.85: Return result
                                                    <0.85: Queue for HITL review -> Retrain
```

## Project Structure
```
src/
├── pipeline/          # Inference API (port 8000)
│   ├── extractors/    # qwen2vl, donut, paddle_trocr
│   ├── api.py         # FastAPI app + demo UI
│   ├── orchestrator.py
│   ├── confidence.py  # Platt calibration + routing
│   └── case_store.py  # Shared SQLite
└── hitl/              # HITL Review App (port 8001)
    └── app.py
templates/             # Jinja2 HTML templates
scripts/               # calibrate, benchmark, export, retrain
tests/                 # pytest suite
```

## Feedback Loop
1. Low-confidence extractions -> `data/cases.db` -> HITL queue at :8001
2. Reviewer approves/corrects -> `python3 scripts/export_feedback.py`
3. Merge + retrain -> `python3 scripts/retrain.py`
