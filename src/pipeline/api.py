"""FastAPI inference API + demo web UI."""
from __future__ import annotations

import base64
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from PIL import Image

from pipeline import orchestrator
from pipeline.case_store import get_stats
from pipeline.confidence import route

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Document AI Pipeline")

_REPO_ROOT = Path(__file__).parent.parent.parent  # src/pipeline/api.py → 3 levels up → repo root
_TEMPLATES = _REPO_ROOT / "templates" / "pipeline"
_STATIC = _REPO_ROOT / "static"
_UPLOADS_DIR = _REPO_ROOT / "data" / "uploads"

templates = Jinja2Templates(directory=str(_TEMPLATES))
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# ---------------------------------------------------------------------------
# In-memory upload cache (max 50 entries)
# ---------------------------------------------------------------------------

_uploads: dict[str, dict[str, Any]] = {}
_MAX_UPLOADS = 50


def _store_upload(upload_id: str, data: dict[str, Any]) -> None:
    if len(_uploads) >= _MAX_UPLOADS:
        oldest_key = next(iter(_uploads))
        _uploads.pop(oldest_key)
    _uploads[upload_id] = data


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "pipeline"}


# ---------------------------------------------------------------------------
# Demo UI endpoints
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse)
async def demo_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/demo/upload")
async def demo_upload(
    request: Request,
    image: UploadFile,
    models: list[str] = Form(default=["qwen2vl"]),
):
    # Ensure at least one model
    if not models:
        models = ["qwen2vl"]

    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    upload_id = str(uuid.uuid4())[:8]
    save_path = _UPLOADS_DIR / f"{upload_id}.jpg"
    pil_image.save(str(save_path), "JPEG", quality=85)

    result_data = orchestrator.process_image(
        pil_image,
        f"uploads/{upload_id}.jpg",
        models,
    )
    _store_upload(upload_id, result_data)

    if len(models) > 1:
        return RedirectResponse(f"/demo/compare/{upload_id}", status_code=303)
    return RedirectResponse(f"/demo/result/{upload_id}/{models[0]}", status_code=303)


@app.get("/demo/result/{upload_id}/{model_name}", response_class=HTMLResponse)
async def demo_result(request: Request, upload_id: str, model_name: str):
    if upload_id not in _uploads:
        return RedirectResponse("/demo", status_code=303)

    data = _uploads[upload_id]
    results = data.get("results", {})

    if model_name not in results:
        return RedirectResponse("/demo", status_code=303)

    result = results[model_name]
    persons = result.get("persons", [])
    confidence = persons[0]["confidence"] if persons else 0.0
    routing = route(confidence)

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "upload_id": upload_id,
            "model_name": model_name,
            "result": result,
            "routing": routing,
            "confidence": confidence,
        },
    )


@app.get("/demo/compare/{upload_id}", response_class=HTMLResponse)
async def demo_compare(request: Request, upload_id: str):
    if upload_id not in _uploads:
        return RedirectResponse("/demo", status_code=303)

    data = _uploads[upload_id]
    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "upload_id": upload_id,
            "results": data.get("results", {}),
            "case_ids": data.get("case_ids", {}),
        },
    )


@app.get("/demo/stats", response_class=HTMLResponse)
async def demo_stats(request: Request):
    stats = get_stats()
    rows = "".join(
        f"<tr><td>{status}</td><td>{count}</td></tr>"
        for status, count in stats.items()
    ) or "<tr><td colspan='2'>No cases yet.</td></tr>"
    total = sum(stats.values())
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Case Stats — Document AI</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f7fa; }}
.container {{ max-width: 700px; margin: 48px auto; background: #fff; border-radius: 10px; padding: 32px; box-shadow: 0 2px 16px rgba(0,0,0,.08); }}
h1 {{ margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #eee; }}
th {{ background: #f0f2f5; font-weight: 600; }}
.total {{ font-weight: 600; margin-top: 12px; }}
a {{ color: #4f46e5; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
  <h1>Case Store Stats</h1>
  <table>
    <thead><tr><th>Status</th><th>Count</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="total">Total: {total}</p>
  <p><a href="/demo">&larr; Back to demo</a></p>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# JSON API endpoint
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    image_b64: str
    models: list[str] | None = None


@app.post("/extract")
async def extract(body: ExtractRequest):
    models = body.models or ["qwen2vl", "donut", "paddle_trocr"]
    try:
        image_bytes = base64.b64decode(body.image_b64)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc

    data = orchestrator.process_image(pil_image, "api_upload", models)
    status_code = 202 if data["any_hard"] else 200
    body_out = {
        "status": "queued" if data["any_hard"] else "ok",
        "case_ids": data["case_ids"],
        "results": data["results"],
    }
    from fastapi.responses import JSONResponse
    return JSONResponse(content=body_out, status_code=status_code)
