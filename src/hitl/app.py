"""
HITL Review Application — FastAPI service on port 8001.

Human reviewers work through a queue of low-confidence extraction cases,
edit fields, approve or correct, feeding corrections back to the training loop.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pipeline.case_store import (
    claim_case,
    get_case,
    get_stats,
    list_cases,
    resolve_case,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent  # src/hitl/app.py → 3 levels up → repo root
_TEMPLATES = _REPO_ROOT / "templates" / "hitl"
_STATIC = _REPO_ROOT / "static"

app = FastAPI(title="Document AI — HITL Review", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATES))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_persons(extraction: dict | str | None) -> list[dict]:
    """Return the persons list from an extraction dict/string, or []."""
    if extraction is None:
        return []
    if isinstance(extraction, str):
        try:
            extraction = json.loads(extraction)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(extraction, dict):
        persons = extraction.get("persons", [])
        if isinstance(persons, list):
            return persons
    return []


def _confidence_class(conf: float | None) -> str:
    if conf is None:
        return "conf-unknown"
    if conf < 0.5:
        return "conf-red"
    if conf < 0.7:
        return "conf-orange"
    if conf < 0.85:
        return "conf-yellow"
    return "conf-green"


# Expose helper in Jinja2 globals
templates.env.globals["confidence_class"] = _confidence_class


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/queue", status_code=302)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "hitl"}


@app.get("/queue", response_class=HTMLResponse)
async def queue(request: Request):
    cases = list_cases(status="open")
    stats = get_stats()
    return templates.TemplateResponse(
        request,
        "queue.html",
        {
            "cases": cases,
            "stats": stats,
            "confidence_class": _confidence_class,
        },
    )


@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_detail(request: Request, case_id: str):
    case = get_case(case_id)
    if case is None:
        return HTMLResponse(content="<h1>Case not found</h1>", status_code=404)

    # Parse extraction — case_store already deserialises JSON blobs
    extraction = case.get("extraction") or {}
    if isinstance(extraction, str):
        try:
            extraction = json.loads(extraction)
        except (json.JSONDecodeError, TypeError):
            extraction = {}

    persons = _parse_persons(extraction)

    return templates.TemplateResponse(
        request,
        "case.html",
        {
            "case": case,
            "extraction": extraction,
            "persons": persons,
            "confidence_class": _confidence_class,
        },
    )


@app.post("/case/{case_id}/claim")
async def claim(case_id: str):
    claim_case(case_id)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


@app.post("/case/{case_id}/submit")
async def submit(request: Request, case_id: str):
    form = await request.form()

    action = form.get("action", "approve")
    reviewer_notes = form.get("reviewer_notes", "") or ""
    approved = action == "approve"

    # Reconstruct corrected persons list from dynamic form fields
    persons: list[dict] = []
    idx = 0
    while True:
        barcode = form.get(f"barcode_number_{idx}")
        name = form.get(f"name_{idx}")
        user_id = form.get(f"user_id_{idx}")
        category = form.get(f"category_{idx}")
        if barcode is None and name is None and user_id is None and category is None:
            break
        persons.append(
            {
                "barcode_number": barcode or "",
                "name": name or "",
                "user_id": user_id or "",
                "category": category or "",
            }
        )
        idx += 1

    corrected = {"persons": persons} if persons else None

    resolve_case(
        case_id=case_id,
        corrected=corrected,
        notes=str(reviewer_notes),
        approved=approved,
    )

    return RedirectResponse(url="/queue", status_code=303)


@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    raw = get_stats()
    # Normalise keys so template always has all keys
    counts = {
        "open": raw.get("open", 0),
        "in_review": raw.get("in_review", 0),
        "approved": raw.get("approved", 0),
        "corrected": raw.get("corrected", 0),
        "exported": raw.get("exported", 0),
    }
    total = sum(counts.values())
    counts["total"] = total
    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "counts": counts,
            "total": total,
        },
    )
