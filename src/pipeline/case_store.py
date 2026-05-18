"""SQLite-backed case store for HITL review queue.

Shared between the inference API and the human-review application.
All public functions open and close their own connection so the module
is safe for multi-process (and multi-thread) use without any external
connection pooling.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# DB path (env-configurable)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent  # src/pipeline/case_store.py → 3 levels up → repo root
_DEFAULT_DB = str(_REPO_ROOT / "data" / "cases.db")
DB_PATH: str = os.environ.get("CASES_DB_PATH", _DEFAULT_DB)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS cases (
    id              TEXT PRIMARY KEY,
    image_path      TEXT NOT NULL,
    image_b64       TEXT,
    model_name      TEXT NOT NULL,
    extraction      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    status          TEXT DEFAULT 'open',
    corrected       TEXT,
    reviewer_notes  TEXT,
    created_at      TEXT,
    reviewed_at     TEXT,
    exported_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_status     ON cases(status);
CREATE INDEX IF NOT EXISTS idx_confidence ON cases(confidence);
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    """Return a new connection with ``row_factory = sqlite3.Row``.

    Also creates the schema if the database file does not yet exist.
    Callers are responsible for closing the connection.
    """
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    # Deserialize JSON blobs
    for key in ("extraction", "corrected"):
        if d.get(key) is not None:
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def insert_case(
    image_path: str,
    image_b64: str | None,
    model_name: str,
    extraction_dict: dict[str, Any],
    confidence: float,
) -> str:
    """Insert a new review case and return its UUID string."""
    case_id = str(uuid.uuid4())
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO cases
                (id, image_path, image_b64, model_name, extraction,
                 confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                case_id,
                image_path,
                image_b64,
                model_name,
                json.dumps(extraction_dict),
                float(confidence),
                _now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return case_id


def get_case(case_id: str) -> dict[str, Any] | None:
    """Return a single case by *case_id*, or ``None`` if not found."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_cases(
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return up to *limit* cases, optionally filtered by *status*."""
    conn = get_db()
    try:
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cases ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]  # type: ignore[misc]
    finally:
        conn.close()


def claim_case(case_id: str) -> None:
    """Mark a case as ``in_review`` (reviewer has opened it)."""
    conn = get_db()
    try:
        conn.execute(
            "UPDATE cases SET status = 'in_review' WHERE id = ?",
            (case_id,),
        )
        conn.commit()
    finally:
        conn.close()


def resolve_case(
    case_id: str,
    corrected: dict[str, Any] | None,
    notes: str,
    approved: bool,
) -> None:
    """Close a case.

    *approved* ``True``  → ``status = 'approved'``  (extraction was correct).
    *approved* ``False`` → ``status = 'corrected'`` (reviewer supplied fixes).
    *corrected* is serialised as JSON; ``None`` means no changes were needed.
    """
    new_status = "approved" if approved else "corrected"
    conn = get_db()
    try:
        conn.execute(
            """
            UPDATE cases
            SET status         = ?,
                corrected      = ?,
                reviewer_notes = ?,
                reviewed_at    = ?
            WHERE id = ?
            """,
            (
                new_status,
                json.dumps(corrected) if corrected is not None else None,
                notes,
                _now_iso(),
                case_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def mark_exported(case_ids: list[str]) -> None:
    """Stamp *case_ids* with the current UTC time in ``exported_at``."""
    if not case_ids:
        return
    now = _now_iso()
    conn = get_db()
    try:
        conn.executemany(
            "UPDATE cases SET exported_at = ? WHERE id = ?",
            [(now, cid) for cid in case_ids],
        )
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict[str, int]:
    """Return a mapping of ``{status: count}`` across all cases."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM cases GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}
    finally:
        conn.close()
