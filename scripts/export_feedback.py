"""
Export reviewed HITL cases to feedback.jsonl for training.
Run from repo root: python3 scripts/export_feedback.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent  # scripts/export_feedback.py → 2 levels up → repo root
sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved/corrected HITL cases to feedback.jsonl.")
    parser.add_argument(
        "--db",
        default=str(_REPO_ROOT / "data" / "cases.db"),
        help="Path to cases SQLite DB",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "feedback.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No cases database found at {db_path}. Nothing to export.")
        sys.exit(0)

    # Override the module-level DB_PATH so get_db() uses the CLI arg
    os.environ["CASES_DB_PATH"] = str(db_path.resolve())

    from pipeline.case_store import get_db, mark_exported

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM cases WHERE status IN ('approved', 'corrected') AND exported_at IS NULL"
        ).fetchall()
        rows = [dict(r) for r in rows]
    finally:
        conn.close()

    if not rows:
        print("No new approved/corrected cases to export.")
        sys.exit(0)

    prompt_text = (
        "Extract all person records from this image. "
        "For each person detected, return a JSON object with: "
        "barcode_number, name, user_id, category."
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict] = []
    skipped = 0

    with open(str(output_path), "a") as fh:
        for row in rows:
            status = row.get("status", "")

            if status == "approved":
                completion_raw = row.get("extraction")
            elif status == "corrected":
                completion_raw = row.get("corrected")
            else:
                skipped += 1
                continue

            if completion_raw is None:
                skipped += 1
                continue

            # completion_raw may already be a dict (deserialized by _row_to_dict) or a string
            if isinstance(completion_raw, str):
                try:
                    completion_dict = json.loads(completion_raw)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
            else:
                completion_dict = completion_raw

            persons = completion_dict.get("persons", [])

            out_row = {
                "image": row["image_path"],
                "person_count": len(persons),
                "prompt": prompt_text,
                "completion": json.dumps({"persons": persons}, ensure_ascii=False),
                "source": "hitl_feedback",
                "model": row["model_name"],
            }
            fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            exported_rows.append(row)

    exported_ids = [r["id"] for r in exported_rows]
    mark_exported(exported_ids)

    print(f"Exported {len(exported_rows)} cases to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows (missing or unparseable completion data).")


if __name__ == "__main__":
    main()
