"""
Merge feedback into training data and optionally trigger retraining.
Run from repo root: python3 scripts/retrain.py [--auto]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent  # scripts/retrain.py → 2 levels up → repo root


def run_export_feedback(db: str, output: str) -> int:
    """Run export_feedback.py as a subprocess. Returns exit code."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "export_feedback.py"),
        "--db", db,
        "--output", output,
    ]
    result = subprocess.run(cmd)
    return result.returncode


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(str(path), "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(str(path), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge feedback data and optionally retrain.")
    parser.add_argument("--auto", action="store_true", help="Automatically run convert_dataset.py and train.py after merge.")
    parser.add_argument(
        "--feedback",
        default=str(_REPO_ROOT / "data" / "feedback.jsonl"),
        help="Path to feedback JSONL",
    )
    parser.add_argument(
        "--dataset",
        default=str(_REPO_ROOT / "data" / "train" / "data_augmented.jsonl"),
        help="Path to training dataset JSONL",
    )
    args = parser.parse_args()

    feedback_path = Path(args.feedback)
    dataset_path = Path(args.dataset)

    # Step 1: Export feedback from HITL case store
    print("Step 1: Exporting reviewed HITL cases...")
    db_path = str(_REPO_ROOT / "data" / "cases.db")
    exit_code = run_export_feedback(db=db_path, output=str(feedback_path))
    if exit_code != 0:
        print(f"export_feedback.py exited with code {exit_code}. Continuing anyway.")

    # Step 2: Merge feedback into dataset
    feedback_rows = load_jsonl(feedback_path)
    if not feedback_rows:
        print(f"No feedback rows found at {feedback_path}. Skipping merge.")
    else:
        print(f"\nStep 2: Merging {len(feedback_rows)} feedback examples into dataset...")

        original_rows = load_jsonl(dataset_path)
        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}. Will create from feedback only.")

        # Build dict keyed by image path; feedback overrides original
        merged: dict[str, dict] = {}
        for row in original_rows:
            key = row.get("image", "")
            merged[key] = row
        feedback_count = 0
        for row in feedback_rows:
            key = row.get("image", "")
            if key not in merged:
                feedback_count += 1
            merged[key] = row  # feedback overrides

        merged_rows = list(merged.values())

        # Backup original
        if dataset_path.exists():
            bak_path = dataset_path.with_suffix(".bak.jsonl")
            shutil.copy2(str(dataset_path), str(bak_path))
            print(f"Backed up original dataset to: {bak_path}")

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(dataset_path, merged_rows)
        print(f"Merged {len(feedback_rows)} feedback examples into dataset (total: {len(merged_rows)} rows)")

    # Step 3: Print instructions
    print("\nStep 3: Next steps")
    print("To regenerate train/eval splits:")
    print("  python3 scripts/convert_dataset.py")
    print("To retrain:")
    print("  Local (MPS): python3 scripts/train.py")
    print("  Cloud:       modal run scripts/train_modal.py")

    if args.auto:
        print("\n--auto flag set. Running convert_dataset.py and train.py...")

        convert_path = _REPO_ROOT / "scripts" / "convert_dataset.py"
        if not convert_path.exists():
            print(f"convert_dataset.py not found at {convert_path}. Skipping.")
        else:
            print("\nRunning convert_dataset.py...")
            result = subprocess.run([sys.executable, str(convert_path)])
            if result.returncode != 0:
                print(f"convert_dataset.py failed with exit code {result.returncode}. Aborting retrain.")
                sys.exit(result.returncode)

        train_path = _REPO_ROOT / "scripts" / "train.py"
        if not train_path.exists():
            print(f"train.py not found at {train_path}. Skipping.")
        else:
            print("\nRunning train.py...")
            result = subprocess.run([sys.executable, str(train_path)])
            if result.returncode != 0:
                print(f"train.py failed with exit code {result.returncode}.")
                sys.exit(result.returncode)
            print("Retraining complete.")


if __name__ == "__main__":
    main()
