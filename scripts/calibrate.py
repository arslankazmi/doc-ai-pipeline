"""
Fits a Platt scaling calibrator on the eval set.
Run from repo root: python3 scripts/calibrate.py

Uses simulated scores for demo purposes (no actual model inference needed).
Replace with real model scores for production calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent  # scripts/calibrate.py → 2 levels up → repo root
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pipeline.confidence import PlattCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a Platt calibrator on simulated eval scores.")
    parser.add_argument("--n", type=int, default=100, help="Number of examples to use (default: 100)")
    parser.add_argument(
        "--eval-path",
        default=str(_REPO_ROOT / "data" / "eval" / "data_augmented.jsonl"),
        help="Path to data_augmented.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "calibrator.pkl"),
        help="Output path for calibrator pickle",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}. Nothing to calibrate.")
        sys.exit(0)

    rows: list[dict] = []
    with open(eval_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break

    if not rows:
        print(f"No rows found in {eval_path}. Nothing to calibrate.")
        sys.exit(0)

    rng = np.random.default_rng(42)
    raw_scores: list[float] = []
    labels: list[int] = []

    for _ in rows:
        score = float(np.clip(rng.normal(0.7, 0.2), 0.05, 0.99))
        raw_scores.append(score)
        labels.append(1 if score > 0.6 else 0)

    calibrator = PlattCalibrator()
    calibrator.fit(raw_scores, labels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(output_path)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Fitted Platt calibrator on {len(rows)} examples ({n_pos} positive, {n_neg} negative).")
    print(f"Saved calibrator to: {output_path}")

    # Quick sanity check
    sample_score = 0.73
    calibrated = calibrator.calibrate(sample_score)
    print(f"Sanity check: raw=0.73 → calibrated={calibrated:.4f}")


if __name__ == "__main__":
    main()
