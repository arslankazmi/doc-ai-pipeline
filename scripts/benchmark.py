"""
Benchmarks all 3 extractors on eval examples.
Run from repo root: python3 scripts/benchmark.py --n 5
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent  # scripts/benchmark.py → 2 levels up → repo root
sys.path.insert(0, str(_REPO_ROOT / "src"))


def normalize_persons(persons: list[dict]) -> list[dict]:
    return sorted(
        [
            {
                "barcode_number": p.get("barcode_number", "").strip(),
                "name": p.get("name", "").strip().lower(),
                "user_id": p.get("user_id", "").strip().upper(),
                "category": p.get("category", "").strip().upper(),
            }
            for p in persons
        ],
        key=lambda x: x["barcode_number"],
    )


def parse_gt_content(content: Any) -> dict:
    """Parse ground truth content — may be a str or a list of dicts."""
    if isinstance(content, list):
        text = content[0].get("text", "") if content else ""
    else:
        text = str(content)
    return json.loads(text)


def load_eval_rows(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def extract_gt(row: dict) -> list[dict]:
    """Return the ground-truth persons list from a conversation row."""
    messages = row.get("messages", [])
    # Last assistant message holds ground truth
    gt_message = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            gt_message = msg
            break
    if gt_message is None:
        return []
    gt_data = parse_gt_content(gt_message.get("content", "{}"))
    return gt_data.get("persons", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark extractors on eval data.")
    parser.add_argument("--n", type=int, default=5, help="Number of eval examples (default: 5)")
    parser.add_argument(
        "--models",
        default="qwen2vl,donut,paddle_trocr",
        help="Comma-separated list of models (default: qwen2vl,donut,paddle_trocr)",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "benchmark_results.md"),
        help="Output markdown file",
    )
    parser.add_argument(
        "--eval-path",
        default=str(_REPO_ROOT / "data" / "eval" / "data_eval.jsonl"),
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--images-dir",
        default=str(_REPO_ROOT / "data" / "images"),
        help="Base directory containing eval images",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}. Exiting.")
        sys.exit(0)

    rows = load_eval_rows(eval_path, args.n)
    if not rows:
        print("No eval rows loaded. Exiting.")
        sys.exit(0)

    print(f"Loaded {len(rows)} eval examples.")

    from pipeline.orchestrator import get_extractor
    from PIL import Image

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    all_model_stats: dict[str, dict] = {}
    images_dir = Path(args.images_dir)

    for model_name in model_names:
        print(f"\n--- Benchmarking model: {model_name} ---")
        try:
            extractor = get_extractor(model_name)
        except Exception as exc:
            print(f"  Failed to load extractor '{model_name}': {exc}")
            all_model_stats[model_name] = {"error": str(exc), "n": 0}
            continue

        records: list[dict] = []
        for idx, row in enumerate(rows):
            images: list[str] = row.get("images", [])
            if not images:
                print(f"  [{idx+1}/{len(rows)}] Skipping row with no images.")
                continue

            image_rel = images[0]
            image_full_path = images_dir / image_rel
            if not image_full_path.exists():
                print(f"  [{idx+1}/{len(rows)}] Image not found: {image_full_path}. Skipping.")
                continue

            gt_persons = extract_gt(row)

            try:
                image = Image.open(str(image_full_path))
                t0 = time.perf_counter()
                result = extractor.extract(image, str(image_full_path))
                latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                print(f"  [{idx+1}/{len(rows)}] Extraction error: {exc}")
                records.append({"exact_match": False, "latency_ms": 0.0, "routing": "error", "confidence": 0.0})
                continue

            # Convert PersonRecord dataclasses to dicts
            pred_persons_raw = [dataclasses.asdict(p) for p in result.persons]

            gt_norm = normalize_persons(gt_persons)
            pred_norm = normalize_persons(pred_persons_raw)
            exact_match = gt_norm == pred_norm

            confidence = result.persons[0].confidence if result.persons else 0.0
            routing = "easy" if confidence >= 0.85 else "hard"

            print(
                f"  [{idx+1}/{len(rows)}] exact_match={exact_match}  "
                f"confidence={confidence:.3f}  routing={routing}  "
                f"latency={latency_ms:.0f}ms"
            )
            records.append(
                {
                    "exact_match": exact_match,
                    "latency_ms": latency_ms,
                    "routing": routing,
                    "confidence": confidence,
                }
            )

        n = len(records)
        if n == 0:
            all_model_stats[model_name] = {"error": "no valid examples", "n": 0}
            continue

        accuracy = sum(r["exact_match"] for r in records) / n
        avg_latency = sum(r["latency_ms"] for r in records) / n
        avg_confidence = sum(r["confidence"] for r in records) / n
        easy_pct = sum(1 for r in records if r["routing"] == "easy") / n * 100

        all_model_stats[model_name] = {
            "n": n,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "easy_pct": easy_pct,
        }

    # Print table
    print("\n\n=== Benchmark Results ===")
    header = f"{'Model':<20} {'N':>5} {'Accuracy':>10} {'Avg Latency':>14} {'Avg Conf':>10} {'Easy%':>8}"
    print(header)
    print("-" * len(header))
    for model_name, stats in all_model_stats.items():
        if "error" in stats:
            print(f"{model_name:<20} {'ERROR':>5}  {stats['error']}")
        else:
            print(
                f"{model_name:<20} {stats['n']:>5} {stats['accuracy']:>9.1%} "
                f"{stats['avg_latency_ms']:>12.0f}ms {stats['avg_confidence']:>10.3f} "
                f"{stats['easy_pct']:>7.1f}%"
            )

    # Write markdown
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as fh:
        fh.write("# Model Benchmark Results\n\n")
        fh.write(f"Evaluated on {len(rows)} examples.\n\n")
        fh.write("| Model | N | Accuracy | Avg Latency | Avg Confidence | Easy% |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for model_name, stats in all_model_stats.items():
            if "error" in stats:
                fh.write(f"| {model_name} | ERROR | {stats['error']} | - | - | - |\n")
            else:
                fh.write(
                    f"| {model_name} | {stats['n']} | {stats['accuracy']:.1%} | "
                    f"{stats['avg_latency_ms']:.0f}ms | {stats['avg_confidence']:.3f} | "
                    f"{stats['easy_pct']:.1f}% |\n"
                )

    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
