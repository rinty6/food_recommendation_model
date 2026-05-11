"""
Test-set evaluator — runs against the held-out test split.

WARNING: Only run this at release time (Phase 5 cut-over).
Running it mid-development inflates results because you may tune toward the test set.

Same metrics as evaluate_dev.py but uses test_manifest.json and writes to
results/test_results/.

Usage:
    python evaluate/evaluate_test.py [--trials 50] [--top-k 10]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommendation_engine.local_dataset import LocalFoodDataset, build_query_vector  # noqa: E402

SLOTS = [
    ("breakfast", "main"),
    ("breakfast", "side"),
    ("lunch",     "main"),
    ("lunch",     "side"),
    ("dinner",    "main"),
    ("dinner",    "side"),
]

SLOT_CALORIE_TARGETS = {
    "breakfast": 450,
    "lunch":     650,
    "dinner":    700,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(splits_dir: Path, name: str) -> set[str]:
    path = splits_dir / f"{name}_manifest.json"
    with open(path) as fh:
        ids = json.load(fh)
    return set(str(x) for x in ids)


def _sample_text_queries(ds: LocalFoodDataset, test_ids: set[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    id_list = rng.sample(sorted(test_ids), min(n * 10, len(test_ids)))
    placeholders = ", ".join("?" for _ in id_list)
    rows = ds.conn.execute(
        f"SELECT food_name FROM {ds.db_table} "
        f"WHERE RecipeId::VARCHAR IN ({placeholders}) "
        f"AND is_australian = TRUE "
        f"AND food_name IS NOT NULL AND TRIM(food_name) <> '' "
        f"LIMIT {n}",
        id_list,
    ).fetchall()
    names = [r[0].strip() for r in rows if r[0] and r[0].strip()]
    return names[:n] if names else ["chicken", "salad", "oats", "eggs", "pasta"]


def _slot_safe_column(meal_type: str, role: str) -> str:
    return f"{meal_type}_{role}_safe"


def _run_trials(
    ds: LocalFoodDataset,
    meal_type: str,
    role: str,
    text_queries: list[str],
    trials: int,
    top_k: int,
    prefetch: int,
    rng: random.Random,
) -> dict:
    latencies_ms: list[float] = []
    total_items = 0
    australian_count = 0
    image_count = 0
    slot_correct_count = 0
    safe_col = _slot_safe_column(meal_type, role)
    slot_target = SLOT_CALORIE_TARGETS.get(meal_type, 600)

    for _ in range(trials):
        query_text = rng.choice(text_queries)
        qv = build_query_vector(slot_target, None)

        t0 = time.perf_counter()
        results = ds.search(
            meal_type=meal_type,
            query_vector=qv,
            top_k=top_k,
            prefetch=prefetch,
            is_australian_user=True,
            text_query=query_text,
            role_hint=role,
            log_search=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        for item in results:
            total_items += 1
            if item.get("is_australian"):
                australian_count += 1
            if item.get("image_url") and str(item["image_url"]).strip():
                image_count += 1
            if item.get(safe_col):
                slot_correct_count += 1

    latencies_arr = sorted(latencies_ms)
    n = len(latencies_arr)
    p50 = latencies_arr[int(n * 0.50)] if n else 0.0
    p95 = latencies_arr[int(n * 0.95)] if n else 0.0
    avg = sum(latencies_arr) / n if n else 0.0

    def pct(num, den):
        return round(100 * num / den, 1) if den else 0.0

    return {
        "trials": trials,
        "top_k": top_k,
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "avg": round(avg, 1)},
        "total_items": total_items,
        "australian_rate": pct(australian_count, total_items),
        "image_coverage": pct(image_count, total_items),
        "slot_correctness": pct(slot_correct_count, total_items),
    }


def _static_coverage(ds: LocalFoodDataset, test_ids: set[str]) -> dict:
    id_sample = sorted(test_ids)[:50000]
    placeholders = ", ".join("?" for _ in id_sample)
    row = ds.conn.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN COALESCE(is_australian, FALSE) THEN 1 ELSE 0 END) AS australian,
            SUM(CASE WHEN image_url IS NOT NULL AND image_url <> '' THEN 1 ELSE 0 END) AS with_image
        FROM {ds.db_table}
        WHERE RecipeId::VARCHAR IN ({placeholders})
        """,
        id_sample,
    ).fetchone()
    total, australian, with_image = row
    def pct(a, b):
        return round(100 * (a or 0) / b, 1) if b else 0.0
    return {
        "test_sample_size": len(id_sample),
        "australian_rate": pct(australian, total),
        "image_coverage": pct(with_image, total),
    }


def _format_table(slot_results: dict[str, dict]) -> str:
    header = f"{'slot':<22}  {'p50ms':>6}  {'p95ms':>6}  {'aus%':>6}  {'img%':>6}  {'slot%':>6}"
    sep = "-" * len(header)
    lines = [header, sep]
    for slot_key, r in slot_results.items():
        lat = r["latency_ms"]
        lines.append(
            f"{slot_key:<22}  {lat['p50']:>6.1f}  {lat['p95']:>6.1f}  "
            f"{r['australian_rate']:>6.1f}  {r['image_coverage']:>6.1f}  {r['slot_correctness']:>6.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    print("=" * 60)
    print("TEST SET EVALUATOR — results are final / release-quality.")
    print("Only run this when preparing a release. Use evaluate_dev.py")
    print("for development iteration.")
    print("=" * 60)
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefetch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "test_results"))
    parser.add_argument("--confirm", action="store_true", help="Required flag to prevent accidental runs.")
    args = parser.parse_args()

    if not args.confirm:
        print("Add --confirm to run the test-set evaluator.", file=sys.stderr)
        return 1

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test manifest…")
    test_ids = _load_manifest(splits_dir, "test")
    print(f"Test split: {len(test_ids):,} RecipeIds")

    print("Loading LocalFoodDataset (v2 DuckDB)…")
    ds = LocalFoodDataset()
    if not ds.is_ready:
        print("ERROR: dataset failed to load.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    text_queries = _sample_text_queries(ds, test_ids, n=200, seed=args.seed)
    print(f"Sampled {len(text_queries)} text queries from test split.")

    print("\nRunning static coverage check…")
    static = _static_coverage(ds, test_ids)
    print(f"  test sample : {static['test_sample_size']:,}")
    print(f"  Australian  : {static['australian_rate']}%")
    print(f"  with image  : {static['image_coverage']}%")

    print(f"\nRunning {args.trials} trials × {len(SLOTS)} slots…")
    slot_results: dict[str, dict] = {}
    for meal_type, role in SLOTS:
        key = f"{meal_type}/{role}"
        print(f"  {key}…", end=" ", flush=True)
        r = _run_trials(ds, meal_type, role, text_queries, args.trials, args.top_k, args.prefetch, rng)
        slot_results[key] = r
        print(f"p50={r['latency_ms']['p50']}ms  aus={r['australian_rate']}%  img={r['image_coverage']}%  slot={r['slot_correctness']}%")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_at": _utc_now_iso(),
        "split": "test",
        "db_path": str(ds.db_path),
        "config": {"trials": args.trials, "top_k": args.top_k, "prefetch": args.prefetch, "seed": args.seed},
        "static_coverage": static,
        "slots": slot_results,
    }

    json_path = out_dir / f"eval_{timestamp}.json"
    txt_path = out_dir / f"eval_{timestamp}.txt"

    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    table = _format_table(slot_results)
    txt_content = (
        f"GoodHealthMate v2 — TEST evaluation  {timestamp}\n"
        f"DB: {ds.db_path}\n"
        f"Trials/slot: {args.trials}  top_k: {args.top_k}  prefetch: {args.prefetch}\n\n"
        f"Static coverage (test sample {static['test_sample_size']:,} rows):\n"
        f"  Australian: {static['australian_rate']}%   Image: {static['image_coverage']}%\n\n"
        f"Per-slot search metrics (is_australian_user=True):\n"
        f"{table}\n\n"
        f"Columns: p50ms/p95ms = latency percentiles, aus% = Australian rate,\n"
        f"         img% = image coverage, slot% = slot eligibility correctness\n"
    )
    with open(txt_path, "w") as fh:
        fh.write(txt_content)

    print()
    print(f"\n{'='*60}")
    print(txt_content)
    print(f"JSON → {json_path}")
    print(f"TXT  → {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
