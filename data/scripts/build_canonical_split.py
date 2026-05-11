"""
Build the canonical 70/15/15 train/val/test split for the recommendation dataset.

Run once. The split never changes after that — re-running with the same seed produces
the same manifests, so the test set never leaks into iteration.

Usage:
    python data/scripts/build_canonical_split.py \
        --db data/processed/cleaned_food_data.duckdb \
        --table cleaned_food_data \
        --out data/splits/

Outputs:
    data/splits/train_manifest.json   list of food_ids in train
    data/splits/val_manifest.json     list of food_ids in val
    data/splits/test_manifest.json    list of food_ids in test
    data/splits/split_metadata.json   seed, date, group counts, content hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb


# Fixed seed — recorded in metadata, never change without bumping the split version.
SEED = 20260510

# Stratification: bucket every food into one of these groups based on RecipeCategory text.
MAJOR_GROUP_RULES = (
    ("breakfast", r"breakfast|cereal|oatmeal|porridge|pancake|waffle|granola|muesli"),
    ("beverage", r"beverag|drink|smoothie|juice|coffee|tea|cocktail|water|milk\b"),
    ("dessert", r"dessert|cake|cookie|pie|ice\s?cream|chocolate|brownie|muffin|pastry|candy|pudding"),
    ("salad", r"salad"),
    ("soup_stew", r"soup|stew|chowder|broth|chili|curry"),
    ("seafood", r"fish|seafood|shrimp|prawn|tuna|salmon|crab|lobster|sushi|sashimi"),
    ("poultry", r"chicken|turkey|duck|poultry"),
    ("red_meat", r"beef|pork|lamb|steak|burger|bacon|sausage|ham|veal"),
    ("pasta_grain", r"pasta|noodle|rice|risotto|grain|quinoa|couscous"),
    ("bread_baked", r"bread|pizza|sandwich|wrap|tortilla|biscuit|scone|roll|bagel|toast"),
    ("vegetable", r"vegetable|veggie|tofu|legume|bean|lentil|mushroom|potato"),
    ("dairy_egg", r"cheese|yogurt|yoghurt|egg|omelet"),
    ("snack", r"snack|appetizer|dip|spread|sauce"),
)


def major_group_for(category_text: str) -> str:
    """Bucket a recipe category into a major group. Anything unmatched goes to 'other'."""
    text = (category_text or "").strip().lower()
    if not text:
        return "other"
    for label, pattern in MAJOR_GROUP_RULES:
        if re.search(pattern, text):
            return label
    return "other"


def stratified_split(
    rows: list[tuple[str, str]],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[str], list[str], list[str], dict]:
    """
    rows: list of (food_id, major_group)
    Returns (train_ids, val_ids, test_ids, per_group_counts).
    """
    rng = random.Random(seed)
    by_group: dict[str, list[str]] = defaultdict(list)
    for food_id, group in rows:
        by_group[group].append(food_id)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    per_group_counts: dict[str, dict[str, int]] = {}

    for group, ids in sorted(by_group.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n = len(ids_sorted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Test gets the remainder so ratios add to exactly n.
        train_part = ids_sorted[:n_train]
        val_part = ids_sorted[n_train : n_train + n_val]
        test_part = ids_sorted[n_train + n_val :]

        train_ids.extend(train_part)
        val_ids.extend(val_part)
        test_ids.extend(test_part)
        per_group_counts[group] = {
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        }

    return train_ids, val_ids, test_ids, per_group_counts


def hash_id_list(ids: list[str]) -> str:
    h = hashlib.sha256()
    for fid in sorted(ids):
        h.update(fid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to the DuckDB file (e.g. data/processed/cleaned_food_data.duckdb)")
    parser.add_argument("--table", default="cleaned_food_data")
    parser.add_argument("--out", required=True, help="Output directory for the split manifests")
    parser.add_argument("--id-column", default="RecipeId")
    parser.add_argument("--category-column", default="RecipeCategory")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    print(f"Reading {args.id_column}, {args.category_column} from {db_path}::{args.table}")
    con = duckdb.connect(str(db_path), read_only=True)
    rows_raw = con.execute(
        f"SELECT {args.id_column}, COALESCE({args.category_column}, '') FROM {args.table}"
    ).fetchall()
    con.close()

    rows = [(str(rid), major_group_for(cat)) for rid, cat in rows_raw if str(rid).strip()]
    print(f"Loaded {len(rows)} rows.")

    train_ids, val_ids, test_ids, per_group_counts = stratified_split(rows, seed=SEED)

    metadata = {
        "version": "v1",
        "seed": SEED,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db_path),
        "source_table": args.table,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "totals": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "all": len(rows),
        },
        "by_group": per_group_counts,
        "hashes": {
            "train": hash_id_list(train_ids),
            "val": hash_id_list(val_ids),
            "test": hash_id_list(test_ids),
        },
    }

    (out_dir / "train_manifest.json").write_text(json.dumps(sorted(train_ids), indent=0))
    (out_dir / "val_manifest.json").write_text(json.dumps(sorted(val_ids), indent=0))
    (out_dir / "test_manifest.json").write_text(json.dumps(sorted(test_ids), indent=0))
    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print("=== Split summary ===")
    print(f"  total rows : {len(rows)}")
    print(f"  train      : {len(train_ids)}  ({len(train_ids)/len(rows):.1%})")
    print(f"  val        : {len(val_ids)}  ({len(val_ids)/len(rows):.1%})")
    print(f"  test       : {len(test_ids)}  ({len(test_ids)/len(rows):.1%})")
    print()
    print("=== Per major group ===")
    for group, counts in sorted(per_group_counts.items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {group:14s}  total={counts['total']:6d}  train={counts['train']:6d}  val={counts['val']:5d}  test={counts['test']:5d}")
    print()
    print(f"Wrote manifests to {out_dir}")


if __name__ == "__main__":
    main()
