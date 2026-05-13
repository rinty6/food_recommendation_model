"""
Re-validate the existing woolworths_checkpoint.csv WITHOUT calling the API.

For each row, joins the food_name back from the source DuckDB and runs the same
validators used by enrich_woolworths.py:

    1. Non-food keyword blocklist (cookbooks, jiggers, air fresheners, …)
    2. Token overlap between food_name and product_name (default ≥ 0.5)
    3. Per-stockcode reuse cap (default 3)

Bad matches are **removed from the checkpoint** so a future enrichment run will
retry them. The cleaned parquet only contains validated matches.

Usage:
    python data/scripts/clean_woolworths_matches.py \
        --db ../machine_learning/dataset_process/off.db \
        --checkpoint data/processed/woolworths_checkpoint.csv \
        --out data/processed/woolworths_images.parquet \
        --backup data/processed/woolworths_checkpoint.before_clean.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd

# Import validators from the enrichment script so the rules stay in one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_woolworths import is_non_food_product, name_overlap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="cleaned_food_data")
    parser.add_argument("--id-column", default="RecipeId")
    parser.add_argument("--name-column", default="food_name")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backup", help="Path to copy the original checkpoint before rewriting")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--reuse-cap", type=int, default=3)
    args = parser.parse_args()

    chk_path = Path(args.checkpoint)
    if not chk_path.exists():
        print(f"Checkpoint not found: {chk_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(chk_path)
    df["food_id"] = df["food_id"].astype(str).str.strip()
    df["woolworths_image_url"] = df["woolworths_image_url"].fillna("")
    df["product_name"] = df["product_name"].fillna("").astype(str)
    print(f"Checkpoint rows: {len(df):,}")

    # Pull food_name for every food_id in the checkpoint
    con = duckdb.connect(str(args.db), read_only=True)
    food_ids = df["food_id"].tolist()
    placeholders = ",".join(["?"] * len(food_ids))
    rows = con.execute(
        f"SELECT CAST({args.id_column} AS VARCHAR), {args.name_column} "
        f"FROM {args.table} WHERE CAST({args.id_column} AS VARCHAR) IN ({placeholders})",
        food_ids,
    ).fetchall()
    con.close()
    id_to_name = {str(rid): (name or "") for rid, name in rows}
    df["food_name"] = df["food_id"].map(id_to_name).fillna("")

    # Validate every row that currently has an image_url
    has_url = df["woolworths_image_url"] != ""
    print(f"Currently with image_url: {has_url.sum():,}")
    print()

    rejections = {"non_food": 0, "low_overlap": 0, "reuse_cap": 0}

    def _reject(idx, reason):
        rejections[reason] += 1
        df.at[idx, "woolworths_image_url"] = ""
        df.at[idx, "product_name"] = f"rejected_{reason}"

    for idx in df[has_url].index:
        food_name = df.at[idx, "food_name"]
        product_name = df.at[idx, "product_name"]
        if is_non_food_product(product_name):
            _reject(idx, "non_food")
            continue
        if name_overlap(food_name, product_name) < args.min_overlap:
            _reject(idx, "low_overlap")
            continue

    # Reuse cap: per stockcode, keep at most N rows.
    ok = df[df["woolworths_image_url"] != ""].copy()
    ok["stockcode"] = ok["woolworths_image_url"].str.extract(r"/(\d+)\.jpg$")
    counts = ok.groupby("stockcode").size()
    overused = counts[counts > args.reuse_cap].index.tolist()
    for sc in overused:
        idxs = ok[ok["stockcode"] == sc].index.tolist()
        for drop_idx in idxs[args.reuse_cap:]:
            _reject(drop_idx, "reuse_cap")

    print("Rejections:")
    for k, v in rejections.items():
        print(f"  {k:<12}: {v:,}")
    print()

    # Drop the rejected rows from the checkpoint entirely so they get retried
    kept = df[df["woolworths_image_url"] != ""].copy()
    rejected = df[df["woolworths_image_url"] == ""].copy()
    print(f"Kept     : {len(kept):,}")
    print(f"Rejected : {len(rejected):,}  (removed from checkpoint, will retry on next API run)")

    # Backup + rewrite checkpoint (only kept rows remain — rejected ones get a fresh shot)
    if args.backup:
        backup_path = Path(args.backup)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chk_path, backup_path)
        print(f"Backup   : {backup_path}")
    kept_cols = ["food_id", "woolworths_image_url", "product_name"]
    kept[kept_cols].to_csv(chk_path, index=False)
    print(f"Checkpoint rewritten: {chk_path}  ({len(kept):,} rows)")

    # Write the cleaned parquet
    parquet_path = Path(args.out)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    kept[["food_id", "woolworths_image_url"]].to_parquet(parquet_path, index=False)
    print(f"Parquet  : {parquet_path}  ({len(kept):,} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
