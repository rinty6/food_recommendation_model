"""
Rebuild the food DuckDB with image_url joined back in.

Coalesces image sources in priority order:
    1. food101_image_url   (curated dish photos, highest quality)
    2. woolworths_image_url (Australian retailer CDN, for branded products)
    3. image_url            (OpenFoodFacts / FatSecret enrichment, fallback)

Pass only the parquets you have; missing ones are skipped automatically.

Reads:
    - source DuckDB (typically v1's machine_learning/dataset_process/off.db)
    - --images path to food_images.parquet              (OFF/FatSecret, optional)
    - --food101 path to food101_images.parquet          (Food-101 curated, optional)
    - --woolworths path to woolworths_images.parquet    (Woolworths CDN, optional)

Writes:
    - new DuckDB at --out with an `image_url VARCHAR` column populated by COALESCE

Idempotent: re-running overwrites the destination.

Usage:
    python data/scripts/rebuild_food_db.py \\
        --src-db ../machine_learning/dataset_process/off.db \\
        --src-table cleaned_food_data \\
        --images data/processed/food_images.parquet \\
        --food101 data/processed/food101_images.parquet \\
        --out data/processed/cleaned_food_data.duckdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _create_view(con, name: str, parquet_path: Path | None, url_column: str) -> bool:
    """Register a view selecting (food_id, <url_column>) from a parquet, or return False if absent."""
    if not parquet_path or not parquet_path.exists():
        return False
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW {name} AS "
        f"SELECT food_id::VARCHAR AS food_id, {url_column} "
        f"FROM read_parquet('{parquet_path.as_posix()}')"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-db", required=True)
    parser.add_argument("--src-table", default="cleaned_food_data")
    parser.add_argument("--curated", help="curated_images.parquet (Tier 0 — manual review, highest priority)")
    parser.add_argument("--food101", help="food101_images.parquet (Tier 1)")
    parser.add_argument("--woolworths", help="woolworths_images.parquet (Tier 2)")
    parser.add_argument("--images", help="OFF/FatSecret food_images.parquet (Tier 3 — fallback)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-table", default="cleaned_food_data")
    parser.add_argument("--id-column", default="RecipeId")
    args = parser.parse_args()

    src_db = Path(args.src_db)
    out_path = Path(args.out)

    if not src_db.exists():
        raise SystemExit(f"Source DB not found: {src_db}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    curated_path = Path(args.curated) if args.curated else None
    food101_path = Path(args.food101) if args.food101 else None
    woolworths_path = Path(args.woolworths) if args.woolworths else None
    images_path = Path(args.images) if args.images else None

    print(f"Building {out_path}")
    print(f"  src       : {src_db}::{args.src_table}")
    print(f"  curated   : {curated_path or '<skip>'}")
    print(f"  food101   : {food101_path or '<skip>'}")
    print(f"  woolworths: {woolworths_path or '<skip>'}")
    print(f"  off       : {images_path or '<skip>'}")

    out_con = duckdb.connect(str(out_path))
    out_con.execute(f"ATTACH '{src_db.as_posix()}' AS src (READ_ONLY)")

    has_curated = _create_view(out_con, "v_curated", curated_path, "image_url AS curated_url")
    has_f101 = _create_view(out_con, "v_f101", food101_path, "food101_image_url AS f101_url")
    has_wool = _create_view(out_con, "v_wool", woolworths_path, "woolworths_image_url AS wool_url")
    has_off = _create_view(out_con, "v_off", images_path, "image_url AS off_url")

    # Build the JOIN + COALESCE expression dynamically. Priority: curated > food101 > woolworths > off.
    joins = []
    sources = []   # in priority order
    if has_curated:
        joins.append(f"LEFT JOIN v_curated ON CAST(s.{args.id_column} AS VARCHAR) = v_curated.food_id")
        sources.append("v_curated.curated_url")
    if has_f101:
        joins.append(f"LEFT JOIN v_f101 ON CAST(s.{args.id_column} AS VARCHAR) = v_f101.food_id")
        sources.append("v_f101.f101_url")
    if has_wool:
        joins.append(f"LEFT JOIN v_wool ON CAST(s.{args.id_column} AS VARCHAR) = v_wool.food_id")
        sources.append("v_wool.wool_url")
    if has_off:
        joins.append(f"LEFT JOIN v_off ON CAST(s.{args.id_column} AS VARCHAR) = v_off.food_id")
        sources.append("v_off.off_url")

    coalesce_expr = f"COALESCE({', '.join(sources)})" if sources else "NULL"

    create_sql = (
        f"CREATE TABLE {args.out_table} AS\n"
        f"SELECT s.*, {coalesce_expr} AS image_url\n"
        f"FROM src.{args.src_table} s\n"
        + "\n".join(joins)
    )
    out_con.execute(create_sql)

    total = out_con.execute(f"SELECT COUNT(*) FROM {args.out_table}").fetchone()[0]
    with_image = out_con.execute(
        f"SELECT COUNT(*) FROM {args.out_table} WHERE image_url IS NOT NULL AND image_url <> ''"
    ).fetchone()[0]

    # Per-source breakdown using the same COALESCE precedence.
    source_breakdown = []
    if has_curated:
        n = out_con.execute(
            f"SELECT COUNT(*) FROM {args.out_table} t "
            f"JOIN v_curated ON CAST(t.{args.id_column} AS VARCHAR) = v_curated.food_id"
        ).fetchone()[0]
        source_breakdown.append(("curated", n))
    if has_f101:
        n = out_con.execute(
            f"SELECT COUNT(*) FROM {args.out_table} t "
            f"JOIN v_f101 ON CAST(t.{args.id_column} AS VARCHAR) = v_f101.food_id"
        ).fetchone()[0]
        source_breakdown.append(("food101", n))
    if has_wool:
        n = out_con.execute(
            f"SELECT COUNT(*) FROM {args.out_table} t "
            f"JOIN v_wool ON CAST(t.{args.id_column} AS VARCHAR) = v_wool.food_id"
        ).fetchone()[0]
        source_breakdown.append(("woolworths", n))
    if has_off:
        n = out_con.execute(
            f"SELECT COUNT(*) FROM {args.out_table} t "
            f"JOIN v_off ON CAST(t.{args.id_column} AS VARCHAR) = v_off.food_id"
        ).fetchone()[0]
        source_breakdown.append(("off", n))
    out_con.close()

    print()
    print("=== Rebuild summary ===")
    print(f"  total rows      : {total:,}")
    print(f"  with image_url  : {with_image:,}  ({with_image/total:.1%} coverage)")
    for label, n in source_breakdown:
        print(f"  matches in {label:<11}: {n:,}")
    print(f"  output          : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
