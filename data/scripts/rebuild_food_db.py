"""
Rebuild the food DuckDB with image_url joined back in.

Reads:
    - source DuckDB (typically v1's machine_learning/dataset_process/off.db)
    - food_images.parquet (output of enrich_food_images.py)

Writes:
    - new DuckDB at --out, same schema as source plus an `image_url VARCHAR` column

Idempotent: re-running overwrites the destination.

Usage:
    python data/scripts/rebuild_food_db.py \\
        --src-db ../machine_learning/dataset_process/off.db \\
        --src-table cleaned_food_data \\
        --images data/processed/food_images.parquet \\
        --out data/processed/cleaned_food_data.duckdb \\
        --out-table cleaned_food_data
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-db", required=True)
    parser.add_argument("--src-table", default="cleaned_food_data")
    parser.add_argument("--images", required=True, help="Path to food_images.parquet")
    parser.add_argument("--out", required=True, help="Path to write the rebuilt DuckDB")
    parser.add_argument("--out-table", default="cleaned_food_data")
    parser.add_argument("--id-column", default="RecipeId")
    args = parser.parse_args()

    src_db = Path(args.src_db)
    images_path = Path(args.images)
    out_path = Path(args.out)

    if not src_db.exists():
        raise SystemExit(f"Source DB not found: {src_db}")
    if not images_path.exists():
        raise SystemExit(f"Images parquet not found: {images_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(f"Building {out_path} from {src_db}::{args.src_table} + {images_path}")

    # Use a fresh DuckDB so we can ATTACH the source read-only and copy with the join applied.
    out_con = duckdb.connect(str(out_path))
    out_con.execute(f"ATTACH '{src_db.as_posix()}' AS src (READ_ONLY)")
    out_con.execute(f"CREATE OR REPLACE TEMP VIEW v_images AS SELECT food_id::VARCHAR AS food_id, image_url FROM read_parquet('{images_path.as_posix()}')")

    out_con.execute(
        f"""
        CREATE TABLE {args.out_table} AS
        SELECT
            s.*,
            v.image_url AS image_url
        FROM src.{args.src_table} s
        LEFT JOIN v_images v
            ON CAST(s.{args.id_column} AS VARCHAR) = v.food_id
        """
    )

    total = out_con.execute(f"SELECT COUNT(*) FROM {args.out_table}").fetchone()[0]
    with_image = out_con.execute(
        f"SELECT COUNT(*) FROM {args.out_table} WHERE image_url IS NOT NULL AND image_url <> ''"
    ).fetchone()[0]
    out_con.close()

    print()
    print("=== Rebuild summary ===")
    print(f"  total rows      : {total}")
    print(f"  with image_url  : {with_image}  ({with_image/total:.1%} coverage)")
    print(f"  output          : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
