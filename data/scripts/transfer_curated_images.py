"""
transfer_curated_images.py — Transfer curated images from OFF food_ids to AUSNUT food_ids by name overlap.

Why this exists:
    When v2 switched from OFF (barcode-keyed) to AUSNUT (F-prefixed), all
    curated images became orphaned (different food_id namespaces).
    Many AUSNUT foods share names with foods the user already curated
    (e.g. user curated an "Apple, raw" OFF row -> the AUSNUT "Apple, raw,
    unpeeled, edible portion" row should inherit the same image).

What it does (non-destructive, additive):
    1. Reads curated_images.parquet — the existing OFF-keyed curations.
    2. Joins each curated row to its OFF food_name from off.db.
    3. For every AUSNUT food in the new DuckDB, finds the best OFF
       curation match by symmetric token overlap.
    4. Accepts the match when:
        - both names share at least 2 significant tokens
        - min(overlap(off->au), overlap(au->off)) >= --min-overlap
        - the AUSNUT id doesn't already have a curated entry
    5. Appends the new (AUSNUT_id -> image_url) rows to curated_images.parquet
       so they get picked up on the next rebuild_food_db.py run.

Original OFF-keyed entries are preserved on disk (you can switch back later).

Usage:
    python data/scripts/transfer_curated_images.py \\
        --off-db ..\\machine_learning\\dataset_process\\off.db \\
        --ausnut-db data\\processed\\cleaned_food_data_ausnut.duckdb \\
        --curated data\\processed\\curated_images.parquet \\
        --min-overlap 0.5

Pass --dry-run to preview matches without writing.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "with", "of", "in", "on", "for",
    "australian", "fresh", "organic", "homemade", "premium", "classic",
    "natural", "original", "imported", "regular", "plain", "commercial",
    "uncooked", "cooked", "raw", "edible", "portion", "not", "further",
    "defined", "added", "type", "kind",
})


def _sig_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (name or "").lower())
            if len(t) > 2 and t not in STOPWORDS}


def _bidir_overlap(a: str, b: str) -> tuple[float, set[str], set[str]]:
    ta, tb = _sig_tokens(a), _sig_tokens(b)
    if not ta or not tb:
        return 0.0, ta, tb
    common = ta & tb
    if not common:
        return 0.0, ta, tb
    return min(len(common) / len(ta), len(common) / len(tb)), ta, tb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-db", default="../machine_learning/dataset_process/off.db",
                        help="Path to v1's OFF DuckDB (source of food_name lookups for OFF curations)")
    parser.add_argument("--ausnut-db", default="data/processed/cleaned_food_data_ausnut.duckdb",
                        help="Path to the AUSNUT-derived DuckDB")
    parser.add_argument("--curated", default="data/processed/curated_images.parquet")
    parser.add_argument("--min-overlap", type=float, default=0.5,
                        help="Symmetric token-overlap threshold (default 0.5).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    off_db = Path(args.off_db)
    ausnut_db = Path(args.ausnut_db)
    curated_path = Path(args.curated)
    for p in (off_db, ausnut_db, curated_path):
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

    curated_df = pd.read_parquet(curated_path)
    curated_df["food_id"] = curated_df["food_id"].astype(str)
    curated_df = curated_df[curated_df["image_url"].astype(str) != ""]
    print(f"Loaded curated parquet: {len(curated_df):,} entries with image_url")

    # Already-AUSNUT entries (F-prefixed or X-prefixed) — skip from transfer
    already_ausnut_mask = curated_df["food_id"].str.match(r"^[FX]\d+$")
    off_curations = curated_df[~already_ausnut_mask].copy()
    existing_ausnut_ids = set(curated_df[already_ausnut_mask]["food_id"])
    print(f"  OFF-keyed (transfer source): {len(off_curations):,}")
    print(f"  Already AUSNUT-keyed (skipped): {len(existing_ausnut_ids):,}")
    print()

    # Look up OFF food_name for each OFF curation
    print("Looking up OFF food names...")
    off_conn = duckdb.connect(str(off_db), read_only=True)
    off_ids = off_curations["food_id"].tolist()
    placeholders = ",".join(["?"] * len(off_ids))
    rows = off_conn.execute(
        f"SELECT CAST(RecipeId AS VARCHAR), food_name "
        f"FROM cleaned_food_data WHERE CAST(RecipeId AS VARCHAR) IN ({placeholders})",
        off_ids,
    ).fetchall()
    off_conn.close()
    off_id_to_name = {str(rid): (name or "") for rid, name in rows}
    off_curations["off_food_name"] = off_curations["food_id"].map(off_id_to_name).fillna("")
    # Drop curations whose original OFF row has gone missing
    off_curations = off_curations[off_curations["off_food_name"] != ""]
    print(f"  resolved names for {len(off_curations):,} OFF curations")
    print()

    # Build search structure: (sig_tokens, food_id, food_name, image_url)
    off_index: list[tuple[set[str], str, str, str]] = []
    for _, r in off_curations.iterrows():
        tokens = _sig_tokens(r["off_food_name"])
        if len(tokens) >= 2:   # skip single-token vague names like "Milk"/"Apple"
            off_index.append((tokens, r["food_id"], r["off_food_name"], r["image_url"]))
    print(f"OFF curations eligible for transfer (>=2 significant tokens): {len(off_index):,}")

    # Load AUSNUT rows
    au_conn = duckdb.connect(str(ausnut_db), read_only=True)
    au_rows = au_conn.execute(
        "SELECT RecipeId, food_name FROM cleaned_food_data"
    ).fetchall()
    au_conn.close()
    print(f"AUSNUT rows to match against: {len(au_rows):,}")
    print()

    # Match each AUSNUT row to its best OFF curation
    matched: list[dict] = []
    skipped_existing = 0
    for au_id, au_name in au_rows:
        au_id = str(au_id)
        if au_id in existing_ausnut_ids:
            skipped_existing += 1
            continue
        au_tokens = _sig_tokens(au_name)
        if len(au_tokens) < 2:
            continue
        best = None  # (score, off_food_id, off_food_name, image_url)
        for off_tokens, off_id, off_name, image_url in off_index:
            common = au_tokens & off_tokens
            if not common:
                continue
            score = min(len(common) / len(au_tokens), len(common) / len(off_tokens))
            if score >= args.min_overlap and (best is None or score > best[0]):
                best = (score, off_id, off_name, image_url)
        if best:
            matched.append({
                "food_id": au_id,
                "image_url": best[3],
                "_score": round(best[0], 2),
                "_au_name": au_name,
                "_off_name": best[2],
                "_off_id": best[1],
            })

    print(f"Skipped (already curated as AUSNUT): {skipped_existing:,}")
    print(f"New AUSNUT curations transferred  : {len(matched):,}")
    print()

    # Print sample matches grouped by score
    if matched:
        print("Sample matches (random):")
        sample = pd.DataFrame(matched).sample(min(20, len(matched)), random_state=42)
        for _, m in sample.iterrows():
            print(f"  [score={m['_score']:.2f}]  AUSNUT '{m['_au_name'][:50]}'  <-  OFF '{m['_off_name'][:50]}'")
        print()

    if args.dry_run:
        print("--dry-run: no changes written")
        return 0

    # Backup and write
    backup = curated_path.with_suffix(".before_transfer.parquet")
    shutil.copy2(curated_path, backup)
    print(f"Backup: {backup}")

    new_rows = pd.DataFrame([
        {
            "food_id": m["food_id"],
            "image_url": m["image_url"],
            "curated_at": datetime.now(timezone.utc).isoformat(),
        }
        for m in matched
    ])
    combined = pd.concat([curated_df.drop(columns=[c for c in curated_df.columns if c not in {"food_id", "image_url", "curated_at"}]), new_rows], ignore_index=True)
    # Drop duplicates: prefer the later (AUSNUT-keyed) entry over the earlier (OFF) one
    combined = combined.drop_duplicates(subset=["food_id"], keep="last")
    combined.to_parquet(curated_path, index=False)
    print(f"Wrote {curated_path}: {len(combined):,} total entries  ({len(new_rows):,} new AUSNUT transfers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
