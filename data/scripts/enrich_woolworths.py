"""
Enrich Australian-eligible food rows with Woolworths product images (Tier 2).

For each row not already covered by Food-101, searches the Woolworths Products
API on RapidAPI, extracts the product stockcode, and builds a CDN image URL.

Image URL pattern: https://cdn0.woolworths.media/content/wowproductimages/large/<stockcode>.jpg

Usage:
    python data/scripts/enrich_woolworths.py \
        --db ../machine_learning/dataset_process/off.db \
        --food101 data/processed/food101_images.parquet \
        --out data/processed/woolworths_images.parquet \
        --checkpoint data/processed/woolworths_checkpoint.csv \
        --limit 490

Set RAPIDAPI_KEY in your .env file (or pass --api-key).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_HOST = "woolworths-products-api.p.rapidapi.com"
SEARCH_ENDPOINT = "https://woolworths-products-api.p.rapidapi.com/woolworths/product-search/"
CDN_BASE = "https://cdn0.woolworths.media/content/wowproductimages/large"
STOCKCODE_RE = re.compile(r"/productdetails/(\d+)", re.IGNORECASE)

ELIGIBILITY = (
    "(COALESCE(breakfast_main_safe, FALSE) OR COALESCE(lunch_main_safe, FALSE) "
    "OR COALESCE(dinner_main_safe, FALSE) OR COALESCE(breakfast_side_safe, FALSE) "
    "OR COALESCE(lunch_side_safe, FALSE) OR COALESCE(dinner_side_safe, FALSE))"
)

# Substrings that mean a Woolworths product is not actually food (books,
# accessories, homeware, gift cards…). If any appear, reject the match.
NON_FOOD_KEYWORDS = (
    "cookbook", " book", "magazine", "journal", "novel",
    "jigger", "shaker", "bottle opener", "wine glass", "tumbler", "stemware",
    "air freshener", "freshener", "candle", "incense",
    "soap", "shampoo", "deodorant", "moisturiser", "moisturizer", "lotion",
    "wallpaper", "ornament", "decoration", "balloon",
    "gift card", "voucher", "toy", "puzzle",
    "battery", "charger", "cable", "phone case",
    "frying pan", "saucepan", "knife set", "cutlery set",
    "pet food", "dog food", "cat food", "bird seed",
)

# Tokens that contribute nothing to "is this the same product" — drop them
# before scoring overlap. Many of our food_names are noisy ("Australian Roasted
# Peanuts Unsalted Crunchy"), so we ignore the framing words.
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "with", "in", "on", "for", "to",
    "australian", "australia", "imported", "fresh", "organic", "natural",
    "premium", "classic", "original", "traditional", "homemade", "frozen",
    "low", "high", "free", "reduced", "unsweetened", "sweetened",
    "g", "kg", "ml", "l", "pk", "ea",
})


def _significant_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 2 and t not in STOPWORDS}


def is_non_food_product(product_name: str) -> bool:
    lower = " " + product_name.lower() + " "
    return any(kw in lower for kw in NON_FOOD_KEYWORDS)


def name_overlap(food_name: str, product_name: str) -> float:
    """Fraction of significant food_name tokens that appear in product_name.

    1.0 means every meaningful word from the food appears in the product.
    Returns 0.0 if the food name has no significant tokens (avoids div-by-zero).
    """
    food_tokens = _significant_tokens(food_name)
    product_tokens = _significant_tokens(product_name)
    if not food_tokens:
        return 0.0
    return len(food_tokens & product_tokens) / len(food_tokens)


def best_acceptable_match(
    food_name: str, results: list[dict], min_overlap: float = 0.5
) -> tuple[dict | None, float, str]:
    """Walk Woolworths search results; return the first product that passes
    blocklist + overlap checks. Returns (result, overlap_score, reject_reason)."""
    last_reason = "no_results"
    for r in results:
        pname = str(r.get("product_name") or "")
        if not pname:
            continue
        if is_non_food_product(pname):
            last_reason = "non_food"
            continue
        overlap = name_overlap(food_name, pname)
        if overlap < min_overlap:
            last_reason = f"low_overlap_{overlap:.2f}"
            continue
        return r, overlap, "ok"
    return None, 0.0, last_reason


def _search_woolworths(query: str, api_key: str, page_size: int = 5) -> list[dict]:
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": api_key,
        "Content-Type": "application/json",
    }
    params = {"query": query, "page": 1, "page_size": page_size}
    resp = requests.get(SEARCH_ENDPOINT, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def _extract_stockcode(product_url: str) -> str | None:
    m = STOCKCODE_RE.search(product_url)
    return m.group(1) if m else None


def _build_image_url(stockcode: str) -> str:
    return f"{CDN_BASE}/{stockcode}.jpg"


def _load_checkpoint(path: Path) -> set[str]:
    """Return set of food_ids already processed (success or skip)."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["food_id"] for row in csv.DictReader(f)}


def _append_checkpoint(path: Path, food_id: str, image_url: str, product_name: str) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["food_id", "woolworths_image_url", "product_name"])
        if is_new:
            writer.writeheader()
        writer.writerow({"food_id": food_id, "woolworths_image_url": image_url, "product_name": product_name})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Source DuckDB path (off.db)")
    parser.add_argument("--table", default="cleaned_food_data")
    parser.add_argument("--id-column", default="RecipeId")
    parser.add_argument("--name-column", default="food_name")
    parser.add_argument("--food101", help="food101_images.parquet — skip already-covered rows")
    parser.add_argument("--out", required=True, help="Output woolworths_images.parquet")
    parser.add_argument("--checkpoint", required=True, help="CSV file for incremental progress")
    parser.add_argument("--limit", type=int, default=490, help="Max API calls per run (default 490)")
    parser.add_argument("--api-key", default="", help="RapidAPI key (falls back to RAPIDAPI_KEY env var)")
    parser.add_argument("--australian-only", action="store_true", help="Only process Australian-eligible rows")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--reuse-cap", type=int, default=3, help="Max distinct food_ids per Woolworths stockcode")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("RAPIDAPI_KEY", "")
    if not api_key:
        print("ERROR: Set RAPIDAPI_KEY in .env or pass --api-key", file=sys.stderr)
        return 1

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    # Load Australian-eligible rows from source DB
    con = duckdb.connect(str(db_path), read_only=True)
    where = (
        f"{ELIGIBILITY} "
        f"AND {args.name_column} IS NOT NULL "
        f"AND TRIM({args.name_column}) <> '' "
        f"AND LOWER(TRIM({args.name_column})) <> 'unknown product'"
    )
    if args.australian_only:
        where += " AND COALESCE(is_australian, FALSE) = TRUE"
    rows = con.execute(
        f"SELECT {args.id_column}, {args.name_column} FROM {args.table} WHERE {where}"
    ).fetchall()
    con.close()
    print(f"Eligible rows: {len(rows):,}")

    # Build exclusion set: food_ids already covered by Food-101
    food101_ids: set[str] = set()
    if args.food101 and Path(args.food101).exists():
        f101 = pd.read_parquet(args.food101)
        food101_ids = set(f101["food_id"].astype(str).str.strip())
        print(f"Food-101 already covers: {len(food101_ids):,} rows (will skip)")

    # Load checkpoint (rows already attempted in previous runs)
    checkpoint_path = Path(args.checkpoint)
    done_ids = _load_checkpoint(checkpoint_path)
    print(f"Checkpoint: {len(done_ids):,} rows already processed")

    # Filter to rows we still need to process
    pending = [
        (str(rid).strip(), str(name or "").strip())
        for rid, name in rows
        if str(rid).strip() not in food101_ids and str(rid).strip() not in done_ids
    ]
    print(f"Pending (need Woolworths lookup): {len(pending):,}")
    print(f"API call budget this run: {args.limit}")
    print()

    matched = 0
    skipped = 0
    errors = 0

    for i, (food_id, name) in enumerate(pending[: args.limit]):
        if i > 0 and i % 50 == 0:
            pct = matched / max(1, i) * 100
            print(f"  [{i}/{min(len(pending), args.limit)}] matched so far: {matched} ({pct:.0f}%)")

        try:
            results = _search_woolworths(name, api_key)
            match, overlap, reason = best_acceptable_match(name, results)
            if match is not None:
                stockcode = _extract_stockcode(match.get("url", ""))
                if stockcode:
                    image_url = _build_image_url(stockcode)
                    _append_checkpoint(checkpoint_path, food_id, image_url, match.get("product_name", ""))
                    matched += 1
                else:
                    _append_checkpoint(checkpoint_path, food_id, "", f"no_stockcode|{match.get('product_name', '')}")
                    skipped += 1
            else:
                _append_checkpoint(checkpoint_path, food_id, "", f"rejected_{reason}")
                skipped += 1
        except Exception as exc:
            print(f"  ERROR [{food_id}] {name!r}: {exc}")
            errors += 1
            _append_checkpoint(checkpoint_path, food_id, "", f"ERROR: {exc}")

        time.sleep(args.delay)

    print()
    print("=== Run summary ===")
    print(f"  processed  : {min(len(pending), args.limit):,}")
    print(f"  matched    : {matched:,}  ({matched / max(1, min(len(pending), args.limit)):.0%} hit rate)")
    print(f"  no match   : {skipped:,}")
    print(f"  errors     : {errors:,}")
    print(f"  remaining  : {max(0, len(pending) - args.limit):,}")

    # Write parquet from all successful checkpoint entries, applying
    # a per-stockcode reuse cap so a single image can't end up on dozens
    # of unrelated foods.
    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path)
        ok = df[df["woolworths_image_url"].notna() & (df["woolworths_image_url"] != "")].copy()
        ok["stockcode"] = ok["woolworths_image_url"].str.extract(r"/(\d+)\.jpg$")
        before = len(ok)
        ok = ok.groupby("stockcode", group_keys=False).head(args.reuse_cap)
        capped = before - len(ok)
        df_out = ok[["food_id", "woolworths_image_url"]]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path, index=False)
        print(f"  parquet rows: {len(df_out):,}  (reuse cap dropped {capped})  → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
