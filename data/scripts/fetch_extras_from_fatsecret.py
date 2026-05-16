"""
Replace LLM-estimated macros in extra_foods.csv with authoritative values
from the FatSecret API.

For each row:
    1. Search FatSecret using food_name (strips ", chicken"/", beef" qualifier
       on a second attempt if the first search has no good match).
    2. Picks the best result by token-overlap of food_name vs the
       FatSecret hit's food name. Requires overlap >= 0.4 to accept.
    3. Fetches the full food detail and reads the per-100 g serving.
       If no 100 g serving exists, converts from any other serving via
       metric_serving_amount.
    4. Overwrites Calories_100g / protein_100g / carbs_100g / fat_100g / sugar_100g.

The original CSV is backed up to extra_foods.before_fatsecret.csv.

Usage:
    python data/scripts/fetch_extras_from_fatsecret.py \\
        --in  data/processed/extra_foods.csv \\
        --out data/processed/extra_foods.csv

FATSECRET_CLIENT_ID and FATSECRET_CLIENT_SECRET must be set in .env.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Reuse the engine's FatSecret client so OAuth + proxy URL stay in one place.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))
from recommendation_engine.fatsecret import FatSecretClient  # noqa: E402


# Tokens that contribute nothing to matching — drop before comparing.
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "with", "of", "in", "on",
    "australian", "fresh", "organic", "homemade", "premium",
})


def _significant_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (name or "").lower())
            if len(t) > 2 and t not in STOPWORDS}


def _overlap(a: str, b: str) -> float:
    """Fraction of a's significant tokens present in b."""
    ta, tb = _significant_tokens(a), _significant_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _per_100g_from_serving(serving: dict) -> dict | None:
    """Read calories/protein/carbs/fat/sugar normalised to 100 g."""
    metric_amount = float(serving.get("metric_serving_amount") or 0)
    metric_unit = str(serving.get("metric_serving_unit") or "").lower()
    if metric_unit not in {"g", "grams"} or metric_amount <= 0:
        return None
    scale = 100.0 / metric_amount

    def f(k):
        try:
            return round(float(serving.get(k) or 0) * scale, 2)
        except (TypeError, ValueError):
            return None

    calories = f("calories")
    protein = f("protein")
    carbs = f("carbohydrate")
    fat = f("fat")
    sugar = f("sugar")
    if calories is None or protein is None or carbs is None or fat is None:
        return None
    return {
        "Calories_100g": calories,
        "protein_100g": protein,
        "carbs_100g": carbs,
        "fat_100g": fat,
        "sugar_100g": sugar if sugar is not None else 0.0,
    }


def _extract_per_100g(food_detail: dict) -> dict | None:
    """From a food.get.v5 payload, extract the best per-100g macros."""
    if not food_detail:
        return None
    food = food_detail.get("food") if "food" in food_detail else food_detail
    if not isinstance(food, dict):
        return None
    servings = food.get("servings", {})
    serving_list = servings.get("serving") if isinstance(servings, dict) else None
    if not serving_list:
        return None
    if isinstance(serving_list, dict):
        serving_list = [serving_list]

    # Prefer the explicit "100 g" serving when available
    for s in serving_list:
        if not isinstance(s, dict):
            continue
        desc = str(s.get("serving_description") or "").lower()
        if "100 g" in desc or "100g" in desc:
            result = _per_100g_from_serving(s)
            if result:
                return result

    # Otherwise scale any gram-based serving to 100 g
    for s in serving_list:
        if isinstance(s, dict):
            result = _per_100g_from_serving(s)
            if result:
                return result
    return None


def _candidate_queries(food_name: str) -> list[str]:
    """Yield search queries in order of specificity, e.g.
        "Banh mi, chicken" -> ["banh mi chicken", "banh mi"]
    """
    name = food_name.strip()
    if not name:
        return []
    # Replace commas with spaces for FatSecret search
    primary = re.sub(r"\s*,\s*", " ", name).strip()
    queries = [primary]
    # Also try the part before the first comma
    if "," in name:
        before_comma = name.split(",", 1)[0].strip()
        if before_comma and before_comma.lower() != primary.lower():
            queries.append(before_comma)
    return queries


def fetch_for_row(client: FatSecretClient, food_name: str) -> tuple[dict | None, str]:
    """Return (macros_dict_or_None, status_message)."""
    best = None  # (overlap_score, food_id, food_name)
    for query in _candidate_queries(food_name):
        try:
            hits = client.search_foods(query, max_results=10, category="recipe")
        except Exception as exc:
            return None, f"search_error: {exc}"

        for hit in hits or []:
            hit_name = str(hit.get("food_name") or "")
            score = _overlap(food_name, hit_name)
            food_id = str(hit.get("food_id") or "").strip()
            if score >= 0.4 and food_id and (best is None or score > best[0]):
                best = (score, food_id, hit_name)

        if best and best[0] >= 0.6:
            break  # great match on first query, no need to try fallback

    if not best:
        return None, "no_match"

    try:
        detail = client.get_food(best[1])
    except Exception as exc:
        return None, f"detail_error: {exc}"

    macros = _extract_per_100g(detail)
    if not macros:
        return None, f"no_100g_serving (matched '{best[2]}')"

    macros["_matched"] = best[2]
    macros["_score"] = round(best[0], 2)
    return macros, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default="data/processed/extra_foods.csv")
    parser.add_argument("--out", default="data/processed/extra_foods.csv")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between requests (Premier Free has unlimited; small delay is courteous)")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    client_id = os.getenv("FATSECRET_CLIENT_ID", "").strip()
    client_secret = os.getenv("FATSECRET_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit("FATSECRET_CLIENT_ID / FATSECRET_CLIENT_SECRET not set in .env")

    client = FatSecretClient(client_id, client_secret)
    df = pd.read_csv(src)
    print(f"Loaded {len(df):,} rows from {src}")

    # Backup the original
    backup = src.with_suffix(".before_fatsecret.csv")
    shutil.copy2(src, backup)
    print(f"Backup written: {backup}")
    print()

    updated = 0
    skipped = 0
    for idx, row in df.iterrows():
        name = str(row["food_name"])
        result, status = fetch_for_row(client, name)
        if result:
            updated += 1
            old = (float(row["Calories_100g"]), float(row["protein_100g"]),
                   float(row["carbs_100g"]), float(row["fat_100g"]))
            new = (result["Calories_100g"], result["protein_100g"],
                   result["carbs_100g"], result["fat_100g"])
            for col in ("Calories_100g", "protein_100g", "carbs_100g", "fat_100g", "sugar_100g"):
                df.at[idx, col] = result[col]
            print(f"  OK  [{idx + 1:>2}/{len(df)}]  {name[:42]:<42}  "
                  f"matched='{result['_matched'][:35]}' score={result['_score']}")
            print(f"        old: kcal={old[0]:.0f} p={old[1]:.1f} c={old[2]:.1f} f={old[3]:.1f}   "
                  f"new: kcal={new[0]:.0f} p={new[1]:.1f} c={new[2]:.1f} f={new[3]:.1f}")
        else:
            skipped += 1
            print(f"  --  [{idx + 1:>2}/{len(df)}]  {name[:42]:<42}  {status}")
        time.sleep(args.delay)

    out = Path(args.out)
    df.to_csv(out, index=False)
    print()
    print("=== Summary ===")
    print(f"  updated  : {updated:,}")
    print(f"  skipped  : {skipped:,}  (kept LLM estimates)")
    print(f"  backup   : {backup}")
    print(f"  output   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
