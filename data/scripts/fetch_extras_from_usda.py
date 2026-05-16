"""
Backfill macros from USDA FoodData Central for rows in extra_foods.csv that
still have LLM-estimated values (i.e. weren't filled by the FatSecret step).

USDA's foundation/SR-legacy/survey datasets report nutrients per 100 g
natively, so we don't hit the "no_100g_serving" problem we had with FatSecret.

Strategy per row:
    1. If the row has already been updated by FatSecret (heuristic: float
       precision in the macros), skip it. Otherwise this is an LLM estimate
       and a candidate for USDA backfill.
    2. Search USDA `/v1/foods/search` with the food_name (and a fallback
       query without ", chicken"/", beef" qualifier).
    3. Prefer hits with dataType in (Foundation, SR Legacy, Survey (FNDDS)) —
       these are generic items, not branded products.
    4. Score by token overlap; require >= 0.4 to accept.
    5. Fetch `/v1/food/<fdcId>` and extract nutrient #1008 (Energy), 1003
       (Protein), 1005 (Carbohydrate), 1004 (Total lipid/fat), 2000 (Total
       sugars). USDA returns these per 100 g for the relevant dataTypes.
    6. Overwrite Calories_100g / protein_100g / carbs_100g / fat_100g / sugar_100g.

Usage:
    python data/scripts/fetch_extras_from_usda.py \\
        --in  data/processed/extra_foods.csv \\
        --out data/processed/extra_foods.csv \\
        --only-llm-estimates

The --only-llm-estimates flag (default ON) preserves any FatSecret values that
are already in the CSV — the LLM estimates use round numbers (e.g. 16.0, 30.0)
whereas FatSecret returns 2-decimal values (e.g. 16.20, 30.49). The script
treats any row where all macros are round numbers as still an LLM estimate.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_FOOD_URL = "https://api.nal.usda.gov/fdc/v1/food"

# USDA API requires dataType as a *repeated* query parameter, NOT a
# comma-separated string. Build it via a list-of-tuples.

# Prefer authoritative generic datasets (no brands)
PREFERRED_DATA_TYPES = ("Foundation", "SR Legacy", "Survey (FNDDS)")

# USDA nutrient numbers — Foundation/SR Legacy use the modern 1003-1008 IDs;
# Survey (FNDDS) uses legacy three-digit numbers. We accept both.
NUTRIENT_NUMBERS = {
    "calories": ("1008", "208"),
    "protein": ("1003", "203"),
    "carbohydrate": ("1005", "205"),
    "fat": ("1004", "204"),
    "sugar": ("2000", "269"),
}

# Nutrient name patterns as a last-resort fallback (case-insensitive substring match)
NUTRIENT_NAMES = {
    "calories": ("energy",),
    "protein": ("protein",),
    "carbohydrate": ("carbohydrate, by difference", "carbohydrate"),
    "fat": ("total lipid (fat)", "total lipid", "total fat"),
    "sugar": ("total sugars", "sugars, total"),
}

STOPWORDS = frozenset({"the", "a", "an", "and", "or", "with", "of", "in", "on",
                       "australian", "fresh", "organic", "homemade", "premium"})


def _significant_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (name or "").lower())
            if len(t) > 2 and t not in STOPWORDS}


def _overlap(a: str, b: str) -> float:
    ta, tb = _significant_tokens(a), _significant_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _candidate_queries(food_name: str) -> list[str]:
    name = food_name.strip()
    if not name:
        return []
    primary = re.sub(r"\s*,\s*", " ", name).strip()
    queries = [primary]
    if "," in name:
        before_comma = name.split(",", 1)[0].strip()
        if before_comma and before_comma.lower() != primary.lower():
            queries.append(before_comma)
    return queries


def _is_llm_estimate(row: pd.Series) -> bool:
    """A row is still LLM-estimated if all of its macros are round numbers
    (one decimal place at most). FatSecret returns multi-decimal values."""
    for col in ("Calories_100g", "protein_100g", "carbs_100g", "fat_100g"):
        try:
            val = float(row[col])
        except (TypeError, ValueError):
            return False
        # Round numbers: integer, or one decimal of .0 or .5
        if abs(val - round(val, 1)) > 0.001:
            return False
        decimal_part = abs(val - int(val))
        if decimal_part not in (0.0, 0.5):
            return False
    return True


def _get_with_retry(url: str, params=None, max_attempts: int = 4) -> requests.Response:
    """USDA's edge often returns spurious 400s; retry a few times."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (400, 502, 503, 504):
                # Likely transient — back off and retry
                time.sleep(0.5 * (attempt + 1))
                last_exc = requests.HTTPError(f"{resp.status_code} on attempt {attempt + 1}")
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_exc or requests.RequestException("USDA request failed after retries")


def search_usda(api_key: str, query: str) -> list[dict]:
    # dataType must be repeated, not comma-joined
    params: list[tuple[str, str]] = [("query", query), ("pageSize", "20"), ("api_key", api_key)]
    for dt in PREFERRED_DATA_TYPES:
        params.append(("dataType", dt))
    resp = _get_with_retry(USDA_SEARCH_URL, params=params)
    return resp.json().get("foods", [])


def get_usda_food(api_key: str, fdc_id: int) -> dict | None:
    resp = _get_with_retry(f"{USDA_FOOD_URL}/{fdc_id}", params={"api_key": api_key})
    return resp.json()


def _extract_nutrient(food: dict, kind: str) -> float | None:
    """Read a per-100g nutrient amount from a USDA food payload.

    Tries (in order): modern nutrient ID -> legacy nutrient number -> name match.
    Handles both response shapes:
      - Search hit:   {"nutrientId": 1008, "nutrientNumber": "208",
                       "nutrientName": "Energy", "value": 220}
      - Detail food:  {"nutrient": {"id": 1008, "number": "208",
                       "name": "Energy"}, "amount": 220}
    """
    target_numbers = NUTRIENT_NUMBERS.get(kind, ())
    target_names = NUTRIENT_NAMES.get(kind, ())

    def _value_of(n: dict) -> float | None:
        for key in ("value", "amount"):
            if key in n and n[key] is not None:
                try:
                    return float(n[key])
                except (TypeError, ValueError):
                    pass
        nested = n.get("nutrient") or {}
        if "amount" in nested and nested["amount"] is not None:
            try:
                return float(nested["amount"])
            except (TypeError, ValueError):
                pass
        return None

    nutrients = food.get("foodNutrients", []) or []

    # 1) Match by nutrient number (covers both modern and legacy IDs)
    for n in nutrients:
        nested = n.get("nutrient") or {}
        num = str(n.get("nutrientNumber") or nested.get("number") or "")
        if num and num in target_numbers:
            v = _value_of(n)
            if v is not None:
                return v

    # 2) Match by nutrient name as a fallback
    for n in nutrients:
        nested = n.get("nutrient") or {}
        name = (n.get("nutrientName") or nested.get("name") or "").lower()
        if any(tn in name for tn in target_names):
            v = _value_of(n)
            if v is not None:
                return v

    return None


def fetch_for_row(api_key: str, food_name: str) -> tuple[dict | None, str]:
    best = None  # (score, fdcId, description, dataType, hit_payload)
    for query in _candidate_queries(food_name):
        try:
            hits = search_usda(api_key, query)
        except Exception as exc:
            return None, f"search_error: {exc}"

        for hit in hits:
            desc = str(hit.get("description") or "")
            data_type = str(hit.get("dataType") or "")
            fdc_id = hit.get("fdcId")
            if not fdc_id:
                continue
            score = _overlap(food_name, desc)
            # Boost generic datasets over branded
            dataset_bonus = 0.1 if data_type in PREFERRED_DATA_TYPES else 0.0
            ranked = score + dataset_bonus
            if score >= 0.4 and (best is None or ranked > best[0]):
                best = (ranked, fdc_id, desc, data_type, hit)

        if best and best[0] >= 0.6:
            break

    if not best:
        return None, "no_match"

    # First try to extract from the search-hit payload itself (one less API call).
    # Falls back to the detail endpoint only if the search hit lacks the data.
    food = best[4] if len(best) >= 5 else None
    cal = _extract_nutrient(food or {}, "calories")
    protein = _extract_nutrient(food or {}, "protein")
    carbs = _extract_nutrient(food or {}, "carbohydrate")
    fat = _extract_nutrient(food or {}, "fat")
    sugar = _extract_nutrient(food or {}, "sugar")

    if any(v is None for v in (cal, protein, carbs, fat)):
        try:
            detail = get_usda_food(api_key, best[1])
        except Exception as exc:
            return None, f"detail_error: {exc}"
        if not detail:
            return None, "empty_detail"
        cal = cal if cal is not None else _extract_nutrient(detail, "calories")
        protein = protein if protein is not None else _extract_nutrient(detail, "protein")
        carbs = carbs if carbs is not None else _extract_nutrient(detail, "carbohydrate")
        fat = fat if fat is not None else _extract_nutrient(detail, "fat")
        sugar = sugar if sugar is not None else _extract_nutrient(detail, "sugar")

    if cal is None or protein is None or carbs is None or fat is None:
        return None, f"incomplete_nutrients (matched '{best[2]}', dataType={best[3]})"

    return {
        "Calories_100g": round(cal, 2),
        "protein_100g": round(protein, 2),
        "carbs_100g": round(carbs, 2),
        "fat_100g": round(fat, 2),
        "sugar_100g": round(sugar if sugar is not None else 0.0, 2),
        "_matched": best[2],
        "_dataType": best[3],
        "_score": round(best[0], 2),
    }, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default="data/processed/extra_foods.csv")
    parser.add_argument("--out", default="data/processed/extra_foods.csv")
    parser.add_argument("--only-llm-estimates", action="store_true", default=True,
                        help="Only update rows that still have LLM round-number values (default).")
    parser.add_argument("--all", action="store_true",
                        help="Override --only-llm-estimates and re-fetch every row.")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    api_key = os.getenv("USDA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("USDA_API_KEY not set in .env")

    df = pd.read_csv(src)
    print(f"Loaded {len(df):,} rows from {src}")

    backup = src.with_suffix(".before_usda.csv")
    shutil.copy2(src, backup)
    print(f"Backup written: {backup}")

    # Decide which rows to process
    target_mask = pd.Series([True] * len(df)) if args.all else df.apply(_is_llm_estimate, axis=1)
    targets = df[target_mask]
    print(f"Targets: {len(targets):,} rows (LLM estimates)" if not args.all else f"Targets: {len(targets):,} rows (all)")
    print()

    updated = 0
    skipped = 0
    for idx, row in targets.iterrows():
        name = str(row["food_name"])
        result, status = fetch_for_row(api_key, name)
        if result:
            updated += 1
            old = (float(row["Calories_100g"]), float(row["protein_100g"]),
                   float(row["carbs_100g"]), float(row["fat_100g"]))
            for col in ("Calories_100g", "protein_100g", "carbs_100g", "fat_100g", "sugar_100g"):
                df.at[idx, col] = result[col]
            new = (result["Calories_100g"], result["protein_100g"],
                   result["carbs_100g"], result["fat_100g"])
            print(f"  OK  [{idx + 1:>2}/{len(df)}]  {name[:42]:<42}  "
                  f"matched='{result['_matched'][:45]}' [{result['_dataType']}]")
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
    print(f"  skipped  : {skipped:,}  (kept previous values)")
    print(f"  backup   : {backup}")
    print(f"  output   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
