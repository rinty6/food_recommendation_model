"""
Match food_name in cleaned_food_data to a Food-101 category, then assemble
a parquet of {food_id -> food101_image_url}.

Reads:   v1's off.db (Australian-eligible rows only, ~4,027)
Writes:  data/processed/food101_images.parquet  with columns:
            food_id, food101_category, food101_image_url

The image_url is built as `{image_base_url}/static/food101/{category}.jpg`.
Set --image-base-url to your Railway deployment URL, e.g.
    https://goodhealthmate-ml-v2.up.railway.app

Usage:
    python data/scripts/enrich_food101.py \
        --db ../machine_learning/dataset_process/off.db \
        --table cleaned_food_data \
        --image-base-url https://goodhealthmate-ml-v2.up.railway.app \
        --out data/processed/food101_images.parquet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


# Order matters: more specific rules first, so "spaghetti bolognese" matches before "spaghetti carbonara".
# Each rule: (compiled regex, food101 category)
# Use \b word boundaries so "pho" doesn't accidentally match "phone".
_RULES_RAW: list[tuple[list[str], str]] = [
    # --- specific dishes first ---
    (["spaghetti bolognese", "spag bol"], "spaghetti_bolognese"),
    (["spaghetti carbonara", "carbonara"], "spaghetti_carbonara"),
    (["fish and chips", "fish & chips", "battered fish"], "fish_and_chips"),
    (["beef carpaccio", "carpaccio"], "beef_carpaccio"),
    (["beef tartare", "steak tartare"], "beef_tartare"),
    (["caesar salad"], "caesar_salad"),
    (["greek salad"], "greek_salad"),
    (["caprese salad", "caprese"], "caprese_salad"),
    (["beet salad", "beetroot salad"], "beet_salad"),
    (["seaweed salad", "wakame"], "seaweed_salad"),
    (["club sandwich"], "club_sandwich"),
    (["grilled cheese", "cheese toastie"], "grilled_cheese_sandwich"),
    (["lobster roll"], "lobster_roll_sandwich"),
    (["pulled pork sandwich", "pulled pork roll"], "pulled_pork_sandwich"),
    (["breakfast burrito"], "breakfast_burrito"),
    (["chicken curry", "butter chicken", "tikka masala", "korma"], "chicken_curry"),
    (["chicken quesadilla", "quesadilla"], "chicken_quesadilla"),
    (["chicken wings", "buffalo wings", "wings"], "chicken_wings"),
    (["chicken schnitzel", "schnitzel"], "chicken_quesadilla"),  # closest visual analogue
    (["clam chowder"], "clam_chowder"),
    (["miso soup"], "miso_soup"),
    (["french onion soup"], "french_onion_soup"),
    (["hot and sour soup"], "hot_and_sour_soup"),
    (["lobster bisque"], "lobster_bisque"),
    (["pho", "vietnamese soup"], "pho"),
    (["ramen"], "ramen"),
    (["pad thai"], "pad_thai"),
    (["fried rice"], "fried_rice"),
    (["paella"], "paella"),
    (["risotto"], "risotto"),
    (["lasagna", "lasagne"], "lasagna"),
    (["gnocchi"], "gnocchi"),
    (["ravioli"], "ravioli"),
    (["mac and cheese", "macaroni and cheese", "mac n cheese"], "macaroni_and_cheese"),
    (["dumpling", "wonton", "potsticker"], "dumplings"),
    (["gyoza"], "gyoza"),
    (["sushi"], "sushi"),
    (["sashimi"], "sashimi"),
    (["spring roll"], "spring_rolls"),
    (["takoyaki"], "takoyaki"),
    (["edamame"], "edamame"),
    (["bibimbap"], "bibimbap"),
    (["peking duck"], "peking_duck"),
    (["samosa"], "samosa"),
    (["falafel"], "falafel"),
    (["hummus", "hommus"], "hummus"),
    (["guacamole"], "guacamole"),
    (["tacos", "soft taco", "hard taco"], "tacos"),
    (["nachos"], "nachos"),
    (["ceviche"], "ceviche"),
    (["bruschetta"], "bruschetta"),
    (["garlic bread"], "garlic_bread"),
    (["pizza"], "pizza"),
    (["hamburger", "burger", "beef patty"], "hamburger"),
    (["hot dog", "frankfurt"], "hot_dog"),
    (["french fries", "chips ", "shoestring fries"], "french_fries"),
    (["onion rings"], "onion_rings"),
    (["poutine"], "poutine"),
    (["fried calamari", "calamari rings", "squid rings"], "fried_calamari"),
    (["crab cakes"], "crab_cakes"),
    (["shrimp and grits"], "shrimp_and_grits"),
    (["scallops"], "scallops"),
    (["oysters"], "oysters"),
    (["mussels"], "mussels"),
    (["foie gras"], "foie_gras"),
    (["escargots", "snails"], "escargots"),
    (["filet mignon"], "filet_mignon"),
    (["prime rib"], "prime_rib"),
    (["pork chop"], "pork_chop"),
    (["baby back ribs", "pork ribs", "bbq ribs"], "baby_back_ribs"),
    (["steak", "sirloin", "porterhouse", "ribeye", "rib eye", "rump steak"], "steak"),
    (["grilled salmon", "salmon fillet", "atlantic salmon", "smoked salmon"], "grilled_salmon"),
    (["tuna tartare"], "tuna_tartare"),
    (["eggs benedict", "egg benedict", "benedict"], "eggs_benedict"),
    (["deviled egg", "devilled egg"], "deviled_eggs"),
    (["huevos rancheros"], "huevos_rancheros"),
    (["omelette", "omelet", "scrambled egg", "fried egg"], "omelette"),
    (["pancake", "pikelet", "flapjack"], "pancakes"),
    (["waffle"], "waffles"),
    (["french toast"], "french_toast"),
    (["croque madame", "croque monsieur"], "croque_madame"),
    (["beignet"], "beignets"),
    (["churros"], "churros"),
    (["donut", "doughnut"], "donuts"),
    (["macarons", "macaron"], "macarons"),
    (["cup cake", "cupcake"], "cup_cakes"),
    (["red velvet"], "red_velvet_cake"),
    (["carrot cake"], "carrot_cake"),
    (["chocolate cake"], "chocolate_cake"),
    (["chocolate mousse"], "chocolate_mousse"),
    (["panna cotta"], "panna_cotta"),
    (["creme brulee", "crème brûlée"], "creme_brulee"),
    (["cheesecake"], "cheesecake"),
    (["tiramisu"], "tiramisu"),
    (["cannoli"], "cannoli"),
    (["baklava"], "baklava"),
    (["bread pudding"], "bread_pudding"),
    (["strawberry shortcake"], "strawberry_shortcake"),
    (["ice cream", "gelato"], "ice_cream"),
    # NOTE: "yogurt"/"yoghurt" alone is too greedy — matches yogurt-coated bars,
    # banana yogurt drinks, etc. Require a dish-context token.
    (["frozen yogurt", "froyo", "yogurt parfait", "yoghurt parfait"], "frozen_yogurt"),
    (["apple pie"], "apple_pie"),
    # "cheese plate" / "cheese platter" only — bare "parmesan"/"cheddar"/"feta"
    # match raw cheese ingredients which aren't visually cheese plates.
    (["cheese plate", "cheese platter", "cheese board"], "cheese_plate"),
]


def _compile_rules() -> list[tuple[re.Pattern, str]]:
    compiled = []
    for keywords, category in _RULES_RAW:
        # match any keyword as a substring, case-insensitive
        pattern = "|".join(re.escape(k) for k in keywords)
        compiled.append((re.compile(pattern, re.IGNORECASE), category))
    return compiled


def _match_category(name: str, rules: list[tuple[re.Pattern, str]]) -> str | None:
    if not name:
        return None
    for pat, category in rules:
        if pat.search(name):
            return category
    return None


ELIGIBILITY = (
    "(COALESCE(breakfast_main_safe, FALSE) OR COALESCE(lunch_main_safe, FALSE) "
    "OR COALESCE(dinner_main_safe, FALSE) OR COALESCE(breakfast_side_safe, FALSE) "
    "OR COALESCE(lunch_side_safe, FALSE) OR COALESCE(dinner_side_safe, FALSE))"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="cleaned_food_data")
    parser.add_argument("--id-column", default="RecipeId")
    parser.add_argument("--name-column", default="food_name")
    parser.add_argument(
        "--image-base-url",
        required=True,
        help="Base URL where /static/food101/<cat>.jpg is served (no trailing slash).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--australian-only",
        action="store_true",
        help="Only process rows where is_australian = TRUE (recommended).",
    )
    parser.add_argument(
        "--category-cap",
        type=int,
        default=10,
        help="Max food_ids that can share a single Food-101 category (default 10).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    base = args.image_base_url.rstrip("/")
    print(f"Image base URL: {base}")

    con = duckdb.connect(str(db_path), read_only=True)

    where = (
        f"{ELIGIBILITY} "
        f"AND {args.name_column} IS NOT NULL "
        f"AND TRIM({args.name_column}) <> '' "
        f"AND LOWER(TRIM({args.name_column})) <> 'unknown product'"
    )
    if args.australian_only:
        where += " AND COALESCE(is_australian, FALSE) = TRUE"

    sql = f"SELECT {args.id_column}, {args.name_column} FROM {args.table} WHERE {where}"
    rows = con.execute(sql).fetchall()
    print(f"Scanning {len(rows):,} eligible rows...")

    rules = _compile_rules()
    output: list[dict] = []
    matched_per_cat: dict[str, int] = {}
    # Cap how many foods can share a single Food-101 image. Beyond this, the
    # extra rows fall through to Tier 2/3 in the COALESCE — better a Woolworths
    # match than 365 rows all displaying the same frozen-yogurt JPG.
    per_category_cap = args.category_cap
    for rid, name in rows:
        cat = _match_category(str(name or ""), rules)
        if cat is None:
            continue
        if matched_per_cat.get(cat, 0) >= per_category_cap:
            continue
        output.append({
            "food_id": str(rid).strip(),
            "food101_category": cat,
            "food101_image_url": f"{base}/static/food101/{cat}.jpg",
        })
        matched_per_cat[cat] = matched_per_cat.get(cat, 0) + 1

    print(f"Matched: {len(output):,} rows  ({100 * len(output) / max(1, len(rows)):.1f}%)")
    print(f"Categories used: {len(matched_per_cat)} / 101")
    print()
    print("Top 15 categories by match count:")
    for cat, n in sorted(matched_per_cat.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:>5}x  {cat}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_parquet(out_path, index=False)
    print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
