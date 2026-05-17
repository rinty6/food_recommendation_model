"""
Build a clean v2 DuckDB from AUSNUT_dataset.csv (Australian National Nutrient
Database). Output is a drop-in replacement for the OFF-derived DuckDB.

Why AUSNUT: generic Australian foods only — no brand names, no duplicates by
construction. Solves the "Bega Stringers Original" recommendation problem.

Meal-slot safety is decided purely by RecipeCategory. AUSNUT's categories are
clean and structured (~700 distinct values), so a category allowlist is more
reliable than v1's text-keyword classifier (which was tuned for OFF product
titles).

Usage:
    python data/scripts/build_ausnut_db.py \\
        --src ../machine_learning/dataset_process/AUSNUT_dataset.csv \\
        --out data/processed/cleaned_food_data.duckdb
        [--extras data/processed/extra_foods.csv]   # optional Asian dishes etc.

The --extras CSV must have the same schema as AUSNUT_dataset.csv. Use it to
hand-add foods AUSNUT lacks (pho, banh mi, kimchi, …).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd


# v1's text classifier is tuned for OpenFoodFacts product titles, not AUSNUT's
# clean structured categories. For AUSNUT we classify by RecipeCategory directly —
# it's the canonical signal and there are only ~700 distinct values to label.

# Categories that should never appear as standalone foods.
# Snacks, desserts, ingredients, condiments, alcohol, supplements, baby food.
HARD_BLOCK_CATEGORIES = {
    # Snacks and sweet treats
    "Biscuit", "Cake", "Cake or bread", "Slice", "Pavlova", "Trifle",
    "Confectionery", "Confectionary", "Chocolate", "Chocolates",
    "Mixed nuts & chocolate", "Lolly", "Lollipop or chupa chup",
    "Marshmallow", "Meringue", "Nougat", "Fudge", "Caramels", "Halvah (Halva)",
    "Honeycomb", "Marzipan", "Turkish delight", "Bar", "Bar or strap",
    "Bar or biscuit", "Snack ball", "Brownie", "Cookie dough", "Cake batter",
    "Rockcake", "Mochi", "Tiramisu", "Custard pudding", "Custard",
    "Pudding", "Mousse", "Fromais frais (fromage frais)", "Ice cream",
    "Ice confection", "Gelato or sorbet", "Jelly", "Jelly crystals",
    "Dairy dessert", "Banana split", "Sundae", "Eclair or profiterole",
    "Snack pack", "Cone", "Croissant", "Doughnut (donut)", "Danish",
    "Strudel", "Baklava", "Cannoli", "Tarte tatin", "Crumble", "Panna cotta",
    "Tart", "Pastry", "Breakfast pastry", "Scone", "Muffin", "Slice",
    "Yorkshire pudding", "Fondue",

    # Savoury snacks / chips
    "Corn chips", "Chip or crisp", "Extruded snack", "Savoury snack",
    "Potato crisps or chips", "Tortilla chips", "Rice crisps or chips",
    "Seaweed crisps or chips", "Sweet potato crisps or chips",
    "Vegetable crisps or chips", "Coconut chips or crisps", "Banana chip",
    "Snack mix", "Wheat snack", "Noodle snack", "Pretzels", "Popcorn",
    "Puffed corn", "Pork rind/crackling", "Fish skin", "Prawn cracker",
    "Pappadam", "Taco shell",

    # Cooking ingredients & seasonings
    "Oil", "Fat", "Shortening", "Fat or oil", "Ghee", "Margarine spread",
    "Margarine", "Butter &/or dairy blend", "Butter", "Dairy blend",
    "Sauce", "Dressing", "Spread", "Marmalade", "Jam", "Honey",
    "Syrup", "Treacle or molasses", "Chutney or relish", "Mustard",
    "Mayonnaise", "Dijonaise", "Wasabi", "Mustard powder", "Vinegar",
    "Vinegar (except balsamic vinegar)", "Stock", "Gravy", "Gravy powder",
    "Cordial", "Cordial base", "Beverage base", "Coffee whitener",
    "Sugar", "Glucose", "Intense sweetener", "Salt", "Salt substitute",
    "Pepper", "Yeast", "Flour", "Cornmeal (polenta)", "Semolina", "Tapioca",
    "Sago", "Breadcrumbs", "Breadcrumbs for coating food", "Coating",
    "Batter for coating food", "Gelatine", "Gluten", "Lecithin", "Pectin",
    "Starch", "Cream of tartar", "Vanilla", "Essence", "Baking powder",
    "Baking soda (bicarbonate)", "Cocoa powder", "Vegetable powder",
    "Vegetable &/or fruit powder", "Maca powder", "Sherbet powder",
    "Custard powder", "Coffee mix powder", "Chai latte mix",
    "Coffee substitute", "Coffee substitute mix", "Coffee & chicory essence",
    "Coffee & milk concentrate", "Juice concentrate", "Icing mix", "Icing",
    "Topping", "Lemon butter", "Almond spread", "Cashew spread",
    "Mixed nut spread", "Mixed nut & seed spread", "Cheese spread",
    "Fish paste or spread", "Rice paper wrapper", "Stuffing", "Lemon peel",
    "Fruit peel", "Beef extract", "Miso", "Casserole base", "Tahini",
    "Pate de foie (chicken liver pate)", "Pate", "Meat paste", "Peanut butter",
    "Horseradish", "Capers",

    # Herbs & spices (flavour, not standalone food)
    "Basil", "Chives", "Coriander", "Dill", "Marjoram", "Mint", "Oregano",
    "Parsley", "Rosemary", "Sage", "Thyme", "Cardamom seed", "Cinnamon",
    "Cloves", "Coriander seed", "Cumin (cummin) seed", "Curry powder",
    "Nutmeg", "Paprika", "Seasoning", "Seasoning mix", "Spices", "Turmeric",
    "Garlic", "Ginger", "Herbs", "Tamarind",

    # Alcohol — we don't recommend alcoholic drinks
    "Beer", "Wine", "Spirit", "Liqueur", "Cocktail", "Cider", "Sherry",
    "Port", "Mulled wine", "Wine cooler", "Mirin", "Sake", "Mixed drink",
    "Bitters", "Spider",

    # Supplements & baby food
    "Protein powder", "Protein drink", "Meal replacement drink",
    "Meal replacement powder", "Oral supplement drink",
    "Oral supplement powder", "Amino acid or creatine drink",
    "Amino acid or creatine powder", "Fibre powder", "Prebiotic powder",
    "Very low energy diet drink", "Very low energy diet drink (Optifast)",
    "Very low energy diet powder", "Medical or special purpose drink",
    "Gel", "Caffeine", "Ethanol 100%", "Fibre", "Folic acid",
    "Maltodextrin", "Protein", "Thiamin", "Vitamin C", "Powder",
    "Infant food", "Infant cereal", "Infant snack", "Infant formula",
    "Infant custard or yoghurt", "Infant pasta", "Infant rusk",
    "Toddler milk", "Babyccino", "Chewing gum",

    # Processed deli meats (cold cuts, not full meals)
    "Berliner", "Brawn (braun)", "Camp pie", "Devon", "Jerky",
    "Kabana or cabanossi", "Mortadella", "Pastrami", "Prosciutto",
    "Salami", "Saveloy", "Speck", "Strasburg", "Terrine",
    "Processed meat", "Spam", "Black pudding", "Frankfurt",

    # Exotic / Aussie native (opt-in later if user wants)
    "Insect", "Witchetty grubs", "Mutton-bird", "Possum", "Snake",
    "Wallaby", "Bandicoot", "Goanna", "Camel", "Crocodile", "Dugong",
    "Echidna", "Pigeon (squab)", "Mangrove worm", "Periwinkle (sea snail)",
    "Pipi", "Yabby (yabbie)", "Bush tomato", "Saltbush", "Kakadu plum",
    "Lotus seed", "Pandanus kernel", "Quandong", "Truffles",
    "Warrigal greens", "Lilly pilly", "Goji berry", "Wattle seed (acacia)",

    # Misc ingredient-like
    "Buttermilk", "Cream", "Eggnog (egg nog)", "Cake or bread",
    "Cheese fruit", "Sundae", "Vegetables & sausages", "Vegetables & steak",
    "Mixed dish",
}

# Categories eligible as a breakfast main
BREAKFAST_MAIN_CATEGORIES = {
    "Breakfast cereal", "Porridge", "Muesli", "Granola", "Muesli or granola",
    "Egg", "Omelette", "Eggs benedict", "Souffle", "Frittata", "Quiche",
    "Pancake", "Crepe or pancake", "Waffle", "French toast", "Crumpet", "Pikelet",
    "Bagel", "Acai", "Acai bowl",
    "Bacon & egg roll", "Breakfast burger", "Breakfast wrap",
    "Oats", "Oat bran", "Wheat bran", "Wheat germ", "Rice bran",
    "Smoothie",
}

# Categories eligible as a lunch/dinner main
LUNCH_DINNER_MAIN_CATEGORIES = {
    # Protein
    "Chicken", "Beef", "Lamb", "Pork", "Veal", "Mutton", "Buffalo", "Goat",
    "Turkey", "Duck", "Goose", "Quail", "Pheasant", "Guinea fowl",
    "Rabbit", "Kangaroo", "Emu", "Ostrich", "Venison", "Pig (pork)",
    "Sausage", "Bacon", "Ham", "Meatloaf", "Meatball or rissole",
    "Rissole or patty", "Meat", "Pork & beef", "Tandoori chicken",
    "Tandoori lamb", "Braised steak & onions", "Stew", "Cabbage roll",

    # Seafood
    "Fish", "Salmon", "Tuna", "Prawn", "Crab", "Lobster or crayfish",
    "Scallop", "Oyster", "Mussel", "Octopus", "Squid or calamari",
    "Mixed seafood", "Sardine", "Barramundi", "Bassa (basa)", "Bream",
    "Cod", "Cod or hake", "Flathead", "Flounder or sole", "Grouper",
    "John dory", "Kingfish", "Ling", "Mullet", "Orange roughy",
    "Silver perch", "Snapper", "Stingray", "Swordfish", "Trout", "Whiting",
    "Mackerel", "Moreton bay bug", "Mixed seafood (except finfish)",
    "Shark (flake)", "Anchovy", "Blue grenadier (hoki)",
    "Fish (including eel & trout)", "Herring", "Tuna mornay",
    "Salmon patty or cake", "Fish patty or cake", "Salt & peppered squid",
    "Tandoori prawn", "Prawn cocktail", "Whitebait", "Abalone",
    "Fish finger",

    # Composite dishes
    "Curry", "Stir-fry", "Casserole", "Pasta dish", "Pasta",
    "Pasta in cream based sauce", "Pasta in tomato based sauce",
    "Spaghetti in sauce", "Lasagne (Lasagna)", "Cannelloni", "Gnocchi",
    "Gnocchi dish", "Risotto", "Paella", "Macaroni & cheese", "Pilaf",
    "Fried rice", "Special fried rice", "Rice bowl", "Burrito bowl",
    "Poke bowl", "Sushi", "Pizza",

    # Burgers / sandwiches / wraps
    "Hamburger", "Bacon burger", "Chicken burger", "Fish burger",
    "Vegan burger", "Vegetable burger", "Vegetable or lentil burger",
    "Sandwich or roll", "Sandwich", "Sandwick", "Filled bread roll",
    "Bun", "Bao bun", "Steamed bun", "Kebab wrap", "Mexican wrap",
    "Chicken wrap", "Wrap", "Hot dog", "Nachos", "Taco", "Falafel",

    # Asian small dishes
    "Dumpling", "Dim sim", "Curry puff", "Samosa", "Spring roll",
    "Rice paper roll", "Chiko roll", "Sausage roll", "Pasty",
    "Prawn toast",

    # Veg/grain mains
    "Tofu (soy bean curd)", "Meat alternative",
    "Tempeh (fermented soy beans)",
    "Textured vegetable protein/soy granules",
    "Rice", "Noodle", "Noodles", "Noodles & pasta", "Cous cous", "Couscous",
    "Quinoa", "Barley", "Amaranth", "Buckwheat groats", "Bulgur (burghul",
    "Millet", "Spelt", "Teff", "Sorghum",

    # Pies, soups (as mains)
    "Pie", "Soup",  # soup is large enough to be a main in many cases

    # Misc
    "Frittata", "Quiche", "Moussaka", "Fritter", "Onion ring",
    "Zucchini slice", "Vegetables", "Mixed seafood (except finfish)",
    "Legumes/lentils with rice/noodles",
    "Sandwich", "Filled bread roll", "Burrito bowl",
    "Ready meal",
}

# Categories eligible as sides (vegetables, fruit, salads, breakfast accompaniments)
SIDE_CATEGORIES = {
    # Vegetables
    "Vegetables", "Mixed vegetables", "Cauliflower", "Broccoli", "Asparagus",
    "Lettuce", "Spinach", "Capsicum", "Bok choy or pak choy",
    "Brussels sprout", "Kale", "Cabbage", "Carrot", "Celery", "Celeriac",
    "Cucumber", "Pumpkin", "Squash", "Zucchini", "Mushroom", "Tomato",
    "Beetroot", "Eggplant", "Sweet potato", "Potato", "Onion", "Leek",
    "Corn", "Bean", "Lentil", "Pea", "Snow pea", "Sugar snap pea",
    "Chickpea", "Baked beans", "Mixed leafy greens", "Salad", "Rocket",
    "Avocado", "Bamboo shoot", "Choko", "Fennel", "Kohlrabi", "Okra",
    "Olive", "Seaweed", "Shallot", "Silverbeet", "Sprout", "Swede",
    "Taro", "Turnip", "Watercress", "Vine leaf", "Radish", "Yam",
    "Cassava", "Parsnip", "Endive", "Water chestnut", "Artichoke",
    "Artichoke heart", "Flower", "Samphire", "Gherkin",
    "Cauliflower rice", "Broccoli & cauliflower rice", "Pickles",

    # Carbs as side
    "Bread", "Bread or bread roll", "Bread roll",

    # Fruit and fruit salad
    "Fruit salad", "Mixed berry", "Mixed fruit", "Fruit", "Tropical fruit",
    "Apple", "Banana", "Pear", "Peach", "Strawberry", "Blueberry",
    "Raspberry", "Cranberry", "Currant", "Blackberry", "Mango",
    "Grape", "Pineapple", "Orange", "Kiwifruit", "Mandarin",
    "Plum", "Apricot", "Cherry", "Lemon", "Lime", "Melon",
    "Pomegranate", "Plantain", "Quince", "Dragon fruit", "Mulberry",
    "Persimmon", "Lychee", "Nectarine", "Passionfruit", "Fig",
    "Date", "Sultana", "Raisin", "Prune (dried plum)",
    "Tangerine or tangor", "Tangelo", "Babaco", "Custard apple",
    "Durian", "Feijoa", "Grapefruit", "Guava", "Jackfruit",
    "Mangosteen", "Pawpaw (papaya)", "Prickly pear", "Rambutan",
    "Star fruit", "Tamarillo", "Wax jambu", "Cumquat (kumquat)",
    "Nutgrass (nut grass)",

    # Dairy/cheese/egg as side or breakfast accompaniment
    "Yoghurt", "Cheese", "Egg",

    # Nuts & seeds (snack-side)
    "Nut", "Mixed nuts", "Mixed nuts & seeds", "Mixed nuts & dried fruit",
    "Nuts & seeds", "Seed", "Seeds", "Coconut",
}

# Drink categories — drinks are a separate role, so they're flagged via
# *_side_safe + reason 'drink' so the combo assembler knows to slot them as the
# drink role rather than as a side.
DRINK_CATEGORIES = {
    "Beverage", "Coffee", "Tea", "Juice", "Milk", "Soft drink", "Water",
    "Mineral water", "Smoothie", "Sports drink", "Energy drink",
    "Soy beverage", "Almond beverage", "Coconut beverage", "Oat beverage",
    "Rice beverage", "Cashew beverage", "Macadamia beverage",
    "Nut beverage", "Quinoa beverage", "Cereal beverage",
    "Pea protein beverage", "Seed beverage", "Hazelnut beverage",
    "Dairy milk alternative", "Flavoured milk", "Iced coffee", "Iced tea",
    "Chai latte", "Milkshake", "Thickshake", "Cordial", "Fruit drink",
    "Drink", "Kombucha", "Kava", "Bubble tea (boba tea)",
    "Vegetable &/or fruit drink", "Iced chocolate",
}

# Per-row dinner side priority based on type
DINNER_SIDE_FAMILY_MAP = {
    "salad": {"Salad", "Fruit salad", "Mixed leafy greens", "Lettuce", "Rocket",
              "Cucumber", "Tomato", "Coleslaw"},
    "vegetable": {"Mixed vegetables", "Cauliflower", "Broccoli", "Asparagus",
                  "Spinach", "Brussels sprout", "Kale", "Cabbage", "Carrot",
                  "Celery", "Pumpkin", "Squash", "Zucchini", "Mushroom",
                  "Bean", "Lentil", "Pea", "Snow pea", "Sugar snap pea",
                  "Sweet potato", "Potato", "Corn", "Eggplant", "Beetroot",
                  "Bok choy or pak choy", "Cauliflower rice",
                  "Broccoli & cauliflower rice", "Capsicum", "Vegetables"},
    "soup": {"Soup"},
}


def _dinner_side_family_from_cat(cat: str) -> str:
    for fam, cats in DINNER_SIDE_FAMILY_MAP.items():
        if cat in cats:
            return fam
    return "other"


def _dinner_side_priority(family: str) -> int:
    return {"salad": 0, "vegetable": 1, "soup": 2}.get(family, 9)


_NAME_BLOCK_TERMS_RE = re.compile(
    r"\b(?:uncooked|raw,?\s*unprepared|powder|concentrate|essence|paste|dehydrated|"
    r"dried\s*(?:powder|flake|mix)|baby\s*food|infant)\b",
    re.IGNORECASE,
)

# Additional patterns that disqualify FRUIT rows as standalone side dishes —
# a frozen banana or a cooked apple is usually a cooking ingredient, not a
# side. Doesn't apply to mains because cooked protein (e.g. "Chicken, roasted")
# is legit.
_FRUIT_PROCESSING_RE = re.compile(
    r"\b(?:frozen|cooked|stewed|preserved|canned|dehydrated|dried|puree|stewed)\b",
    re.IGNORECASE,
)

# Categories where the fruit-processing filter applies. Specifically these are
# fruit-and-similar categories where the unprocessed version is the natural side.
_FRUIT_LIKE_CATEGORIES = frozenset({
    "Apple", "Banana", "Pear", "Peach", "Strawberry", "Blueberry",
    "Raspberry", "Cranberry", "Currant", "Blackberry", "Mango",
    "Grape", "Pineapple", "Orange", "Kiwifruit", "Mandarin",
    "Plum", "Apricot", "Cherry", "Lemon", "Lime", "Melon",
    "Pomegranate", "Plantain", "Quince", "Dragon fruit", "Mulberry",
    "Persimmon", "Lychee", "Nectarine", "Passionfruit", "Fig",
    "Date", "Sultana", "Raisin", "Prune (dried plum)",
    "Mixed berry", "Mixed fruit", "Fruit", "Tropical fruit",
    "Tangerine or tangor", "Tangelo", "Babaco", "Custard apple",
    "Durian", "Feijoa", "Grapefruit", "Guava", "Jackfruit",
    "Mangosteen", "Pawpaw (papaya)", "Prickly pear", "Rambutan",
    "Star fruit", "Tamarillo", "Wax jambu", "Cumquat (kumquat)",
})


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add meal-slot safety columns using AUSNUT RecipeCategory as the canonical signal."""
    cat = df["RecipeCategory"].fillna("").str.strip()
    name = df["food_name"].fillna("").astype(str)

    # food_name patterns that should never be a meal even if the category is valid
    # (e.g. "Acai, powder" has category "Acai" but name "powder" — not a meal).
    is_unprepared = name.str.contains(_NAME_BLOCK_TERMS_RE, na=False)

    # Within fruit categories, processed forms (frozen, cooked, stewed, …) are
    # cooking ingredients, not stand-alone sides. "Banana, frozen" is for
    # smoothies; "Banana, cavendish, peeled, raw" is a real banana.
    is_processed_fruit = (
        cat.isin(_FRUIT_LIKE_CATEGORIES)
        & name.str.contains(_FRUIT_PROCESSING_RE, na=False)
    )

    # Start with everything blocked, then enable based on category
    df["breakfast_main_safe"] = cat.isin(BREAKFAST_MAIN_CATEGORIES) & ~is_unprepared
    df["lunch_main_safe"] = cat.isin(LUNCH_DINNER_MAIN_CATEGORIES) & ~is_unprepared
    df["dinner_main_safe"] = cat.isin(LUNCH_DINNER_MAIN_CATEGORIES) & ~is_unprepared

    df["breakfast_side_safe"] = (
        (cat.isin(SIDE_CATEGORIES) | cat.isin(DRINK_CATEGORIES))
        & ~is_unprepared & ~is_processed_fruit
    )
    df["lunch_side_safe"] = (
        (cat.isin(SIDE_CATEGORIES) | cat.isin(DRINK_CATEGORIES))
        & ~is_unprepared & ~is_processed_fruit
    )

    # Dinner side: previously veg/salad/soup only, but combo pools were getting
    # starved (sometimes only 2-3 sides survived filtering). Now allow the full
    # SIDE_CATEGORIES set so carb sides (Bread, Rice, Pasta) and vegetables can
    # all serve as dinner sides. Drinks remain too.
    df["dinner_side_safe"] = (
        (cat.isin(SIDE_CATEGORIES) | cat.isin(DRINK_CATEGORIES))
        & ~is_unprepared & ~is_processed_fruit
    )
    df["dinner_side_family"] = cat.map(_dinner_side_family_from_cat)
    df["dinner_side_priority"] = df["dinner_side_family"].map(_dinner_side_priority).astype("int16")

    # Reason columns — record why each row was accepted/rejected.
    df["breakfast_main_reason"] = df.apply(
        lambda r: "safe" if r["breakfast_main_safe"]
        else ("hard_block" if r["RecipeCategory"] in HARD_BLOCK_CATEGORIES
              else "not_breakfast_main"),
        axis=1,
    )
    df["lunch_main_reason"] = df.apply(
        lambda r: "safe" if r["lunch_main_safe"]
        else ("hard_block" if r["RecipeCategory"] in HARD_BLOCK_CATEGORIES
              else "not_lunch_main"),
        axis=1,
    )
    df["dinner_main_reason"] = df.apply(
        lambda r: "safe" if r["dinner_main_safe"]
        else ("hard_block" if r["RecipeCategory"] in HARD_BLOCK_CATEGORIES
              else "not_dinner_main"),
        axis=1,
    )
    df["breakfast_side_reason"] = df.apply(
        lambda r: ("drink_role" if r["RecipeCategory"] in DRINK_CATEGORIES and r["breakfast_side_safe"]
                   else "safe" if r["breakfast_side_safe"]
                   else "hard_block" if r["RecipeCategory"] in HARD_BLOCK_CATEGORIES
                   else "not_breakfast_side"),
        axis=1,
    )
    df["lunch_side_reason"] = df.apply(
        lambda r: ("drink_role" if r["RecipeCategory"] in DRINK_CATEGORIES and r["lunch_side_safe"]
                   else "safe" if r["lunch_side_safe"]
                   else "hard_block" if r["RecipeCategory"] in HARD_BLOCK_CATEGORIES
                   else "not_lunch_side"),
        axis=1,
    )
    df["dinner_side_reason"] = df.apply(
        lambda r: ("drink_role" if r["RecipeCategory"] in DRINK_CATEGORIES and r["dinner_side_safe"]
                   else "safe" if r["dinner_side_safe"]
                   else "hard_block" if r["RecipeCategory"] in HARD_BLOCK_CATEGORIES
                   else "not_dinner_side"),
        axis=1,
    )

    return df


_CANON_RE = re.compile(r"[^a-z0-9 ]")


def _canonical_name(name: str) -> str:
    """For deduplication. Strips parenthetical descriptors so 'Beer, light'
    and 'Beer, light (alcohol 1-2.9%)' collapse to the same row."""
    s = (name or "").lower()
    s = re.sub(r"\([^)]*\)", " ", s)  # drop parenthetical content
    s = _CANON_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to AUSNUT_dataset.csv")
    parser.add_argument("--out", default="data/processed/cleaned_food_data.duckdb")
    parser.add_argument("--extras", help="Optional CSV of hand-added foods (same schema)")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Skip dedup-by-canonical-name (default: dedup)")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"AUSNUT not found: {src}")

    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded AUSNUT: {len(df):,} rows")

    if args.extras:
        extras_path = Path(args.extras)
        if extras_path.exists():
            extras = pd.read_csv(extras_path)
            extras.columns = [c.strip() for c in extras.columns]
            for col in df.columns:
                if col not in extras.columns:
                    extras[col] = None
            df = pd.concat([df, extras[df.columns]], ignore_index=True)
            print(f"After extras  ({len(extras)} added): {len(df):,} rows")
        else:
            print(f"Extras file not found, skipping: {extras_path}")

    # Dedupe by canonical name
    if not args.keep_duplicates:
        df["_canon"] = df["food_name"].fillna("").map(_canonical_name)
        before = len(df)
        df = df[df["_canon"] != ""].drop_duplicates("_canon", keep="first").drop(columns=["_canon"])
        print(f"After dedup    : {len(df):,} (removed {before - len(df):,} duplicates / empty names)")

    # AUSNUT IS Australian — overwrite any inconsistencies
    df["is_australian"] = True

    # Role-aware serving sizes. AUSNUT reports per-100g nutrients, but a "100g
    # serving" is unrealistically small for mains/drinks. The engine targets
    # ~1,100 kcal for dinner at 2,200 kcal/day; if all foods are 100g servings,
    # main+side+drink combos peak around 500-600 kcal and the calorie-fit
    # filter rejects everything (valid_pool=0).
    #
    # Heuristic — derived from typical Australian portion guidelines:
    #   - Mains      : 250 g serving (chicken breast, fish fillet, pasta dish)
    #   - Sides      : 150 g serving (vegetables, rice, salad)
    #   - Drinks     : 250 g/ml serving (250 ml beverage)
    #   - Snack/etc. : 100 g default
    cat_for_serving = df["RecipeCategory"].fillna("").str.strip()
    is_main_cat = cat_for_serving.isin(
        BREAKFAST_MAIN_CATEGORIES | LUNCH_DINNER_MAIN_CATEGORIES
    )
    is_drink_cat = cat_for_serving.isin(DRINK_CATEGORIES)
    is_side_cat = cat_for_serving.isin(SIDE_CATEGORIES)

    # main wins over side when a category is in both (e.g. "Bread" is both lunch
    # main and side — recommend portion sized for main use).
    serving_grams = pd.Series(100.0, index=df.index)
    serving_grams = serving_grams.where(~is_side_cat, 150.0)
    serving_grams = serving_grams.where(~is_drink_cat, 250.0)
    serving_grams = serving_grams.where(~is_main_cat, 250.0)
    df["serving_grams"] = serving_grams

    scale = df["serving_grams"].astype(float) / 100.0
    for src_col, dst_col in [
        ("Calories_100g", "serving_calories"),
        ("protein_100g", "serving_protein"),
        ("carbs_100g", "serving_carbs"),
        ("fat_100g", "serving_fats"),
    ]:
        per100 = pd.to_numeric(df[src_col], errors="coerce").fillna(0.0)
        df[dst_col] = (per100 * scale).round(2)

    # Cheap health score from rating (1–5 → 0–1); will be replaced by a real
    # nutrient-derived score later.
    df["health_score"] = (
        pd.to_numeric(df["AggregatedRating"], errors="coerce").fillna(2.5) / 5.0
    ).clip(0, 1)

    # Apply v1's classifiers + AUSNUT-specific overrides
    print("Classifying meal-slot safety...")
    df = _annotate(df)

    # Summary
    total = len(df)
    print()
    print("=== Meal-slot eligibility ===")
    for col in [
        "breakfast_main_safe", "lunch_main_safe", "dinner_main_safe",
        "breakfast_side_safe", "lunch_side_safe", "dinner_side_safe",
    ]:
        n = int(df[col].sum())
        print(f"  {col:<22}: {n:>5,}  ({n/total:5.1%})")

    # Any-meal eligibility
    any_eligible = (
        df["breakfast_main_safe"] | df["lunch_main_safe"] | df["dinner_main_safe"]
        | df["breakfast_side_safe"] | df["lunch_side_safe"] | df["dinner_side_safe"]
    )
    print(f"\n  Eligible for >=1 meal slot: {int(any_eligible.sum()):,}")

    # Write to DuckDB
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = duckdb.connect(str(out))
    con.register("df", df)
    con.execute("CREATE TABLE cleaned_food_data AS SELECT * FROM df")
    con.close()
    print(f"\nWrote {out} ({out.stat().st_size / 1024:.1f} KB / {out.stat().st_size / 1024 / 1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
