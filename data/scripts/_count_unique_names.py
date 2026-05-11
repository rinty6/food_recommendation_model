"""Count rows + distinct names across progressively tighter filters."""
from pathlib import Path
import sys

import duckdb

DB = Path(__file__).resolve().parents[3] / "machine_learning" / "dataset_process" / "off.db"

if not DB.exists():
    print(f"DB not found at {DB}", file=sys.stderr)
    sys.exit(1)

con = duckdb.connect(str(DB), read_only=True)

ELIGIBILITY = (
    "(COALESCE(breakfast_main_safe, FALSE) OR COALESCE(lunch_main_safe, FALSE) "
    "OR COALESCE(dinner_main_safe, FALSE) OR COALESCE(breakfast_side_safe, FALSE) "
    "OR COALESCE(lunch_side_safe, FALSE) OR COALESCE(dinner_side_safe, FALSE))"
)
NOT_UNKNOWN = "LOWER(TRIM(food_name)) <> 'unknown product'"
HAS_NAME = "food_name IS NOT NULL AND TRIM(food_name) <> ''"
AUSTRALIAN = "COALESCE(is_australian, FALSE) = TRUE"

LEVELS = [
    ("eligibility only", f"{ELIGIBILITY} AND {HAS_NAME}"),
    ("+ drop 'unknown product'", f"{ELIGIBILITY} AND {HAS_NAME} AND {NOT_UNKNOWN}"),
    ("+ Australian only (recommended)", f"{ELIGIBILITY} AND {HAS_NAME} AND {NOT_UNKNOWN} AND {AUSTRALIAN}"),
]

print(f"{'filter level':<35}  {'rows':>10}  {'unique names':>13}  {'rows/name':>9}")
print("-" * 75)
for label, where in LEVELS:
    rows = con.execute(f"SELECT COUNT(*) FROM cleaned_food_data WHERE {where}").fetchone()[0]
    unique = con.execute(
        f"SELECT COUNT(DISTINCT LOWER(TRIM(food_name))) FROM cleaned_food_data WHERE {where}"
    ).fetchone()[0]
    ratio = rows / max(1, unique)
    print(f"{label:<35}  {rows:>10,}  {unique:>13,}  {ratio:>9.1f}")

# Sample the recommended-level top names so we can confirm they're English.
print()
print("Top 10 names at the recommended level:")
recommended_where = f"{ELIGIBILITY} AND {HAS_NAME} AND {NOT_UNKNOWN} AND {AUSTRALIAN}"
top_repeats = con.execute(
    f"""
    SELECT LOWER(TRIM(food_name)) AS name, COUNT(*) AS occurrences
    FROM cleaned_food_data
    WHERE {recommended_where}
    GROUP BY 1
    ORDER BY occurrences DESC
    LIMIT 10
    """
).fetchall()
for name, n in top_repeats:
    print(f"  {n:>6}x  {name[:80]}")
