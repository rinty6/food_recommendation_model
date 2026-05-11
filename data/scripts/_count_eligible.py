"""Quick one-off: count how many cleaned_food_data rows are eligible for any meal slot."""
from pathlib import Path
import sys

import duckdb

DB = Path(__file__).resolve().parents[3] / "machine_learning" / "dataset_process" / "off.db"

if not DB.exists():
    print(f"DB not found at {DB}", file=sys.stderr)
    sys.exit(1)

con = duckdb.connect(str(DB), read_only=True)

total = con.execute("SELECT COUNT(*) FROM cleaned_food_data").fetchone()[0]
eligible = con.execute(
    """
    SELECT COUNT(*) FROM cleaned_food_data
    WHERE COALESCE(breakfast_main_safe, FALSE)
       OR COALESCE(lunch_main_safe, FALSE)
       OR COALESCE(dinner_main_safe, FALSE)
       OR COALESCE(breakfast_side_safe, FALSE)
       OR COALESCE(lunch_side_safe, FALSE)
       OR COALESCE(dinner_side_safe, FALSE)
    """
).fetchone()[0]

print(f"total      : {total:,}")
print(f"eligible   : {eligible:,}  ({eligible/total:.1%})")
print(f"non-elig.  : {total - eligible:,}  ({(total-eligible)/total:.1%})")
