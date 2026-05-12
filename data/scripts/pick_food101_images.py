"""
Pick one canonical image per Food-101 category and bundle them for upload.

Reads:   food_recognition/data/food-101/food-101/images/<category>/<file>.jpg
Writes:  machine_learning_v2/data/processed/food101_curated/<category>.jpg  (101 files)
         machine_learning_v2/data/processed/food101_curated.zip             (single upload)

The script picks the first JPG alphabetically per category. If you don't like a
specific category's pick, just replace that single file before zipping.

Usage:
    python data/scripts/pick_food101_images.py
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # machine_learning_v2/
REPO_ROOT = ROOT.parent                               # meal_planning_app/
FOOD101_SRC = REPO_ROOT / "food_recognition" / "data" / "food-101" / "food-101" / "images"
CLASSES_JSON = REPO_ROOT / "food_recognition" / "data" / "splits" / "food101_classes.json"

OUT_DIR = ROOT / "data" / "processed" / "food101_curated"
OUT_ZIP = ROOT / "data" / "processed" / "food101_curated.zip"


def main() -> int:
    if not FOOD101_SRC.exists():
        print(f"ERROR: Food-101 image folder not found at {FOOD101_SRC}", file=sys.stderr)
        return 1
    if not CLASSES_JSON.exists():
        print(f"ERROR: food101_classes.json not found at {CLASSES_JSON}", file=sys.stderr)
        return 1

    with open(CLASSES_JSON) as fh:
        classes = json.load(fh)  # {"0": "apple_pie", "1": "baby_back_ribs", ...}

    categories = sorted(set(classes.values()))
    print(f"Found {len(categories)} Food-101 categories")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    picked = 0
    missing = []
    for cat in categories:
        cat_dir = FOOD101_SRC / cat
        if not cat_dir.is_dir():
            missing.append(cat)
            continue
        jpgs = sorted(cat_dir.glob("*.jpg"))
        if not jpgs:
            missing.append(cat)
            continue
        dst = OUT_DIR / f"{cat}.jpg"
        shutil.copy2(jpgs[0], dst)
        picked += 1

    print(f"Picked {picked}/{len(categories)} images → {OUT_DIR}")
    if missing:
        print(f"WARNING: missing categories: {missing}")

    # Bundle into a single zip for upload to GitHub Release.
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(OUT_DIR.glob("*.jpg")):
            zf.write(f, arcname=f.name)
    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Bundle written: {OUT_ZIP} ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  1. Inspect the curated folder. Replace any image you don't like with a better")
    print(f"     one from food_recognition/data/food-101/food-101/images/<category>/.")
    print(f"  2. Rerun this script to refresh the zip, OR just rezip the folder manually.")
    print(f"  3. Upload {OUT_ZIP.name} to a new GitHub Release (e.g. tag v1.1.0-images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
