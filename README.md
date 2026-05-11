# machine_learning_v2

Sibling rewrite of `machine_learning/` with a cleaner data pipeline.

The headline differences from v1:

1. **Canonical train / val / test split.** Stratified 70/15/15 by major food group, deterministic seed, written once to `data/splits/*.json` and never regenerated. The test manifest is touched only at release time.
2. **Images are first-class.** `cleaned_food_data` is rebuilt with an `image_url` column populated by an offline enrichment job (OpenFoodFacts first, FatSecret on miss). The runtime engine returns image URLs natively — backend doesn't need to enrich.
3. **Clean separation between data, code, and results.** `data/`, `src/`, `train/`, `evaluate/`, `results/` directories instead of a single flat `dataset_process/` with everything mixed in.
4. **v1 keeps running.** This folder is offline until Phase 5 of the migration.

For the full plan see [Checklists & Planning/Others/MACHINE_LEARNING_V2_PLAN.md](../Checklists%20&%20Planning/Others/MACHINE_LEARNING_V2_PLAN.md).

---

## Folder layout

```
machine_learning_v2/
├── README.md                       you are here
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                        source datasets, gitignored (regenerate from source)
│   ├── processed/                  cleaned_food_data.duckdb, food_images.parquet (gitignored)
│   ├── splits/                     train/val/test manifests (committed)
│   └── scripts/
│       ├── build_canonical_split.py
│       ├── enrich_food_images.py
│       └── rebuild_food_db.py
│
├── src/
│   └── recommendation_engine/      runtime ranking engine (ported from v1)
│
├── train/                          weight-fitting scripts
├── evaluate/
│   ├── evaluate_dev.py             val-only — used during iteration
│   └── evaluate_test.py            test-only — release-time only
│
├── results/                        run logs (gitignored)
│
├── tests/                          unit tests
└── benchmarks_archive/             zipped v1 benchmark folders
```

---

## Initial setup

```powershell
cd machine_learning_v2
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add a `.env` based on `.env.example` (once it exists; for now copy v1's `.env`).

---

## Data pipeline — order matters

Run these once in order. Each step is resumable and idempotent.

### 1. Enrich images (offline, takes hours)

```powershell
python data/scripts/enrich_food_images.py `
    --db ..\machine_learning\dataset_process\off.db `
    --table cleaned_food_data `
    --out data\processed\food_images.parquet
```

Source order is OpenFoodFacts API (free, no quota) → FatSecret search (paid quota, via the Fly.io proxy at `https://goodhealthmate-fs.fly.dev`). The script flushes to parquet every 200 rows so a Ctrl+C never loses progress; rerun with the same arguments to resume.

Required env (only for the FatSecret fallback):
```
FATSECRET_CLIENT_ID=...
FATSECRET_CLIENT_SECRET=...
```

Pass `--skip-fatsecret` for a free OFF-only dry run.

### 2. Rebuild the food DuckDB with images

```powershell
python data/scripts/rebuild_food_db.py `
    --src-db ..\machine_learning\dataset_process\off.db `
    --src-table cleaned_food_data `
    --images data\processed\food_images.parquet `
    --out data\processed\cleaned_food_data.duckdb `
    --out-table cleaned_food_data
```

Joins the v1 source table with the images parquet on `RecipeId` and writes a fresh DuckDB at `data/processed/cleaned_food_data.duckdb` with the new `image_url` column.

### 3. Build the canonical split

```powershell
python data/scripts/build_canonical_split.py `
    --db data\processed\cleaned_food_data.duckdb `
    --table cleaned_food_data `
    --out data\splits\
```

Writes `train_manifest.json`, `val_manifest.json`, `test_manifest.json`, `split_metadata.json`. Commit these. Do not regenerate the split unless you bump the version intentionally.

---

## Where v2 currently is

- [x] Folder skeleton in place
- [x] Phase 1: data pipeline run — `food_images.parquet` enriched (56% coverage), `cleaned_food_data.duckdb` rebuilt with `image_url`, canonical 70/15/15 split written to `data/splits/`
- [x] Phase 2: pipeline executed (4,027 Australian-eligible rows enriched, 2,341 with images)
- [x] Phase 3: `recommendation_engine/` ported from v1, `image_url` surfaces in responses; baseline eval: p50 ~35 ms, aus=100%, img=100%, slot=100%
- [x] Phase 4: evaluators written (`evaluate_dev.py`, `evaluate_test.py`), baseline logged to `results/val_logs/`; `app.py` and `Dockerfile` written
- [ ] Phase 5: deploy v2 to Railway, update backend `ML_SERVICE_URL`, smoke-test, cut over, decommission v1
