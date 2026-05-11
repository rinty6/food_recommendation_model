"""
Enrich every food in cleaned_food_data with an image URL.

Source order:
    1. OpenFoodFacts API  (free, open license, no quota, ~25M products)
    2. FatSecret search   (paid quota, via Fly proxy, better recipe coverage)

Output:
    data/processed/food_images.parquet
        food_id (str), image_url (str|null), source (str|null), last_verified_at (iso str)

The script is:
    - resumable    (skips ids already present in the output parquet)
    - idempotent   (re-running with same input produces same output)
    - rate-limited (sleeps between calls so we don't hammer either API)

Usage:
    python data/scripts/enrich_food_images.py \\
        --db ../machine_learning/dataset_process/off.db \\
        --table cleaned_food_data \\
        --out data/processed/food_images.parquet \\
        --limit 1000          # optional, default = all rows
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


# Tune these if APIs throttle. OFF guideline is ~100 req/min/app — stay under that across all workers combined.
OFF_RATE_LIMIT_SECONDS = 0.5
FATSECRET_RATE_LIMIT_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 8

OFF_SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
OFF_FIELDS = "code,product_name,image_url,image_front_url,image_small_url"

# FatSecret hits go through the static-IP proxy we set up at fly.io.
# These constants must match the ones in machine_learning/recommendation_engine/fatsecret.py
FATSECRET_TOKEN_URL = "https://goodhealthmate-fs.fly.dev/connect/token"
FATSECRET_API_URL = "https://goodhealthmate-fs.fly.dev/rest/server.api"


# --------------------------------------------------------------------------- OFF

def search_openfoodfacts(query: str, session: requests.Session) -> str | None:
    params = {
        "search_terms": query,
        "fields": OFF_FIELDS,
        "page_size": 1,
        "sort_by": "popularity_key",
    }
    try:
        resp = session.get(OFF_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS,
                           headers={"User-Agent": "GoodHealthMate-Enrichment/1.0 (open dataset prep)"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        for product in (data.get("products") or [])[:1]:
            for key in ("image_front_url", "image_url", "image_small_url"):
                value = product.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
    except (requests.RequestException, ValueError):
        return None
    return None


# ---------------------------------------------------------------------- FATSECRET

class FatSecretClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token: str | None = None
        self.token_expires_at: float = 0.0

    def _refresh_token(self) -> None:
        resp = requests.post(
            FATSECRET_TOKEN_URL,
            data={"grant_type": "client_credentials", "scope": "premier"},
            auth=(self.client_id, self.client_secret),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        body = resp.json()
        self.token = body["access_token"]
        self.token_expires_at = time.time() + max(60, int(body.get("expires_in", 3600)) - 60)

    def _ensure_token(self) -> str:
        if not self.token or time.time() >= self.token_expires_at:
            self._refresh_token()
        return self.token  # type: ignore[return-value]

    def search_image(self, query: str) -> str | None:
        try:
            token = self._ensure_token()
            params = {
                "method": "foods.search",
                "search_expression": query,
                "format": "json",
                "max_results": "1",
                "include_food_images": "true",
            }
            resp = requests.get(
                FATSECRET_API_URL,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            foods = (data.get("foods") or {}).get("food")
            if foods is None:
                return None
            if isinstance(foods, dict):
                foods = [foods]
            for food in foods[:1]:
                food_image = food.get("food_image")
                if isinstance(food_image, dict):
                    url = food_image.get("image_url")
                    if isinstance(url, str) and url.startswith("http"):
                        return url
                if isinstance(food_image, str) and food_image.startswith("http"):
                    return food_image
        except (requests.RequestException, ValueError):
            return None
        return None


# --------------------------------------------------------------------------- IO

def load_input_rows(
    db_path: Path,
    table: str,
    id_col: str,
    name_col: str,
    where_clause: str = "",
) -> list[tuple[str, str]]:
    con = duckdb.connect(str(db_path), read_only=True)
    sql = f"SELECT {id_col}, COALESCE({name_col}, '') FROM {table}"
    if where_clause.strip():
        sql += f" WHERE {where_clause}"
    rows = con.execute(sql).fetchall()
    con.close()
    return [(str(r[0]).strip(), str(r[1]).strip()) for r in rows if str(r[0]).strip()]


def load_done_set(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path, columns=["food_id"])
    return set(df["food_id"].astype(str).tolist())


def append_results(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["food_id"], keep="last")
    else:
        combined = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)


# --------------------------------------------------------------------------- MAIN

def iterate_pending(rows: Iterable[tuple[str, str]], done: set[str]) -> Iterable[tuple[str, str]]:
    for food_id, name in rows:
        if food_id in done or not name:
            continue
        yield food_id, name


def lookup_one(
    food_id: str,
    name: str,
    session: requests.Session,
    fatsecret_client: "FatSecretClient | None",
    rate_lock: threading.Lock,
) -> dict:
    """Resolve a single food. Each worker calls this. Rate limits applied via the shared lock."""
    image_url: str | None = None
    source: str | None = None

    # Serialize OFF calls across workers so we never exceed ~2 req/s aggregate (under OFF's 100/min guideline).
    with rate_lock:
        time.sleep(OFF_RATE_LIMIT_SECONDS)
        image_url = search_openfoodfacts(name, session)
    if image_url:
        source = "openfoodfacts"

    if not image_url and fatsecret_client is not None:
        with rate_lock:
            time.sleep(FATSECRET_RATE_LIMIT_SECONDS)
            image_url = fatsecret_client.search_image(name)
        if image_url:
            source = "fatsecret"

    return {
        "food_id": food_id,
        "image_url": image_url,
        "source": source,
        "last_verified_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to source DuckDB file")
    parser.add_argument("--table", default="cleaned_food_data")
    parser.add_argument("--id-column", default="RecipeId")
    parser.add_argument("--name-column", default="food_name")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N new lookups (0 = all)")
    parser.add_argument("--flush-every", type=int, default=200, help="Append to parquet every N rows")
    parser.add_argument("--skip-fatsecret", action="store_true", help="OFF only, useful for local dry-runs")
    parser.add_argument(
        "--where",
        default="",
        help='Optional SQL WHERE clause to narrow input rows (e.g. "breakfast_main_safe OR lunch_main_safe").',
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent worker threads (default 1).")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    rows = load_input_rows(db_path, args.table, args.id_column, args.name_column, where_clause=args.where)
    print(f"Loaded {len(rows)} rows from {db_path}" + (f" (filtered: {args.where})" if args.where else ""))
    done = load_done_set(out_path)
    print(f"Already enriched: {len(done)}. Pending: {len(rows) - len(done)}.")

    fatsecret_client: FatSecretClient | None = None
    if not args.skip_fatsecret:
        client_id = os.getenv("FATSECRET_CLIENT_ID", "").strip()
        client_secret = os.getenv("FATSECRET_CLIENT_SECRET", "").strip()
        if client_id and client_secret:
            fatsecret_client = FatSecretClient(client_id, client_secret)
        else:
            print("FATSECRET_CLIENT_ID / FATSECRET_CLIENT_SECRET not set — skipping FatSecret fallback.")

    session = requests.Session()
    rate_lock = threading.Lock()
    buffer: list[dict] = []
    counts = {"off_hit": 0, "fatsecret_hit": 0, "miss": 0, "processed": 0}
    started_at = time.time()

    pending = list(iterate_pending(rows, done))
    if args.limit:
        pending = pending[: args.limit]

    try:
        if args.workers <= 1:
            # Sequential path keeps memory + log flow simple for small runs.
            for food_id, name in pending:
                result = lookup_one(food_id, name, session, fatsecret_client, rate_lock)
                if result["source"] == "openfoodfacts":
                    counts["off_hit"] += 1
                elif result["source"] == "fatsecret":
                    counts["fatsecret_hit"] += 1
                else:
                    counts["miss"] += 1
                buffer.append(result)
                counts["processed"] += 1
                if len(buffer) >= args.flush_every:
                    append_results(out_path, buffer)
                    buffer = []
                    elapsed = time.time() - started_at
                    rate = counts["processed"] / max(1.0, elapsed)
                    print(
                        f"[{counts['processed']:>6}/{len(pending)}]  "
                        f"off={counts['off_hit']}  fatsecret={counts['fatsecret_hit']}  miss={counts['miss']}  "
                        f"rate={rate:.2f}/s"
                    )
        else:
            # Concurrent path. The rate_lock serializes the actual API calls so OFF doesn't get hammered;
            # workers parallelize JSON parsing and the per-call wait time, giving a real speedup without
            # blowing the 100 req/min guideline.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(lookup_one, food_id, name, session, fatsecret_client, rate_lock): food_id
                    for food_id, name in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result["source"] == "openfoodfacts":
                        counts["off_hit"] += 1
                    elif result["source"] == "fatsecret":
                        counts["fatsecret_hit"] += 1
                    else:
                        counts["miss"] += 1
                    buffer.append(result)
                    counts["processed"] += 1
                    if len(buffer) >= args.flush_every:
                        append_results(out_path, buffer)
                        buffer = []
                        elapsed = time.time() - started_at
                        rate = counts["processed"] / max(1.0, elapsed)
                        eta_s = (len(pending) - counts["processed"]) / max(0.01, rate)
                        print(
                            f"[{counts['processed']:>6}/{len(pending)}]  "
                            f"off={counts['off_hit']}  fatsecret={counts['fatsecret_hit']}  miss={counts['miss']}  "
                            f"rate={rate:.2f}/s  eta={eta_s/3600:.1f}h"
                        )

    except KeyboardInterrupt:
        print("\nInterrupted — flushing buffer before exit.")

    append_results(out_path, buffer)

    print()
    print("=== Enrichment summary ===")
    print(f"  processed     : {counts['processed']}")
    print(f"  OFF hits      : {counts['off_hit']}")
    print(f"  FatSecret hits: {counts['fatsecret_hit']}")
    print(f"  misses        : {counts['miss']}")
    if counts["processed"] > 0:
        cover = (counts["off_hit"] + counts["fatsecret_hit"]) / counts["processed"]
        print(f"  coverage      : {cover:.1%}")
    print(f"  output        : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
