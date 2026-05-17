"""
Local Flask app for manually curating food images.

Browse the Australian-eligible food rows in a grocery-website-style grid.
For each food, see the current image (Tier 0 curated / Tier 1 food101 /
Tier 2 woolworths / Tier 3 off), and replace it by pasting a new URL.

Curations are saved to data/processed/curated_images.parquet. After curating,
re-run rebuild_food_db.py with --curated to bake the new URLs into the DuckDB.

Usage:
    python data/scripts/curate_images.py \\
        --db data/processed/cleaned_food_data.duckdb \\
        --curated data/processed/curated_images.parquet

    Then open http://127.0.0.1:5050 in your browser.

The app binds to 127.0.0.1 only — it's a local tool, not for deployment.
"""
from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from flask import Flask, jsonify, request

# Reuse the same category sets used to classify safety, so the curation tool's
# role filter agrees with what the engine actually treats as drinks/mains/sides.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ausnut_db import (  # noqa: E402
    BREAKFAST_MAIN_CATEGORIES,
    DRINK_CATEGORIES,
    LUNCH_DINNER_MAIN_CATEGORIES,
    SIDE_CATEGORIES,
)

_ALL_MAIN_CATEGORIES = BREAKFAST_MAIN_CATEGORIES | LUNCH_DINNER_MAIN_CATEGORIES


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Food Image Curation</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 0; background: #f6f7f8; color: #222; }
  header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #e0e2e4;
           padding: 12px 20px; z-index: 100; }
  header h1 { margin: 0 0 8px 0; font-size: 18px; }
  .filters { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .filters select, .filters input { padding: 6px 10px; border: 1px solid #ccc;
                                     border-radius: 6px; font-size: 13px; }
  .filters input[type=text] { width: 260px; }
  .stats { color: #888; font-size: 12px; margin-left: auto; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 14px; padding: 18px; }
  .card { background: #fff; border-radius: 10px; padding: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.05); display: flex; flex-direction: column; }
  .card img { width: 100%; height: 160px; object-fit: cover; border-radius: 6px;
              background: #eef0f2; }
  .card .placeholder { width: 100%; height: 160px; border-radius: 6px; background: #eef0f2;
                       display: flex; align-items: center; justify-content: center; color: #aaa;
                       font-size: 12px; }
  .card .name { font-size: 13px; font-weight: 600; margin: 8px 0 4px;
                line-height: 1.3; min-height: 34px; overflow: hidden; }
  .card .meta { font-size: 11px; color: #666; margin-bottom: 8px; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px;
           font-weight: 600; text-transform: uppercase; margin-right: 4px; }
  .badge.curated { background: #d4edda; color: #155724; }
  .badge.food101 { background: #cce5ff; color: #004085; }
  .badge.woolworths { background: #fff3cd; color: #856404; }
  .badge.off { background: #f8d7da; color: #721c24; }
  .badge.none { background: #e2e3e5; color: #555; }
  .badge.role-main { background: #fde2e4; color: #842029; }
  .badge.role-side { background: #d1e7dd; color: #0f5132; }
  .badge.role-drink { background: #cfe2ff; color: #084298; }
  .actions { display: flex; gap: 6px; margin-top: auto; }
  .actions button { flex: 1; padding: 6px 8px; border: 1px solid #ccc; background: #fff;
                    border-radius: 6px; cursor: pointer; font-size: 12px; }
  .actions button.primary { background: #0d6efd; color: #fff; border-color: #0d6efd; }
  .actions button:hover { opacity: .85; }
  .pager { display: flex; justify-content: center; gap: 8px; padding: 20px; }
  .pager button { padding: 8px 14px; border: 1px solid #ccc; background: #fff;
                  border-radius: 6px; cursor: pointer; }
  .pager span { padding: 8px 14px; }
  .modal-bg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.4); z-index: 200; }
  .modal { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
           background: #fff; padding: 22px; border-radius: 12px; width: 460px; max-width: 90vw;
           box-shadow: 0 8px 24px rgba(0,0,0,.2); z-index: 201; }
  .modal h2 { margin: 0 0 14px; font-size: 16px; }
  .modal input { width: 100%; padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px;
                 font-size: 13px; margin-bottom: 10px; }
  .modal .preview { width: 100%; height: 200px; object-fit: contain; background: #eef0f2;
                    border-radius: 6px; margin-bottom: 10px; }
  .modal .actions button { padding: 8px 14px; }
</style>
</head>
<body>
<header>
  <h1>Food Image Curation</h1>
  <div class="filters">
    <input type="text" id="search" placeholder="Search food name…">
    <select id="filter-category">
      <option value="">All meal slots</option>
      <option value="breakfast">Breakfast eligible</option>
      <option value="lunch">Lunch eligible</option>
      <option value="dinner">Dinner eligible</option>
    </select>
    <select id="filter-source">
      <option value="">Any image source</option>
      <option value="curated">Curated (Tier 0)</option>
      <option value="food101">Food-101 (Tier 1)</option>
      <option value="woolworths">Woolworths (Tier 2)</option>
      <option value="off">OFF (Tier 3)</option>
      <option value="none">Missing image</option>
    </select>
    <select id="filter-role">
      <option value="">Any role</option>
      <option value="main">Mains only</option>
      <option value="side">Sides only</option>
      <option value="drink">Drinks only</option>
    </select>
    <select id="sort">
      <option value="missing-first">Sort: missing images first</option>
      <option value="name">Sort: name A→Z</option>
    </select>
    <span class="stats" id="stats">Loading…</span>
  </div>
</header>

<div class="grid" id="grid"></div>

<div class="pager">
  <button onclick="changePage(-1)">← Prev</button>
  <span id="page-info">Page 1</span>
  <button onclick="changePage(1)">Next →</button>
</div>

<div class="modal-bg" id="modal-bg" onclick="closeModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <h2 id="modal-title">Replace image</h2>
    <div style="font-size: 12px; color: #666; margin-bottom: 10px;" id="modal-food"></div>
    <input type="text" id="modal-url" placeholder="Paste image URL (jpg/png)">
    <img class="preview" id="modal-preview" src="" alt="">
    <div class="actions">
      <button onclick="closeModal()">Cancel</button>
      <button class="primary" onclick="saveCuration()">Save</button>
    </div>
  </div>
</div>

<script>
let currentPage = 1;
let currentFoodId = null;

function buildQuery() {
  const params = new URLSearchParams({
    page: currentPage,
    page_size: 24,
    search: document.getElementById('search').value,
    slot: document.getElementById('filter-category').value,
    source: document.getElementById('filter-source').value,
    role: document.getElementById('filter-role').value,
    sort: document.getElementById('sort').value,
  });
  return params.toString();
}

async function loadFoods() {
  const resp = await fetch('/api/foods?' + buildQuery());
  const data = await resp.json();
  renderGrid(data);
  document.getElementById('stats').textContent =
    `${data.total.toLocaleString()} foods · ${data.curated.toLocaleString()} curated · ${data.missing.toLocaleString()} missing`;
  document.getElementById('page-info').textContent = `Page ${currentPage} / ${data.total_pages}`;
}

function renderGrid(data) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const food of data.foods) {
    const card = document.createElement('div');
    card.className = 'card';
    const img = food.image_url
      ? `<img src="${food.image_url}" onerror="this.outerHTML='<div class=placeholder>image not loading</div>'">`
      : `<div class="placeholder">no image</div>`;
    const roleBadges = (food.role || []).map(r => `<span class="badge role-${r}">${r}</span>`).join('');
    card.innerHTML = `
      ${img}
      <div class="name">${escapeHtml(food.food_name || '(no name)')}</div>
      <div class="meta">
        <span class="badge ${food.source}">${food.source}</span>
        ${roleBadges}
        <span style="opacity:.6">id: ${food.food_id}</span>
      </div>
      <div class="actions">
        <button onclick="clearCuration('${food.food_id}')">Mark wrong</button>
        <button class="primary" onclick="openModal('${food.food_id}', ${JSON.stringify(food.food_name).replace(/"/g, '&quot;')})">Replace</button>
      </div>
    `;
    grid.appendChild(card);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function openModal(foodId, foodName) {
  currentFoodId = foodId;
  document.getElementById('modal-food').textContent = foodName;
  document.getElementById('modal-url').value = '';
  document.getElementById('modal-preview').src = '';
  document.getElementById('modal-bg').style.display = 'block';
}

function closeModal(e) {
  if (e && e.target.id !== 'modal-bg') return;
  document.getElementById('modal-bg').style.display = 'none';
}

document.getElementById('modal-url').addEventListener('input', e => {
  document.getElementById('modal-preview').src = e.target.value;
});

async function saveCuration() {
  const url = document.getElementById('modal-url').value.trim();
  if (!url) return alert('Paste an image URL first.');
  const resp = await fetch('/api/curate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({food_id: currentFoodId, image_url: url}),
  });
  if (!resp.ok) return alert('Save failed: ' + (await resp.text()));
  closeModal();
  loadFoods();
}

async function clearCuration(foodId) {
  if (!confirm('Mark this image as wrong? It will be blanked from the curated parquet.')) return;
  const resp = await fetch('/api/curate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({food_id: foodId, image_url: ''}),
  });
  if (!resp.ok) return alert('Failed: ' + (await resp.text()));
  loadFoods();
}

function changePage(delta) {
  currentPage = Math.max(1, currentPage + delta);
  loadFoods();
}

document.getElementById('search').addEventListener('input', () => { currentPage = 1; loadFoods(); });
document.getElementById('filter-category').addEventListener('change', () => { currentPage = 1; loadFoods(); });
document.getElementById('filter-source').addEventListener('change', () => { currentPage = 1; loadFoods(); });
document.getElementById('filter-role').addEventListener('change', () => { currentPage = 1; loadFoods(); });
document.getElementById('sort').addEventListener('change', () => { currentPage = 1; loadFoods(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal({target: {id: 'modal-bg'}}); });

loadFoods();
</script>
</body>
</html>"""


# --- Source classification ---------------------------------------------------

def _classify_source(url: str, curated_ids: set[str], food_id: str) -> str:
    """Tag where this image likely came from based on the URL host."""
    if not url:
        return "none"
    if food_id in curated_ids:
        return "curated"
    if "/static/food101/" in url:
        return "food101"
    if "woolworths.media" in url:
        return "woolworths"
    if "openfoodfacts.org" in url or "off." in url:
        return "off"
    return "curated"  # unknown URL — assume manually added


def _classify_role(recipe_category: str) -> list[str]:
    """Return the list of roles (main/side/drink) this food can play."""
    roles: list[str] = []
    if recipe_category in _ALL_MAIN_CATEGORIES:
        roles.append("main")
    if recipe_category in SIDE_CATEGORIES:
        roles.append("side")
    if recipe_category in DRINK_CATEGORIES:
        roles.append("drink")
    return roles


# --- Curation storage --------------------------------------------------------

class CurationStore:
    """Thread-safe append-only store backed by a parquet file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        if path.exists():
            self._df = pd.read_parquet(path)
        else:
            self._df = pd.DataFrame(columns=["food_id", "image_url", "curated_at"])
        self._df["food_id"] = self._df["food_id"].astype(str)

    def set(self, food_id: str, image_url: str) -> None:
        with self._lock:
            food_id = str(food_id)
            now = datetime.now(timezone.utc).isoformat()
            self._df = self._df[self._df["food_id"] != food_id]
            self._df = pd.concat(
                [self._df, pd.DataFrame([{"food_id": food_id, "image_url": image_url, "curated_at": now}])],
                ignore_index=True,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._df.to_parquet(self.path, index=False)

    def get(self, food_id: str) -> str | None:
        row = self._df[self._df["food_id"] == str(food_id)]
        if row.empty:
            return None
        return row.iloc[-1]["image_url"]

    def ids(self) -> set[str]:
        return set(self._df[self._df["image_url"].astype(str) != ""]["food_id"].tolist())


# --- Flask app ---------------------------------------------------------------

def create_app(db_path: Path, curated_path: Path) -> Flask:
    app = Flask(__name__)
    store = CurationStore(curated_path)

    @app.route("/")
    def index():
        return HTML_PAGE

    @app.route("/api/foods")
    def foods():
        page = max(1, int(request.args.get("page", 1)))
        page_size = max(1, min(100, int(request.args.get("page_size", 24))))
        search = (request.args.get("search") or "").strip()
        slot = (request.args.get("slot") or "").strip()
        source = (request.args.get("source") or "").strip()
        role = (request.args.get("role") or "").strip()
        sort = (request.args.get("sort") or "missing-first").strip()

        con = duckdb.connect(str(db_path), read_only=True)

        # Only Australian-eligible rows are meaningful for curation
        where = ["COALESCE(is_australian, FALSE) = TRUE"]
        if slot == "breakfast":
            where.append("(breakfast_main_safe OR breakfast_side_safe)")
        elif slot == "lunch":
            where.append("(lunch_main_safe OR lunch_side_safe)")
        elif slot == "dinner":
            where.append("(dinner_main_safe OR dinner_side_safe)")
        if search:
            where.append(f"LOWER(food_name) LIKE '%{search.lower().replace(chr(39), chr(39)+chr(39))}%'")
        where_sql = " AND ".join(where) if where else "TRUE"

        all_rows = con.execute(
            f"SELECT RecipeId, food_name, RecipeCategory, image_url FROM cleaned_food_data WHERE {where_sql}"
        ).fetchall()
        con.close()

        curated_ids = store.ids()
        rows = []
        for rid, name, recipe_cat, url in all_rows:
            food_id = str(rid)
            # If curated, override with curated URL
            curated_url = store.get(food_id)
            effective_url = curated_url if (curated_url is not None and curated_url != "") else (url or "")
            src = _classify_source(effective_url, curated_ids, food_id)
            row_role = _classify_role(recipe_cat or "")
            rows.append({
                "food_id": food_id,
                "food_name": name or "",
                "category": recipe_cat or "",
                "image_url": effective_url,
                "source": src,
                "role": row_role,
            })

        # Source filter
        if source:
            rows = [r for r in rows if r["source"] == source]

        # Role filter (main / side / drink) — uses RecipeCategory as the signal
        if role:
            rows = [r for r in rows if role in r["role"]]

        # Sort: missing-first puts uncovered rows at the top so user curates them
        if sort == "missing-first":
            rows.sort(key=lambda r: (0 if r["source"] == "none" else (1 if r["source"] == "off" else 2), r["food_name"].lower()))
        else:
            rows.sort(key=lambda r: r["food_name"].lower())

        total = len(rows)
        missing = sum(1 for r in rows if r["source"] == "none")
        curated_count = sum(1 for r in rows if r["source"] == "curated")
        start = (page - 1) * page_size
        page_rows = rows[start:start + page_size]
        total_pages = max(1, (total + page_size - 1) // page_size)

        return jsonify({
            "foods": page_rows,
            "total": total,
            "curated": curated_count,
            "missing": missing,
            "total_pages": total_pages,
        })

    @app.route("/api/curate", methods=["POST"])
    def curate():
        data = request.get_json() or {}
        food_id = str(data.get("food_id") or "").strip()
        image_url = str(data.get("image_url") or "").strip()
        if not food_id:
            return "food_id required", 400
        store.set(food_id, image_url)
        return jsonify({"ok": True})

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/processed/cleaned_food_data.duckdb",
                        help="Path to the rebuilt DuckDB (default: data/processed/cleaned_food_data.duckdb)")
    parser.add_argument("--curated", default="data/processed/curated_images.parquet",
                        help="Path to the curated-images parquet (created if missing)")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DuckDB not found: {db_path}")

    curated_path = Path(args.curated)
    app = create_app(db_path, curated_path)
    print(f"Serving curation UI at http://127.0.0.1:{args.port}")
    print(f"DuckDB         : {db_path}")
    print(f"Curated parquet: {curated_path}")
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
