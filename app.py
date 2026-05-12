from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# Add src/ to path so `recommendation_engine` resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommendation_engine import RecommendationService  # noqa: E402

load_dotenv()

app = Flask(__name__)
recommendation_service = RecommendationService.from_env()

# Curated Food-101 dish images live at /app/static/food101/<category>.jpg inside the container.
STATIC_FOOD101_DIR = Path(__file__).resolve().parent / "static" / "food101"


@app.route("/static/food101/<path:filename>", methods=["GET"])
def food101_image(filename: str):
    return send_from_directory(STATIC_FOOD101_DIR, filename, max_age=86400)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({
        "ok": True,
        "food101_images_available": STATIC_FOOD101_DIR.exists() and any(STATIC_FOOD101_DIR.glob("*.jpg")),
    })


@app.route("/api/prime", methods=["POST"])
def prime():
    try:
        data = request.json or {}
        payload = recommendation_service.prime_user_context(data)
        print("**** API /prime completed")
        return jsonify(payload)
    except Exception as exc:
        print("Prime Error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/prime/status", methods=["POST"])
def prime_status():
    try:
        data = request.json or {}
        payload = recommendation_service.get_prime_response_warmup_status(data)
        print("**** API /prime/status completed")
        return jsonify(payload)
    except Exception as exc:
        print("Prime Status Error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/runtime-metrics", methods=["POST"])
def runtime_metrics():
    try:
        payload = recommendation_service.get_runtime_metrics()
        print("**** API /runtime-metrics completed")
        return jsonify(payload)
    except Exception as exc:
        print("Runtime Metrics Error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/recommendation", methods=["POST"])
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json or {}
        payload = recommendation_service.recommend(data)
        meal_type = (data or {}).get("mealType") or (data or {}).get("slot") or "all"
        print(f"**** API /recommend completed: meal_type={meal_type}")
        return jsonify(payload)
    except Exception as exc:
        print("Recommendation Error:", exc)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    debug_flag = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}
    host = str(os.getenv("HOST", "0.0.0.0")).strip() or "0.0.0.0"
    port = int(str(os.getenv("PORT", "8080")).strip() or "8080")
    app.run(host=host, port=port, debug=debug_flag)
