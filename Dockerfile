# --- build stage: install Python deps ---
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder (keeps final image lean)
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY app.py .

# Download data assets at image build time.
# Set these as Railway Build Variables (not env vars):
#   FOOD_DB_URL          = github release URL for cleaned_food_data.duckdb
#   FOOD101_ZIP_URL      = github release URL for food101_curated.zip (optional, Tier 1 images)
#   GITHUB_TOKEN         = classic PAT with repo scope (only needed for private releases)
ARG FOOD_DB_URL
ARG FOOD101_ZIP_URL
ARG GITHUB_TOKEN
RUN apt-get update -qq && apt-get install -y --no-install-recommends wget ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p data/processed static/food101 \
    && if [ -n "$FOOD_DB_URL" ]; then \
           echo "Downloading DuckDB..." \
           && wget -q \
                --header="Authorization: token ${GITHUB_TOKEN}" \
                --header="Accept: application/octet-stream" \
                -O data/processed/cleaned_food_data.duckdb \
                "$FOOD_DB_URL" \
           && echo "DuckDB: $(du -sh data/processed/cleaned_food_data.duckdb)"; \
       else \
           echo "FOOD_DB_URL not set — skipping DuckDB download"; \
       fi \
    && if [ -n "$FOOD101_ZIP_URL" ]; then \
           echo "Downloading Food-101 curated images..." \
           && wget -q \
                --header="Authorization: token ${GITHUB_TOKEN}" \
                --header="Accept: application/octet-stream" \
                -O /tmp/food101.zip \
                "$FOOD101_ZIP_URL" \
           && unzip -q /tmp/food101.zip -d static/food101/ \
           && rm /tmp/food101.zip \
           && echo "Food-101 images: $(ls static/food101/ | wc -l) files"; \
       else \
           echo "FOOD101_ZIP_URL not set — Tier 1 images disabled"; \
       fi

EXPOSE 8080
ENV PORT=8080

# 2 workers: one always-warm, one handles bursts. 120 s timeout covers cold DB open.
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info"]
