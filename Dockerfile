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

# Download the pre-built DuckDB at image build time.
# Set these two as Railway Build Variables (not env vars):
#   FOOD_DB_URL    = https://github.com/USER/REPO/releases/download/v1.0.0/cleaned_food_data.duckdb
#   GITHUB_TOKEN   = your classic PAT with repo scope
ARG FOOD_DB_URL
ARG GITHUB_TOKEN
RUN apt-get update -qq && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p data/processed \
    && if [ -n "$FOOD_DB_URL" ]; then \
           echo "Downloading DuckDB..." \
           && wget -q \
                --header="Authorization: token ${GITHUB_TOKEN}" \
                --header="Accept: application/octet-stream" \
                -O data/processed/cleaned_food_data.duckdb \
                "$FOOD_DB_URL" \
           && echo "Download complete: $(du -sh data/processed/cleaned_food_data.duckdb)"; \
       else \
           echo "FOOD_DB_URL not set — skipping download"; \
       fi

EXPOSE 8080
ENV PORT=8080

# 2 workers: one always-warm, one handles bursts. 120 s timeout covers cold DB open.
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info"]
