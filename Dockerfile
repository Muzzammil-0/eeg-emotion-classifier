FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Render provides PORT env var; fallback to 10000 locally
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} sdf:app"]
