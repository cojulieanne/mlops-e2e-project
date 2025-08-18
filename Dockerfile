FROM python:3.12-slim

WORKDIR /app

# System deps just in case
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Create venv
RUN uv venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app:/app/src"

# Copy pyproject + lock
COPY pyproject.toml uv.lock* ./

# Install Airflow with postgres extras
RUN uv pip install "apache-airflow[postgres]==3.0.3"
RUN pip install --no-cache-dir gdown

# Re-force psycopg2-binary explicitly (ensures it lands in venv site-packages)
RUN pip install --no-cache-dir "apache-airflow[postgres]==3.0.3" psycopg2-binary


RUN pip install --no-cache-dir asyncpg

# Now compile + install your project deps
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    rm requirements.txt

COPY src/ ./src/

CMD ["python", "src/run_pipeline.py"]