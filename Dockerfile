# FROM python:3.12-slim

# # Set working directory
# WORKDIR /app

# # Install uv
# RUN pip install --no-cache-dir uv

# # Create a virtual environment inside the image
# RUN uv venv /opt/venv
# RUN mkdir -p data
# RUN mkdir -p data/bronze
# RUN mkdir -p data/silver
# RUN mkdir -p data/gold
# RUN mkdir -p models
# RUN mkdir -p reports
# RUN mkdir -p logs

# # Activate the venv for all future RUN/CMD instructions
# ENV VIRTUAL_ENV=/app/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# ENV PYTHONPATH="/app:/app/src"

# # Copy pyproject and lock file
# COPY pyproject.toml uv.lock* ./

# # Install dependencies in the venv (no --system needed)
# RUN uv pip compile pyproject.toml -o requirements.txt && \
#     uv pip install -r requirements.txt && \
#     rm requirements.txt

# # Copy source code
# COPY src/ ./src/


# # Set default command
# CMD ["python", "src/run_pipeline.py"]
# # CMD ["tail", "-f", "/dev/null"]

# Stage 1: Grab uv binary from clean Python
FROM python:3.12-slim as uv-builder
RUN pip install uv && cp $(which uv) /uv

# Stage 2: Your airflow image
FROM apache/airflow:3.0.3

# Copy uv binary into final image
COPY --from=uv-builder /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# 👇 Step 1: Create clean isolated venv (avoids pre-installed bad packages)
RUN uv venv /app/venv

# 👇 Step 2: Ensure venv is used
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Step 3: Copy your pyproject and lock file
COPY --chown=airflow:0 pyproject.toml uv.lock* ./

# Step 4: Install dependencies into fresh venv
RUN uv pip install -r pyproject.toml

# Step 5: Copy your app and data folders
COPY --chown=airflow:0 src/ ./src/
COPY --chown=airflow:0 data/bronze/ ./data/bronze/
COPY --chown=airflow:0 data/silver/ ./data/silver/
COPY --chown=airflow:0 data/gold/ ./data/gold/
COPY --chown=airflow:0 models/ ./models/
COPY --chown=airflow:0 reports/ ./reports/
COPY --chown=airflow:0 logs/ ./logs/

# Default command
CMD ["python", "src/run_pipeline.py"]
