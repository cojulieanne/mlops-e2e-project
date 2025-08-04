FROM python:3.12.2

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Create a virtual environment inside the image
RUN uv venv /app/venv

# Activate the venv for all future RUN/CMD instructions
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app:/app/src"

# Copy pyproject and lock file
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    rm requirements.txt

# Create project folders
RUN mkdir -p data/bronze data/silver data/gold models reports logs

# Copy source code
COPY src/ ./src/

# Set default command
CMD ["python", "src/run_pipeline.py"]
