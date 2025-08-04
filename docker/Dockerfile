FROM python:3.12.2

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Create a virtual environment inside the image
RUN uv venv /opt/venv
RUN mkdir -p data
RUN mkdir -p data/bronze
RUN mkdir -p data/silver
RUN mkdir -p data/gold
RUN mkdir -p models
RUN mkdir -p reports
RUN mkdir -p logs

# Activate the venv for all future RUN/CMD instructions
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app:/app/src"

# Copy pyproject and lock file
COPY pyproject.toml uv.lock* ./

# Install dependencies in the venv (no --system needed)
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt && \
    rm requirements.txt

# Copy source code
COPY src/ ./src/


# Set default command
CMD ["python", "src/run_pipeline.py"]
# CMD ["tail", "-f", "/dev/null"]