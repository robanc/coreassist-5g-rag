FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer-free console output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/root/.cache/huggingface"

# Copy the uv executable from the official uv container image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# libgomp1 is required by numerical and machine-learning libraries.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a separate layer for better build caching.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy the application source.
COPY . .

# Complete the environment after the source is available.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8501

CMD ["streamlit", "run", "app/CoreAssist.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]