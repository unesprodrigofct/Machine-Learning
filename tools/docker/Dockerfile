# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==1.8.3 \
    && poetry config virtualenvs.create false \
    && poetry install --without dev

COPY src ./src
COPY configs ./configs
COPY data ./data

# Default command overridden in runtime environments
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dev
RUN poetry install --with dev

FROM base AS production
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
