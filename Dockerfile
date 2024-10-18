# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y build-essential curl git

ENV POETRY_VIRTUALENVS_CREATE=true

RUN pip install --upgrade pip && pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-root

COPY . .



FROM python:3.10-slim

# Install runtime dependencies (e.g., curl for health checks)
RUN apt-get update && apt-get install -y curl && apt-get clean

WORKDIR /app
COPY --from=builder /app /app

ENV BENTOML_HOME="/bentoml"

EXPOSE 3000

# Set the entry point to start BentoML service
ENTRYPOINT ["bentoml", "serve", "--port", "3000"]
