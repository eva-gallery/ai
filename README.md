# EVA Gallery AI API

## Quickstart

1. Install dependencies with `poetry install` while in the folder
2. For full functionality, set up and configure local timescale pg image
3. Run `poetry run alembic upgrade head` to apply migrations
4. `bentoml serve .` to start the API server

## Deployment

1. Build the service with `bentoml build`
2. Containerize with `bentoml containerize evagallery_ai_api:latest --opt progress=plain` (requires Docker with Buildkit)
3. Run `docker run -p 3000:3000 evagallery_ai_api:latest` to start the containerized API server

## Configuration

The service can be configured through environment variables using the prefix `EVA_AI` followed by category and subcategory with double underscores.

For example:
- `EVA_AI_BENTOML__API__WORKERS=10` 
- `EVA_AI_POSTGRES__HOST=localhost`

See `settings.yaml` for the full configuration structure.
