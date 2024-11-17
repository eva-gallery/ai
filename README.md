# EVA Gallery AI API

This service is a BentoML service that provides an API for the EVA Gallery AI and uses a separate injected BentoML inference service which is currently part of the same container, but can be deployed separately at a later time.

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

See `settings.yaml` for the full configuration structure. Environment variables override values in `settings.yaml`, and `settings.yaml` only serves as a fallback for unset environment variables.

Ensure that the `ENV_FOR_DYNACONF` environment variable is set to the appropriate environment, out of `[development, production, testing]` so that the correct fallback configuration is used.

Recommended list of environment variables / categories to set:
- `EVA_AI_POSTGRES__*`
- `EVA_AI_EVA_BACKEND__*`
- `EVA_AI_MODEL__CACHE_DIR`

The rest of the environment variables are ideally left to be developer managed through `settings.yaml`.
