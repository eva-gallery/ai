"""Main API service module for the AI API.

This module provides the main API service that handles image processing, search operations,
and database interactions. It coordinates between the inference service, database,
and external services while managing authentication and request processing.

The service provides endpoints for:
- Image processing and watermarking
- Similarity-based image search
- Text-based image search
- Health and readiness checks
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import uuid
from functools import lru_cache
from typing import Any, cast

import aiohttp
import bentoml
import jwt
import numpy as np
from fastapi import FastAPI, UploadFile
from jwt.exceptions import InvalidTokenError
from PIL.Image import Image as PILImage  # noqa: TC002
from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import JSONResponse

from ai_api import API_SERVICE_KWARGS, settings
from ai_api.database import AIOPostgres
from ai_api.model import (
    AddWatermarkRequest,
    AIGeneratedStatus,
    APIServiceProto,
    BackendPatchRequest,
    ImageDuplicateStatus,
    ImageSearchResponse,
    InferenceServiceProto,
    RawImageSearchRequest,
    SearchResponse,
)
from ai_api.orm import GalleryEmbedding as ORMGalleryEmbedding, Image as ORMImage
from ai_api.services import InferenceService
from ai_api.util import get_logger


@lru_cache(maxsize=4096)
async def _get_and_cache_query_embeddings(query: str, embedding_service: InferenceServiceProto) -> tuple[float, ...]:
    """Get and cache text query embeddings.

    :param query: Text query to embed.
    :type query: str
    :param embedding_service: Service to generate embeddings.
    :type embedding_service: InferenceServiceProto
    :returns: Tuple of embedding values.
    :rtype: tuple[float, ...]
    """
    return tuple((await embedding_service.embed_text([query]))[0])


@lru_cache(maxsize=4096)
async def _search_image_id_cache(image_id: int) -> tuple[float] | None:
    """Get cached image embeddings by ID.

    :param image_id: Database ID of the image.
    :type image_id: int
    :returns: Tuple of embedding values if found.
    :rtype: tuple[float] | None
    """
    async with AIOPostgres() as conn:
        # First get the embedding for the input image_id
        embedding_stmt = select(ORMGalleryEmbedding).where(ORMGalleryEmbedding.image_id == image_id)
        embedding_result = await conn.execute(embedding_stmt)
        embedding = await embedding_result.scalar_one_or_none()  # type: ignore[misc]
        return tuple(embedding.image_embedding) if embedding else None


app = FastAPI()


@bentoml.service(
    name="evagallery_ai_api",
    **API_SERVICE_KWARGS,
)
@bentoml.asgi_app(app)
class APIService(APIServiceProto):
    """Main API service for handling image processing and search operations.

    This service coordinates between the inference service, database, and external services
    to provide image processing, search, and management capabilities.

    Attributes:
        embedding_service: Service for generating embeddings and AI operations.
        logger: Logger instance for service logging.
        ctx: BentoML context for request handling.
        db_healthy: Flag indicating database health status.
        background_tasks: Set of running background tasks.
        jwt_secret: Secret for JWT token validation.

    """

    def __init__(self) -> None:
        """Initialize the API service with required dependencies."""
        self.embedding_service: InferenceServiceProto = cast(InferenceServiceProto, bentoml.depends(InferenceService))

        self.logger = get_logger()
        self.ctx = bentoml.Context()
        self.db_healthy = False
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.jwt_secret = settings.jwt_secret
        asyncio.run_coroutine_threadsafe(self._init_postgres(), asyncio.get_event_loop())

    def _verify_jwt(self, ctx: bentoml.Context) -> None:
        """Verify JWT token from request Authorization header.

        :param ctx: BentoML request context.
        :type ctx: bentoml.Context
        :raises ValueError: If token is missing or invalid.
        """
        if os.getenv("CI"):
            return

        auth_header = ctx.request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            ctx.response.status_code = 401
            raise ValueError

        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except InvalidTokenError as e:
            ctx.response.status_code = 401
            msg = f"Invalid JWT token: {e!s}"
            raise ValueError(msg) from e

    async def _init_postgres(self) -> None:
        """Initialize the Postgres database connection.

        :raises Exception: If database initialization fails.
        """
        try:
            AIOPostgres(url=settings.postgres.url)
            self._migrate_database()
        except Exception as e:
            self.logger.exception("Failed to initialize Postgres client: {e}", e=e)
            raise
        self.db_healthy = True

    def _migrate_database(self) -> None:
        """Run database migrations using Alembic.

        This method is skipped in CI environments.
        """
        if os.getenv("CI"):
            return
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")

    async def _save_and_notify(
        self,
        image_pil: PILImage,
        duplicate_status: ImageDuplicateStatus,
        image_model: ORMImage,
        gallery_embedding_model: ORMGalleryEmbedding,
        most_similar_image_uuid: uuid.UUID | None = None,
    ) -> None:
        """Save image data and notify external services.

        :param image_pil: PIL image to save.
        :type image_pil: PILImage
        :param duplicate_status: Status indicating if image is duplicate.
        :type duplicate_status: ImageDuplicateStatus
        :param image_model: Database model for image data.
        :type image_model: ORMImage
        :param gallery_embedding_model: Database model for image embeddings.
        :type gallery_embedding_model: ORMGalleryEmbedding
        :param most_similar_image_uuid: UUID of most similar image if duplicate.
        :type most_similar_image_uuid: uuid.UUID | None
        :raises: May raise exceptions during database or HTTP operations.
        """
        # Prepare headers with JWT token
        headers = {
            "Authorization": f"Bearer {jwt.encode({'sub': 'ai-service'}, self.jwt_secret, algorithm='HS256')}",
        }

        if duplicate_status is not ImageDuplicateStatus.OK:
            patch_request = BackendPatchRequest(
                image_uuid=image_model.image_uuid,
                closest_match_uuid=most_similar_image_uuid,
                image_duplicate_status=duplicate_status,
                modified_image_uuid=None,
                ai_generated_status=image_model.generated_status,
                metadata=image_model.image_metadata,
            )

            data = aiohttp.FormData()
            data.add_field("json", patch_request.model_dump_json(), content_type="application/json")

            async with aiohttp.ClientSession() as session, session.patch(
                settings.eva_backend.backend_image_patch_route,
                data=data,
                headers=headers,
            ) as response:
                if response.status not in (200, 201):
                    self.logger.error("Failed to patch backend: {response}", response=await response.text())

        async with AIOPostgres() as conn:
            # Add and flush image data model
            conn.add(image_model)
            await conn.flush()

            # Set image_id on gallery embedding and add
            gallery_embedding_model.image_id = image_model.id
            conn.add(gallery_embedding_model)
            await conn.commit()

        # Convert PIL image to bytes for storage
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format=image_pil.format or "PNG")
        image_bytes.seek(0)

        # Send PATCH request to backend with full model
        patch_request = BackendPatchRequest(
            image_uuid=image_model.original_image_uuid,
            image_duplicate_status=ImageDuplicateStatus.OK,
            closest_match_uuid=None,
            modified_image_uuid=image_model.image_uuid,
            ai_generated_status=image_model.generated_status,
            metadata=image_model.image_metadata,
        )

        # Send PATCH request to backend with form data
        data = aiohttp.FormData()
        data.add_field("json", patch_request.model_dump_json(), content_type="application/json")
        data.add_field("image", image_bytes, filename="image.png", content_type="image/png")

        async with aiohttp.ClientSession() as session, session.patch(
            settings.eva_backend.backend_image_patch_route,
            data=data,
            headers=headers,
        ) as response:
            if response.status not in (200, 201):
                self.logger.error("Failed to patch backend: {response}", response=await response.text())

    async def _process_image_data(
        self,
        image_pil: PILImage,
        artwork_uuid: str,
        ai_generated_status: AIGeneratedStatus,
        metadata: dict[str, Any],
    ) -> None:
        """Process image data including embedding, duplicate checking, and watermarking.

        :param image_pil: PIL image to process.
        :type image_pil: PILImage
        :param artwork_uuid: UUID of the artwork.
        :type artwork_uuid: str
        :param ai_generated_status: Initial AI generation status.
        :type ai_generated_status: AIGeneratedStatus
        :param metadata: Additional image metadata.
        :type metadata: dict[str, Any]
        :raises: May raise exceptions during image processing or database operations.
        """
        # Generate a shorter hash that will fit in 32 bits
        original_hash = hashlib.sha256(image_pil.tobytes()).hexdigest()[:8]  # First 32 bits (8 hex chars)

        image_embed = (await self.embedding_service.embed_image([image_pil]))[0]

        duplicate_status = ImageDuplicateStatus.OK
        most_similar_image_uuid = None

        if metadata.get("ignore_duplicate_check", False) in (False, None):
            # Check for duplicates
            watermark_result, ai_watermark = await asyncio.gather(
                self.embedding_service.check_watermark([image_pil]),
                self.embedding_service.check_ai_watermark([image_pil]),
            )
            has_watermark, watermark = watermark_result[0]
            ai_watermark = ai_watermark[0]

            if ai_watermark:
                duplicate_status = ImageDuplicateStatus.PLAGIARIZED
            elif has_watermark and watermark:
                async with AIOPostgres().session() as conn:
                    result = await conn.execute(
                        select(ORMImage)
                        .filter_by(image_hash=watermark),  # Changed from md5_hash to image_hash
                    )
                    if result.scalar_one_or_none():
                        result = await conn.execute(
                            select(ORMGalleryEmbedding.image_embedding, ORMGalleryEmbedding.image_id)
                            .order_by(ORMGalleryEmbedding.image_embedding_distance_to(image_embed)),
                        )

                        float_vec, image_id = (await (await result.scalars()).one())  # type: ignore[misc]

                        result = await conn.execute(
                            select(ORMImage.image_uuid)
                            .where(ORMImage.id == image_id),
                        )

                        most_similar_image_uuid = await (await result.scalars()).one()  # type: ignore[misc]

                        # check if dot product is more than 95% similar
                        if np.dot(image_embed, float_vec) > settings.model.detection.threshold:
                            duplicate_status = ImageDuplicateStatus.EXISTS

        if duplicate_status is not ImageDuplicateStatus.OK:
            await self._save_and_notify(
                duplicate_status=duplicate_status,
                most_similar_image_uuid=most_similar_image_uuid,
                image_pil=image_pil,
                image_model=ORMImage(
                    original_image_uuid=artwork_uuid,
                    generated_status=ai_generated_status,
                    image_metadata=metadata,
                    image_hash=original_hash,
                ),
                gallery_embedding_model=ORMGalleryEmbedding(
                    image_embedding=image_embed,
                    watermarked_image_embedding=None,
                    metadata_embedding=None,
                    user_caption_embedding=None,
                    generated_caption_embedding=None,
                ),
            )
            return

        # Get embeddings and caption
        generated_image_caption, metadata_embedding = await asyncio.gather(
            self.embedding_service.generate_caption([image_pil]),
            self.embedding_service.embed_text([str(metadata)]),
        )

        generated_image_caption = generated_image_caption[0]
        metadata_embedding = metadata_embedding[0]

        # Detect if AI generated
        watermarked_image_pil = image_pil
        watermarked_image_embed = None
        if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
            ai_generated_status = (await self.embedding_service.detect_ai_generation([image_pil]))[0]
            if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
                watermarked_image_pil = (await self.embedding_service.add_ai_watermark([image_pil]))[0]

        watermarked_image_pil: PILImage = (await self.embedding_service.add_watermark([
            AddWatermarkRequest(image=watermarked_image_pil, watermark_text=original_hash),
        ]))[0]
        watermarked_image_embed, generated_caption_embed = await asyncio.gather(
            self.embedding_service.embed_image([watermarked_image_pil]),
            self.embedding_service.embed_text([generated_image_caption]),
        )
        watermarked_image_embed = watermarked_image_embed[0]
        generated_caption_embed = generated_caption_embed[0]

        user_caption = metadata.get("caption")
        user_caption_embed = None
        if user_caption:
            user_caption_embed = (await self.embedding_service.embed_text([user_caption]))[0]

        metadata["ai_generated"] = ai_generated_status
        metadata["duplicate_status"] = duplicate_status

        self.ctx.state["queued_processing"] -= 1

        # Save to database and send to backend
        await self._save_and_notify(
            duplicate_status=duplicate_status,
            image_pil=watermarked_image_pil,
            image_model=ORMImage(
                original_image_id=artwork_uuid,
                image_metadata=metadata,
                image_hash=original_hash,
            ),
            gallery_embedding_model=ORMGalleryEmbedding(
                image_embedding=image_embed,
                watermarked_image_embedding=watermarked_image_embed,
                metadata_embedding=metadata_embedding,
                user_caption_embedding=user_caption_embed,
                generated_caption_embedding=generated_caption_embed,
            ),
        )

    @app.get(path="/image/search_query", response_model=None)
    async def search_query(self, query: str, count: int = 50, page: int = 0) -> SearchResponse:
        """Search for images using a text query.

        :param query: Text query to search with.
        :type query: str
        :param count: Number of results per page.
        :type count: int
        :param page: Page number (0-based).
        :type page: int
        :returns: Search response containing matching image UUIDs.
        :rtype: SearchResponse
        :raises ValueError: If JWT token is invalid.
        """
        self._verify_jwt(self.ctx)
        embedded_text: tuple[float, ...] = await _get_and_cache_query_embeddings(
            query,
            self.embedding_service,
        )

        async with AIOPostgres() as conn:
            stmt = (
                select(ORMGalleryEmbedding.image_id)
                .order_by(ORMGalleryEmbedding.image_embedding_distance_to(embedded_text))
                .limit(count)
                .offset(page * count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore[misc]

        return SearchResponse(image_uuid=list(results))

    @app.get(path="/image/search_image", response_model=None)
    async def search_image(self, image_uuid: str, count: int = 50, page: int = 0) -> ImageSearchResponse:
        """Search for similar images using an image UUID.

        :param image_uuid: UUID of the reference image.
        :type image_uuid: str
        :param count: Number of results per page.
        :type count: int
        :param page: Page number (0-based).
        :type page: int
        :returns: Search response containing similar image UUIDs.
        :rtype: ImageSearchResponse
        :raises ValueError: If JWT token is invalid.
        """
        self._verify_jwt(self.ctx)
        image_uuid_obj: uuid.UUID = uuid.UUID(image_uuid)
        async with AIOPostgres() as conn:
            # First get the image ID from the UUID
            image_id_stmt = select(ORMImage.id).where(ORMImage.image_uuid == image_uuid_obj, ORMImage.public.is_(True))
            image_id_result = await conn.execute(image_id_stmt)
            image_id = await image_id_result.scalar_one_or_none()  # type: ignore[misc]

            if image_id is None:
                self.ctx.response.status_code = 404
                return ImageSearchResponse(image_uuid=[])

            # Get the embedding for this image ID
            embedding = await _search_image_id_cache(image_id)
            if embedding is None:
                self.ctx.response.status_code = 404
                return ImageSearchResponse(image_uuid=[])

            # Get similar images, excluding the input image
            stmt = (
                select(ORMImage.image_uuid)
                .join(ORMGalleryEmbedding, ORMGalleryEmbedding.image_id == ORMImage.id)
                .where(ORMImage.id != image_id)
                .order_by(ORMGalleryEmbedding.image_embedding_distance_to(embedding))
                .limit(count)
                .offset(page * count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore[misc]

        return ImageSearchResponse(image_uuid=list(results))

    @app.get(path="/image/search_image_raw", response_model=None)
    async def search_image_raw(
        self,
        image: UploadFile,
        count: int = 50,
        page: int = 0,
    ) -> ImageSearchResponse:
        """Search for similar images using a raw image.

        :param image: Raw image data to search with.
        :type image: UploadFile
        :param count: Number of results per page.
        :type count: int
        :param page: Page number (0-based).
        :type page: int
        :returns: Search response containing similar image UUIDs.
        :rtype: ImageSearchResponse
        :raises ValueError: If JWT token is invalid.
        """
        self._verify_jwt(self.ctx)
        if self.ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            self.ctx.response.status_code = 503
            return ImageSearchResponse(image_uuid=[])  # type: ignore[misc]

        self.ctx.state["queued_processing"] = self.ctx.state.get("queued_processing", 0) + 1

        # Open file as bytes IO stream
        image_bytes = io.BytesIO(await image.read())
        image_bytes.seek(0)
        image_pil = Image.open(image_bytes).convert("RGB")
        image_embed = (await self.embedding_service.embed_image([image_pil]))[0]

        self.ctx.state["queued_processing"] -= 1

        async with AIOPostgres() as conn:
            stmt = (
                select(ORMGalleryEmbedding.image_id)
                .order_by(ORMGalleryEmbedding.image_embedding_distance_to(image_embed))
                .limit(count)
                .offset(page * count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore[misc]

        return ImageSearchResponse(image_uuid=list(results))  # type: ignore[misc]

    @app.patch(path="/image/set-public", response_model=None)  # type: ignore[misc]
    async def publish_image(self, image_uuid: str | list[str]) -> JSONResponse:
        """Publish one or multiple images by setting their public flag to True.

        :param image_uuid: Single image UUID or list of image UUIDs to publish.
        :type image_uuid: str | list[str]
        :raises SQLAlchemyError: If the database update fails.
        :returns: JSONResponse with status code and error message if any.
        :rtype: JSONResponse
        """
        if isinstance(image_uuid, str):
            image_uuid = [image_uuid]

        async with AIOPostgres() as conn:
            try:
                stmt = select(ORMImage.image_uuid).where(ORMImage.image_uuid.in_(image_uuid))
                result = await conn.execute(stmt)
                existing_uuids = set(await (await result.scalars()).all())

                missing_uuids = set(image_uuid) - existing_uuids
                if missing_uuids:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": "One or more requested images not found",
                            "missing_uuids": list(missing_uuids),
                        },
                    )

                # Perform the update
                stmt = (
                    update(ORMImage)
                    .where(ORMImage.image_uuid.in_(image_uuid))
                    .values(public=True)
                )
                result = await conn.execute(stmt)

                # Double check affected rows
                if result.rowcount != len(image_uuid):
                    self.logger.exception("Expected to update {img_count} rows, but updated {row_count}", img_count=len(image_uuid), row_count=result.rowcount)
                    conn.rollback()
                    return JSONResponse(status_code=500, content={"error": "Failed to publish all images"})

            except SQLAlchemyError as e:
                self.logger.exception("Failed to publish images: {e}", e=e)
                raise

        return JSONResponse(status_code=200, content={})

    @bentoml.api(
        route="/image/process",
    )
    async def process_image(
        self,
        image: PILImage,
        image_uuid: str,
        ai_generated_status: str,
        metadata: dict[str, Any],
        ctx: bentoml.Context,
    ) -> None:
        """Process an image for storage and analysis.

        :param image: Image data to process.
        :type image: PILImage
        :param image_uuid: UUID for the image.
        :type image_uuid: str
        :param ai_generated_status: Initial AI generation status.
        :type ai_generated_status: str
        :param metadata: Additional image metadata.
        :type metadata: dict[str, Any]
        :param ctx: BentoML request context.
        :type ctx: bentoml.Context
        :raises ValueError: If JWT token is invalid.
        """
        self._verify_jwt(ctx)
        if ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            ctx.response.status_code = 503
            return

        image_pil = image.convert("RGB")

        task = asyncio.create_task(self._process_image_data(
            image_pil=image_pil,
            artwork_uuid=image_uuid,
            ai_generated_status=AIGeneratedStatus(ai_generated_status),
            metadata=metadata or {},
        ))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        ctx.state["queued_processing"] = ctx.state.get("queued_processing", 0) + 1
        ctx.response.status_code = 201

    @app.get(path="/healthz", response_model=dict[str, str])
    async def healthz(self) -> dict[str, str]:
        """Check service health status.

        :returns: Dictionary containing health status.
        :rtype: dict[str, str]
        """
        if not self.db_healthy:
            self.ctx.response.status_code = 503
            return {"status": "unhealthy"}

        self.ctx.response.status_code = 200
        return {"status": "healthy"}

    @app.get(path="/readyz", response_model=dict[str, str])
    async def readyz(self) -> dict[str, str]:
        """Check if service is ready to handle requests.

        :returns: Dictionary containing readiness status.
        :rtype: dict[str, str]
        :raises Exception: If readiness check fails.
        """
        if self.ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            self.ctx.response.status_code = 503
            return {"status": "not ready"}

        try:
            # Test database connection
            async with AIOPostgres().session() as conn:
                await conn.execute(select(1))

            # Test embedding service with timeout
            try:
                await asyncio.wait_for(self.embedding_service.readyz(), timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.exception("Embedding service readiness check timed out")
                raise
        except Exception as e:
            self.logger.exception("Readiness check failed: {e}", e=e)
            self.ctx.response.status_code = 503
            raise
        else:
            self.ctx.response.status_code = 200
            return {"status": "ready"}
