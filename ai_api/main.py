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
import io
import os
import traceback
import uuid
from functools import lru_cache
from typing import Any

import aiohttp
import bentoml
import jwt
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from imagehash import dhash, phash, whash
from jwt.exceptions import InvalidTokenError
from PIL import Image
from PIL.Image import Image as PILImage  # noqa: TC002
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from ai_api import API_SERVICE_KWARGS, settings
from ai_api.database import AIOPostgres
from ai_api.model import (
    AIGeneratedStatus,
    APIServiceProto,
    BackendPatchRequest,
    ImageDuplicateStatus,
    ImageSearchResponse,
    InferenceServiceProto,
    SearchResponse,
)
from ai_api.model.api.embed import EmbedRequest
from ai_api.orm import GalleryEmbedding as ORMGalleryEmbedding
from ai_api.orm import Image as ORMImage
from ai_api.services import InferenceService
from ai_api.util import get_logger

hash_method = {
    "phash": phash,
    "dhash": dhash,
    "whash": whash,
}


app = FastAPI()
logger = get_logger()


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
        hash_fn: Function to generate perceptual hashes.
        hash_threshold: Threshold for hash similarity comparison.

    """

    def __init__(self) -> None:
        """Initialize the API service with required dependencies."""
        self.logger = logger
        self.ctx = bentoml.Context()
        self.db_healthy = False
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.jwt_secret = settings.jwt_secret
        self.hash_fn = hash_method[settings.model.hash.method]
        self.hash_threshold = settings.model.hash.threshold
        self.embedding_service: InferenceServiceProto = None # type: ignore[]

        app.add_exception_handler(exc_class_or_status_code=Exception, handler=self.global_exception_handler)

        # Create initialization tasks
        loop = asyncio.get_event_loop()
        db_task = loop.create_task(self._init_postgres())
        embedding_task = loop.create_task(self._init_embedding_service())

        # Store tasks and add callbacks
        self.background_tasks.add(db_task)
        self.background_tasks.add(embedding_task)
        db_task.add_done_callback(lambda t: self._set_db_status(t))
        embedding_task.add_done_callback(lambda t: self._set_embedding_service(t))

    def _set_db_status(self, task: asyncio.Task[None]) -> None:
        """Set database health status from task result.

        :param task: Completed database initialization task.
        """
        try:
            task.result()
            self.db_healthy = True
        except Exception as e:
            self.logger.exception("Database initialization failed: {e}", e=e)
        finally:
            self.background_tasks.discard(task)

    def _set_embedding_service(self, task: asyncio.Task[InferenceServiceProto]) -> None:
        """Set embedding service from task result.

        :param task: Completed embedding service initialization task.
        """
        try:
            self.embedding_service = task.result()
        except Exception as e:
            self.logger.exception("Embedding service initialization failed: {e}", e=e)
        finally:
            self.background_tasks.discard(task)

    async def _init_postgres(self) -> None:
        """Initialize the Postgres database connection."""
        try:
            AIOPostgres(url=settings.postgres.url)
            self._migrate_database()
        except Exception as e:
            self.logger.exception("Failed to initialize Postgres client: {e}", e=e)
            raise

    async def _init_embedding_service(self) -> InferenceServiceProto:
        """Initialize the embedding service asynchronously.

        :returns: Initialized inference service.
        :rtype: InferenceServiceProto
        """
        worker_index = getattr(bentoml.server_context, "worker_index", 1) - 1
        service = InferenceService(worker_index=worker_index)
        await service.readyz()
        return service

    @lru_cache(maxsize=4096)
    @staticmethod
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
    @staticmethod
    async def _search_image_id_cache(image_id: int) -> tuple[float] | None:
        """Get cached image embeddings by ID.

        :param image_id: Database ID of the image.
        :type image_id: int
        :returns: Tuple of embedding values if found.
        :rtype: tuple[float] | None
        """
        try:
            async with AIOPostgres().session() as conn:
                # First get the embedding for the input image_id
                embedding_stmt = select(ORMGalleryEmbedding).where(ORMGalleryEmbedding.image_id == image_id)
                embedding_result = await conn.execute(embedding_stmt)
                embedding = embedding_result.scalar_one_or_none()
                return tuple(embedding.image_embedding) if embedding else None
        except Exception as e:
            logger.exception("Failed to search for image embeddings: {e}", e=e)
            return None

    async def global_exception_handler(self, request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for the API.

        :param request: The request object.
        :param exc: The exception raised.
        :return: A JSONResponse object with the error code and message.
        """
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "500",
                "request_info": request,
                "error_message": "Internal Server Error",
                "error_traceback": traceback.format_exc(limit=5),
                "exception": exc,
            },
        )

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
        modified_image_uuid: str | None = None,
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
        # Generate perceptual hash
        image_hash = self.hash_fn(image_pil).hash.tobytes()
        image_embed = (await self.embedding_service.embed_image([image_pil]))[0]

        duplicate_status = ImageDuplicateStatus.OK
        most_similar_image_uuid = None

        if metadata.get("ignore_duplicate_check", False) in (False, None):
            # Check for AI watermark
            ai_watermark = (await self.embedding_service.check_ai_watermark([image_pil]))[0]

            if ai_watermark:
                duplicate_status = ImageDuplicateStatus.PLAGIARIZED
            else:
                async with AIOPostgres().session() as conn:
                    # Find most similar image by perceptual hash
                    result = await conn.execute(
                        select(ORMImage.image_uuid)
                        .where(ORMImage.public.is_(True))
                        .order_by(
                            ORMImage.image_hash.hamming_similarity(image_hash).desc(),
                        )
                        .having(ORMImage.image_hash.hamming_similarity(image_hash) > self.hash_threshold)
                        .limit(1),
                    )

                    most_similar = result.scalar_one_or_none()
                    if most_similar:
                        duplicate_status = ImageDuplicateStatus.EXISTS
                        most_similar_image_uuid = most_similar

        if duplicate_status is not ImageDuplicateStatus.OK:
            await self._save_and_notify(
                duplicate_status=duplicate_status,
                most_similar_image_uuid=most_similar_image_uuid,
                image_pil=image_pil,
                image_model=ORMImage(
                    original_image_uuid=artwork_uuid,
                    generated_status=ai_generated_status,
                    image_metadata=metadata,
                    image_hash=image_hash,
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
                watermarked_image_pil = (await self.embedding_service.add_ai_watermark([image_pil], prompts=[generated_image_caption]))[0]

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

        metadata["ignore_duplicate_check"] = False

        orm_image_kwargs = {
            "original_image_uuid": artwork_uuid,
            "image_metadata": metadata,
            "image_hash": image_hash,
        }

        if modified_image_uuid:
            orm_image_kwargs["image_uuid"] = modified_image_uuid

        # Save to database and send to backend
        await self._save_and_notify(
            duplicate_status=duplicate_status,
            image_pil=watermarked_image_pil,
            image_model=ORMImage(**orm_image_kwargs),
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
        embedded_text: tuple[float, ...] = await self._get_and_cache_query_embeddings(
            query,
            self.embedding_service,
        )

        self.logger.debug("Got embeddings")

        try:
            async with AIOPostgres().session() as conn:
                stmt = (
                    select(ORMGalleryEmbedding.image_id)
                    .order_by(ORMGalleryEmbedding.image_embedding_distance_to(embedded_text))
                    .limit(count)
                    .offset(page * count)
                )

                result = await conn.execute(stmt)
                results = result.scalars().all()
        except Exception as e:
            self.logger.exception("Failed to search for images: {e}", e=e)
            results = []

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
        try:
            async with AIOPostgres().session() as conn:
                # First get the image ID from the UUID
                image_id_stmt = select(ORMImage.id).where(ORMImage.image_uuid == image_uuid_obj, ORMImage.public.is_(True))
                image_id_result = await conn.execute(image_id_stmt)
                image_id = image_id_result.scalar_one_or_none()

                if image_id is None:
                    self.ctx.response.status_code = 404
                    return ImageSearchResponse(image_uuid=[])

                # Get the embedding for this image ID
                embedding = await self._search_image_id_cache(image_id)
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
                results = result.scalars().all()
        except Exception as e:
            self.logger.exception("Failed to search for images: {e}", e=e)
            results = []

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

        try:
            async with AIOPostgres().session() as conn:
                stmt = (
                    select(ORMGalleryEmbedding.image_id)
                    .order_by(ORMGalleryEmbedding.image_embedding_distance_to(image_embed))
                    .limit(count)
                    .offset(page * count)
                )

                result = await conn.execute(stmt)
                results = result.scalars().all()
        except Exception as e:
            self.logger.exception("Failed to search for images: {e}", e=e)
            results = []

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

    @app.get(path="/image/exists")
    async def image_exists(self, image_uuid: str) -> Response:
        """Check if an image exists in the database."""
        async with AIOPostgres().session() as conn:
            result = await conn.execute(select(ORMImage.image_uuid).where(ORMImage.image_uuid == image_uuid))
        return Response(status_code=200 if result.scalar_one_or_none() is not None else 404)

    @app.get(path="/image/needs-reprocessing")
    async def image_needs_reprocessing(self, image_uuid: str) -> Response:
        """Check if an image needs reprocessing.

        :param image_uuid: UUID of the image to check.
        :type image_uuid: str
        :returns: Response with 200 if image needs reprocessing (hash is null), 404 if not found.
        :rtype: Response
        """
        async with AIOPostgres().session() as conn:
            result = await conn.execute(
                select(ORMImage.image_uuid)
                .where(
                    ORMImage.image_uuid == image_uuid,
                    ORMImage.image_hash.is_(None),
                ),
            )
        return Response(status_code=200 if result.scalar_one_or_none() is not None else 404)

    @bentoml.api(
        route="/image/reprocess",
    )
    async def reprocess_image(
        self,
        image: PILImage,
        image_uuid: str,
        metadata: dict[str, Any],
        modified_image_uuid: str,
        ctx: bentoml.Context,
    ) -> None:
        """Reprocess an image for storage and analysis.

        :param image: Image data to process.
        :type image: PILImage
        :param image_uuid: UUID for the image.
        :type image_uuid: str
        :param modified_image_uuid: UUID for the modified image.
        :type modified_image_uuid: str
        :param ctx: BentoML request context.
        :type ctx: bentoml.Context
        :raises ValueError: If JWT token is invalid.
        """
        self._verify_jwt(ctx)

        image_pil = image.convert("RGB")
        if ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            ctx.response.status_code = 503
            return

        task = asyncio.create_task(self._process_image_data(
            image_pil=image_pil,
            artwork_uuid=image_uuid,
            ai_generated_status=AIGeneratedStatus.GENERATED_PROTECTED,
            metadata=metadata,
            modified_image_uuid=modified_image_uuid,
        ))

        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        self.ctx.state["queued_processing"] = self.ctx.state.get("queued_processing", 0) + 1
        ctx.response.status_code = 201

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

    @app.get(path="/healthz")
    async def healthz(self) -> JSONResponse:
        """Check service health status.

        :returns: Health status response.
        :rtype: JSONResponse
        """
        if settings.debug:
            return JSONResponse(status_code=200, content={"status": "healthy"})

        if not self.db_healthy:
            return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "database not healthy"})

        return JSONResponse(status_code=200, content={"status": "healthy"})

    @app.get(path="/readyz")
    async def readyz(self) -> JSONResponse:
        """Check if service is ready to handle requests.

        :returns: Readiness status response.
        :rtype: JSONResponse
        """
        if self.ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "queue full"})

        if not self.db_healthy:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "database not ready"})

        if self.embedding_service is None: # type: ignore[]
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "embedding service not ready"})

        try:
            await asyncio.wait_for(self.embedding_service.readyz(), timeout=30.0)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "embedding service timeout"})

        return JSONResponse(status_code=200, content={"status": "ready"})

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
    )
    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text inputs.

        :param texts: List of text strings to embed.
        :type texts: list[str]
        :returns: List of normalized embedding vectors.
        :rtype: list[list[float]]
        """
        return await self.embedding_service.embed_text(texts)

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def embed_image(self, images: list[PILImage]) -> list[list[float]]:
        """Generate embeddings for a list of images.

        :param images: List of PIL images to embed.
        :type images: List[PILImage]
        :returns: List of normalized embedding vectors.
        :rtype: list[list[float]]
        """
        return await self.embedding_service.embed_image(images)

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def generate_caption(self, images: list[PILImage]) -> list[str]:
        """Generate descriptive captions for a list of images.

        :param images: List of PIL images to generate captions for.
        :type images: List[PILImage]
        :returns: List of generated caption strings.
        :rtype: list[str]
        """
        return await self.embedding_service.generate_caption(images)

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def detect_ai_generation(self, images: list[PILImage]) -> list[AIGeneratedStatus]:
        """Detect whether images were generated by AI.

        :param images: List of PIL images to check.
        :type images: List[PILImage]
        :returns: List of AI generation status enums.
        :rtype: list[AIGeneratedStatus]
        """
        return await self.embedding_service.detect_ai_generation(images)

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.fast_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.fast_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
    )
    async def check_ai_watermark(self, images: list[PILImage]) -> list[bool]:
        """Check for the presence of AI-specific watermarks in images.

        :param images: List of PIL images to check for AI watermarks.
        :type images: list[PILImage]
        :returns: List of boolean values indicating AI watermark presence.
        :rtype: list[bool]
        """
        return await self.embedding_service.check_ai_watermark(images)

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=settings.bentoml.inference.slow_batched_op_max_batch_size,
        max_latency_ms=settings.bentoml.inference.slow_batched_op_max_latency_ms,
        input_spec=EmbedRequest,
        output_spec=EmbedRequest,
    )
    async def add_ai_watermark(self, images: list[PILImage], prompts: list[str]) -> list[PILImage]:
        """Add AI-specific watermarks to a list of images.

        :param images: List of PIL images to add AI watermarks to.
        :type images: List[PILImage]
        :returns: List of watermarked PIL images.
        :rtype: list[PILImage]
        """
        return await self.embedding_service.add_ai_watermark(images, prompts=prompts)
