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
import traceback
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import aiohttp
import bentoml
import jwt
import tenacity
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
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

if TYPE_CHECKING:
    from collections.abc import Callable

app = FastAPI()
logger = get_logger()


T = TypeVar("T")
P = ParamSpec("P")


def async_lru_cache(maxsize: int = 128) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Create an LRU cache decorator for async functions.

    :param maxsize: Maximum size of the cache.
    :type maxsize: int
    :returns: Decorator function.
    """
    cache = {}

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = str(args) + str(kwargs)
            if key in cache:
                return cache[key]

            result = await func(*args, **kwargs)  # type: ignore[arg-type]

            if len(cache) >= maxsize:
                # Remove oldest item (first item in dict)
                cache.pop(next(iter(cache)))

            cache[key] = result
            return result

        return wrapper  # type: ignore[return-value]

    return decorator

@async_lru_cache(maxsize=4096)
async def _get_query_embeddings(query: str, embedding_service: InferenceServiceProto) -> tuple[float, ...]:
    """Get and cache text query embeddings.

    :param query: Text query to embed.
    :param embedding_service: Service to generate embeddings.
    :returns: Tuple of embedding values.
    """
    return tuple((await embedding_service.embed_text([query]))[0])

@async_lru_cache(maxsize=4096)
async def _get_image_embedding(image_id: int) -> tuple[float] | None:
    """Get cached image embeddings by ID.

    :param image_id: Database ID of the image.
    :returns: Tuple of embedding values if found.
    """
    try:
        async with AIOPostgres().session() as conn:
            logger.debug("Executing embedding query for image ID: {id}", id=image_id)
            embedding_stmt = select(ORMGalleryEmbedding).where(
                ORMGalleryEmbedding.image_id == image_id,
            )
            embedding_result = await conn.execute(embedding_stmt)
            embedding = embedding_result.scalar_one_or_none()

            if embedding is None:
                logger.debug("No embedding found for image ID: {id}", id=image_id)
                return None

            logger.debug("Found embedding for image ID: {id}", id=image_id)
            return tuple(embedding.image_embedding)
    except Exception as e:
        logger.exception("Failed to search for image embeddings: {e}", e=e)
        return None


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
        background_tasks: Set of running background tasks.
        jwt_secret: Secret for JWT token validation.

    """

    def __init__(self) -> None:
        """Initialize the API service with required dependencies."""
        self.logger = logger
        self.healthy = True
        self.db_ready = False
        self.ctx = bentoml.Context()
        self.ctx.state["queued_processing"] = 0
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.embedding_service: InferenceServiceProto = None  # type: ignore[]

        app.add_exception_handler(exc_class_or_status_code=Exception, handler=self.global_exception_handler)

        # Get the running event loop from BentoML's worker
        loop = asyncio.get_event_loop()

        # Start tasks in the event loop without waiting
        async def start_tasks() -> None:
            embedding_task = asyncio.create_task(self._init_embedding_service())
            migration_task = asyncio.create_task(self._init_postgres())

            self.logger.info("Started model and migration tasks")

            self.background_tasks.add(migration_task)
            self.background_tasks.add(embedding_task)

            embedding_task.add_done_callback(self._set_embedding_service)
            embedding_task.add_done_callback(self.global_task_resolver)
            migration_task.add_done_callback(self._check_migration_success)
            migration_task.add_done_callback(self.global_task_resolver)

        asyncio.run_coroutine_threadsafe(start_tasks(), loop)

    def _set_embedding_service(self, task: asyncio.Task[InferenceServiceProto]) -> None:
        """Set embedding service from task result.

        :param task: Completed embedding service initialization task.
        """
        try:
            self.embedding_service = task.result()
        except Exception as e:
            self.logger.exception("Embedding service initialization failed: {e}", e=e)

    def _check_migration_success(self, task: asyncio.Task[Any]) -> None:
        if task.done() and task.exception() is not None:
            self.healthy = False
            raise task.exception()  # type: ignore[BaseException]
        self.db_ready = True

    async def _init_postgres(self) -> None:
        """Initialize the Postgres database connection."""
        try:
            self.logger.info("Initializing Postgres client")
            AIOPostgres(url=settings.postgres.url)
            self.logger.info("Postgres client initialized. Running migrations.")
            self._migrate_database()
            self.logger.info("Migrations completed successfully.")
        except Exception as e:
            self.logger.exception("Failed to initialize Postgres client or run migrations: {e}", e=e)
            raise

    def global_task_resolver(self, task: asyncio.Task[Any]) -> None:
        """Resolve all tasks and handle exceptions.

        :raises asyncio.TimeoutError: If tasks take too long to complete.
        :raises Exception: If any task fails during execution.
        """
        try:
            if task.done():
                if task.exception() is not None:
                    self.logger.exception("Task failed: {task}", task=task)
                return
            task.cancel()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            self.background_tasks.discard(task)

    def decrement_queued_processing(self, _: asyncio.Task[None]) -> None:
        """Decrement the queued processing state."""
        self.logger.debug("Decrementing queued processing")
        self.ctx.state["queued_processing"] -= 1
        self.logger.debug("Queued processing: {queued_processing}", queued_processing=self.ctx.state["queued_processing"])

    async def _init_embedding_service(self) -> InferenceServiceProto:
        """Initialize the embedding service asynchronously.

        :returns: Initialized inference service.
        :rtype: InferenceServiceProto
        """
        worker_index = getattr(bentoml.server_context, "worker_index", 1) - 1
        service = InferenceService(worker_index=worker_index)
        await service.readyz()
        return service

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

    @staticmethod
    def _verify_jwt(request: Request | None = None, ctx: bentoml.Context | None = None) -> None:
        """Verify JWT token from request Authorization header.

        :param request: HTTP request.
        :type request: Request
        :raises ValueError: If token is missing or invalid.
        """
        if not settings.verify_jwt:
            return

        auth_header = (ctx.request.headers.get("Authorization") if ctx else "") or (request.headers.get("Authorization") if request else "")
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.exception("Missing auth header in jwt verification.")
            if ctx:
                ctx.response.status_code = 401
            raise HTTPException(status_code=401, detail="Unauthorized")

        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        except InvalidTokenError as e:
            logger.exception("Invalid JWT token.")
            if ctx:
                ctx.response.status_code = 401
            raise HTTPException(status_code=401, detail="Unauthorized") from e

    def _migrate_database(self) -> None:
        """Run database migrations using Alembic.

        This method is skipped in CI environments.
        """
        if settings.debug:
            return

        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")

    async def _save_and_notify(  # noqa: C901, PLR0915
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
            "Authorization": f"Bearer {jwt.encode({'sub': 'ai-service'}, settings.jwt_secret, algorithm='HS256')}",
        }

        self.logger.debug("Sending PATCH request to backend for {image_uuid}", image_uuid=image_model.image_uuid)

        if duplicate_status is not ImageDuplicateStatus.OK:
            self.logger.debug("Duplicate status for {image_uuid}: {duplicate_status}", image_uuid=image_model.image_uuid, duplicate_status=duplicate_status)
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

            try:
                async with aiohttp.ClientSession() as session, session.patch(
                    settings.eva_backend.backend_image_patch_route,
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status not in (200, 201):
                        self.logger.error("Failed to patch backend: {response}", response=await response.text())
            except asyncio.TimeoutError:
                self.logger.exception("Request to backend timed out for {image_uuid}", image_uuid=image_model.image_uuid)
            except aiohttp.ClientError as e:
                self.logger.exception("Failed to connect to backend for {image_uuid}: {e}", image_uuid=image_model.image_uuid, e=e)
            except Exception as e:
                self.logger.exception("Unexpected error while patching backend for {image_uuid}: {e}", image_uuid=image_model.image_uuid, e=e)
            return

        self.logger.debug("Saving image data for {image_uuid}", image_uuid=image_model.image_uuid)

        try:
            async with AIOPostgres().session() as conn:
                # Add and flush image data model
                conn.add(image_model)
                await conn.flush()

                # Set image_id on gallery embedding and add
                gallery_embedding_model.image_id = image_model.id
                conn.add(gallery_embedding_model)
                await conn.commit()

                self.logger.debug("Saved image data for {image_uuid}", image_uuid=image_model.image_uuid)
        except Exception as e:
            self.logger.exception("Failed to save image data for {image_uuid}: {e}", image_uuid=image_model.image_uuid, e=e)
            raise

        # Convert PIL image to bytes for storage
        self.logger.debug("Converting PIL image to bytes for {image_uuid}", image_uuid=image_model.image_uuid)
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format=image_pil.format or "PNG")
        image_bytes.seek(0)

        self.logger.debug("Converted PIL image to bytes for {image_uuid}", image_uuid=image_model.image_uuid)

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

        self.logger.debug("Sending PATCH request to backend for {image_uuid}", image_uuid=image_model.image_uuid)

        try:
            async with aiohttp.ClientSession() as session, session.patch(
                settings.eva_backend.backend_image_patch_route,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status not in (200, 201):
                    self.logger.error("Failed to patch backend: {response}", response=await response.text())
        except asyncio.TimeoutError:
            self.logger.exception("Request to backend timed out for {image_uuid}", image_uuid=image_model.image_uuid)
        except aiohttp.ClientError as e:
            self.logger.exception("Failed to connect to backend for {image_uuid}: {e}", image_uuid=image_model.image_uuid, e=e)
        except Exception as e:
            self.logger.exception("Unexpected error while patching backend for {image_uuid}: {e}", image_uuid=image_model.image_uuid, e=e)

        self.logger.debug("All done")

    async def _process_image_data(  # noqa: PLR0915
        self,
        image_pil: PILImage,
        artwork_uuid: str,
        ai_generated_status: AIGeneratedStatus,
        metadata: dict[str, Any],
        modified_image_uuid: str | None = None,
    ) -> None:
        """Process image data including embedding and watermarking.

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
        image_embed = (await self.embedding_service.embed_image([image_pil]))[0]
        self.logger.debug("Type of image_embed: {type}", type=type(image_embed))

        self.logger.debug("Generated image embedding for {image_uuid}", image_uuid=artwork_uuid)

        duplicate_status = ImageDuplicateStatus.OK
        most_similar_image_uuid = None

        if metadata.get("ignore_duplicate_check", False) in (False, None):
            # Check for AI watermark
            self.logger.debug("Checking for AI watermark for {image_uuid}", image_uuid=artwork_uuid)
            ai_watermark = (await self.embedding_service.check_ai_watermark([image_pil]))[0]
            self.logger.debug("AI watermark check result for {image_uuid}: {ai_watermark}", image_uuid=artwork_uuid, ai_watermark=ai_watermark)
            if ai_watermark:
                duplicate_status = ImageDuplicateStatus.PLAGIARIZED
            else:
                # Check for duplicate via embedding similarity
                self.logger.debug("Checking for duplicates via embedding similarity for {image_uuid}", image_uuid=artwork_uuid)
                async with AIOPostgres().session() as conn:
                    # Find most similar image by embedding distance
                    distance_col = ORMGalleryEmbedding.image_embedding_distance_to(image_embed).label("distance")
                    stmt = (
                        select(ORMImage.image_uuid, distance_col)
                        .join(ORMGalleryEmbedding, ORMGalleryEmbedding.image_id == ORMImage.id)
                        .where(
                            distance_col >= settings.model.similarity_threshold,
                        )
                        .order_by(distance_col.desc())  # DESC because higher values are more similar
                        .limit(1)
                    )
                    result = await conn.execute(stmt)
                    row = result.first()
                    if row:
                        most_similar = row.image_uuid
                        distance = row.distance
                        self.logger.debug(
                            "Found similar image for {image_uuid}: {most_similar} with distance {distance}",
                            image_uuid=artwork_uuid,
                            most_similar=most_similar,
                            distance=distance,
                        )
                        duplicate_status = ImageDuplicateStatus.EXISTS
                        most_similar_image_uuid = most_similar

        if duplicate_status is not ImageDuplicateStatus.OK:
            self.logger.debug("Duplicate status for {image_uuid}: {duplicate_status} in _process_image_data", image_uuid=artwork_uuid, duplicate_status=duplicate_status)
            self.logger.debug("Saving and notifying for {image_uuid}", image_uuid=artwork_uuid)
            await self._save_and_notify(
                duplicate_status=duplicate_status,
                most_similar_image_uuid=most_similar_image_uuid,
                image_pil=image_pil,
                image_model=ORMImage(
                    original_image_uuid=artwork_uuid,
                    image_uuid=artwork_uuid,
                    generated_status=ai_generated_status,
                    image_metadata=metadata,
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
        self.logger.debug("Getting embeddings and caption for {image_uuid}", image_uuid=artwork_uuid)
        generated_image_caption, metadata_embedding = await asyncio.gather(
            self.embedding_service.generate_caption([image_pil]),
            self.embedding_service.embed_text([str(metadata)]),
        )

        generated_image_caption = generated_image_caption[0]
        metadata_embedding = metadata_embedding[0]
        self.logger.debug("Generated image caption for {image_uuid}: {caption}", image_uuid=artwork_uuid, caption=generated_image_caption)

        # Detect if AI generated
        watermarked_image_pil = image_pil
        watermarked_image_embed = None
        if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
            self.logger.debug("Detecting AI generation for {image_uuid}", image_uuid=artwork_uuid)
            ai_generated_status = (await self.embedding_service.detect_ai_generation([image_pil]))[0]
            if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
                self.logger.debug("Adding AI watermark for {image_uuid}", image_uuid=artwork_uuid)
                watermarked_image_pil = (await self.embedding_service.add_ai_watermark([image_pil], prompts=[generated_image_caption]))[0]

        self.logger.debug("Embedding watermarked image for {image_uuid}", image_uuid=artwork_uuid)
        watermarked_image_embed, generated_caption_embed = await asyncio.gather(
            self.embedding_service.embed_image([watermarked_image_pil]),
            self.embedding_service.embed_text([generated_image_caption]),
        )
        watermarked_image_embed = watermarked_image_embed[0]
        generated_caption_embed = generated_caption_embed[0]

        self.logger.debug("Embedding generated caption for {image_uuid}", image_uuid=artwork_uuid)

        user_caption = metadata.get("caption")
        self.logger.debug("User caption for {image_uuid}: {user_caption}", image_uuid=artwork_uuid, user_caption=user_caption)
        user_caption_embed = None
        if user_caption:
            self.logger.debug("Embedding user caption for {image_uuid}", image_uuid=artwork_uuid)
            user_caption_embed = (await self.embedding_service.embed_text([user_caption]))[0]
            self.logger.debug("Embedded user caption for {image_uuid}: {user_caption_embed}", image_uuid=artwork_uuid, user_caption_embed=str(user_caption_embed)[:100])

        metadata["ai_generated"] = ai_generated_status.value
        metadata["duplicate_status"] = duplicate_status.value

        self.logger.debug("Metadata for {image_uuid}: {metadata}", image_uuid=artwork_uuid, metadata=metadata)

        self.logger.debug("Removed from queued_processing")

        metadata["ignore_duplicate_check"] = False

        orm_image_kwargs: dict[str, Any] = {
            "original_image_uuid": artwork_uuid,
            "image_metadata": metadata,
        }

        if modified_image_uuid:
            orm_image_kwargs["image_uuid"] = modified_image_uuid
        else:
            orm_image_kwargs["image_uuid"] = str(uuid.uuid4())

        orm_image_kwargs["generated_status"] = ai_generated_status.value
        orm_image_kwargs["public"] = False
        orm_image_kwargs["user_annotation"] = user_caption
        orm_image_kwargs["generated_annotation"] = generated_image_caption

        # Save to database and send to backend
        self.logger.debug("Saving and notifying for {image_uuid}", image_uuid=artwork_uuid)
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
            most_similar_image_uuid=most_similar_image_uuid,
        )

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=1, max=5))
    async def _get_search_query_results(self, embedded_text: tuple[float, ...], count: int = 50, page: int = 0) -> SearchResponse:
        async with AIOPostgres().session() as conn:
            stmt = (
                select(ORMImage.image_uuid)
                .join(ORMGalleryEmbedding, ORMGalleryEmbedding.image_id == ORMImage.id)
                .where(ORMImage.public.is_(True))
                .order_by(ORMGalleryEmbedding.image_embedding_distance_to(embedded_text).desc())
                .limit(count)
                .offset(page * count)
            )

            self.logger.debug("Executing search query: {stmt}", stmt=str(stmt))

            result = await conn.execute(stmt)
            results = result.scalars().all()
            self.logger.debug("Found {count} results", count=len(results))

        return SearchResponse(image_uuid=list(results))

    @app.get(path="/image/search_query", response_model=None, dependencies=[Depends(_verify_jwt)])
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
        embedded_text: tuple[float, ...] = await _get_query_embeddings(
            query,
            self.embedding_service,
        )

        self.logger.debug("Got embeddings")

        try:
            return await self._get_search_query_results(embedded_text, count, page)
        except Exception as e:
            self.logger.exception("Failed to search for images: {e}", e=e)
            return SearchResponse(image_uuid=[])

    @app.get(path="/image/search_image", response_model=None, dependencies=[Depends(_verify_jwt)])
    async def search_image(self, image_uuid: str, count: int = 50, page: int = 0) -> ImageSearchResponse:
        """Search for similar images using an image UUID.

        :param image_uuid: UUID of the reference image which must be the modified image UUID.
        :type image_uuid: str
        :param count: Number of results per page.
        :type count: int
        :param page: Page number (0-based).
        :type page: int
        :returns: Search response containing similar image UUIDs.
        :rtype: ImageSearchResponse
        :raises ValueError: If JWT token is invalid.
        """
        self.logger.info("Starting image search for UUID: {uuid} with count={count}, page={page}", uuid=image_uuid, count=count, page=page)

        try:
            async with AIOPostgres().session() as conn:
                self.logger.debug("Database session established")

                # First get the image ID from the UUID
                image_id_stmt = select(ORMImage.id).where(ORMImage.image_uuid == image_uuid, ORMImage.public.is_(True))
                self.logger.debug("Executing image ID query: {stmt}", stmt=str(image_id_stmt))

                image_id_result = await conn.execute(image_id_stmt)
                image_id = image_id_result.scalar_one_or_none()
                self.logger.debug("Image ID query result: {id}", id=image_id)

                if image_id is None:
                    self.logger.info("Image not found or not public: {uuid}", uuid=image_uuid)
                    self.ctx.response.status_code = 404
                    return ImageSearchResponse(image_uuid=[])

                # Get the embedding for this image ID
                self.logger.debug("Fetching embedding from cache for image ID: {id}", id=image_id)
                embedding = await _get_image_embedding(image_id)

                if embedding is None:
                    self.logger.info("No embedding found for image ID: {id}", id=image_id)
                    self.ctx.response.status_code = 404
                    return ImageSearchResponse(image_uuid=[])

                self.logger.debug("Got embedding, length: {len}", len=len(embedding))

                # Get similar images, excluding the input image
                stmt = (
                    select(ORMImage.image_uuid)
                    .join(ORMGalleryEmbedding, ORMGalleryEmbedding.image_id == ORMImage.id)
                    .where(ORMImage.id != image_id)
                    .order_by(ORMGalleryEmbedding.image_embedding_distance_to(embedding).desc())
                    .limit(count)
                    .offset(page * count)
                )

                self.logger.debug("Executing similarity search query: {stmt}", stmt=str(stmt))
                result = await conn.execute(stmt)
                self.logger.debug("Similarity search query executed")

                results = result.scalars().all()
                self.logger.debug("Found {count} similar images", count=len(results))

        except Exception as e:
            self.logger.exception("Failed to search for images: {e}", e=e)
            results = []

        self.logger.info("Completed image search for UUID: {uuid}, found {count} results", uuid=image_uuid, count=len(results))
        return ImageSearchResponse(image_uuid=list(results))

    @app.get(path="/image/search_image_raw", response_model=None, dependencies=[Depends(_verify_jwt)])
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

    @app.patch(path="/image/set-public", response_model=None, dependencies=[Depends(_verify_jwt)])  # type: ignore[misc]
    async def publish_image(self, image_uuid: str | list[str]) -> JSONResponse:
        """Publish one or multiple images by setting their public flag to True.

        :param image_uuid: Single image UUID or list of image UUIDs to publish.
        :type image_uuid: str | list[str]
        :raises ValueError: If JWT token is invalid.
        :raises SQLAlchemyError: If the database update fails.
        :returns: JSONResponse with status code and error message if any.
        :rtype: JSONResponse
        """
        if isinstance(image_uuid, str):
            image_uuid = [image_uuid]

        self.logger.debug("Publishing images: {image_uuid}", image_uuid=image_uuid)

        try:
            async with AIOPostgres().session() as conn:
                # Perform the update and return number of rows updated
                stmt = (
                    update(ORMImage)
                    .where(ORMImage.image_uuid.in_(image_uuid))
                    .values(public=True)
                )
                result = await conn.execute(stmt)

                if result.rowcount == 0:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": "No images found to update",
                            "missing_uuids": image_uuid,
                        },
                    )

                if result.rowcount != len(image_uuid):
                    self.logger.error(
                        "Expected to update {img_count} rows, but updated {row_count}",
                        img_count=len(image_uuid),
                        row_count=result.rowcount,
                    )
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": "Some images not found",
                            "updated_count": result.rowcount,
                            "requested_count": len(image_uuid),
                        },
                    )

                await conn.commit()
                self.logger.debug("Committed transaction")

        except SQLAlchemyError as e:
            self.logger.exception("Failed to publish images: {e}", e=e)
            return JSONResponse(
                status_code=500,
                content={"error": "Database error occurred while publishing images"},
            )

        return JSONResponse(status_code=200, content={})

    @app.get(path="/image/exists", dependencies=[Depends(_verify_jwt)])
    async def image_exists(self, image_uuid: str) -> Response:
        """Check if an image exists in the database."""
        async with AIOPostgres().session() as conn:
            result = await conn.execute(select(ORMImage.image_uuid).where(ORMImage.image_uuid == image_uuid))
        return Response(status_code=200 if result.scalar_one_or_none() is not None else 404)

    @app.get(path="/image/check-public", dependencies=[Depends(_verify_jwt)])
    async def check_public(self, image_uuid: str) -> Response:
        """Check if an image is marked as public in the database.

        :param image_uuid: UUID of the image to check.
        :type image_uuid: str
        :returns: Response with 200 if image is public, 404 if not found or not public.
        :rtype: Response
        """
        async with AIOPostgres().session() as conn:
            result = await conn.execute(
                select(ORMImage.image_uuid)
                .where(
                    ORMImage.image_uuid == image_uuid,
                    ORMImage.public.is_(True),
                ),
            )
        return Response(status_code=200 if result.scalar_one_or_none() is not None else 404)

    @app.get(path="/image/needs-reprocessing", dependencies=[Depends(_verify_jwt)])
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
        self._verify_jwt(request=None, ctx=ctx)

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

        self.ctx.state["queued_processing"] += 1

        self.background_tasks.add(task)
        task.add_done_callback(self.global_task_resolver)
        task.add_done_callback(self.decrement_queued_processing)

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
        self._verify_jwt(request=None, ctx=ctx)
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
        self.logger.debug("Added image processing task to background tasks for {image_uuid}", image_uuid=image_uuid)

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

        if not self.healthy:
            return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "migration not complete or couldn't connect to database"})

        try:
            async with AIOPostgres().session() as conn:
                await conn.execute(select(1))
        except SQLAlchemyError:
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

        if not self.db_ready:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "database not ready, migrations didn't complete yet"})

        try:
            async with AIOPostgres().session():
                pass
        except SQLAlchemyError:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "database not ready"})

        if self.embedding_service is None: # type: ignore[]
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": "embedding service not ready"})

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
