from __future__ import annotations

import asyncio
import hashlib
import io
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import aiohttp
import bentoml
import numpy as np
from bentoml.io import JSON, File, Multipart
from PIL import Image as PILImage
from sqlalchemy import select

from . import settings
from .database import AIOPostgres
from .model import (
    AIGeneratedStatus,
    BackendPatchRequest,
    ImageDuplicateStatus,
    ImageSearchRequest,
    ImageSearchResponse,
    ProcessImageRequest,
    RawImageSearchRequest,
    SearchRequest,
    SearchResponse,
)
from .orm import GalleryEmbedding, Image
from .services import AddWatermarkRequest, InferenceService, InferenceServiceProto
from .util import get_logger

if TYPE_CHECKING:
    from uuid import UUID


@bentoml.service(
    name="evagallery_ai_api",
    **settings.bentoml.service.api.to_dict(),
)
class APIService:
    def __init__(self) -> None:
        self.embedding_service: InferenceServiceProto = cast(InferenceServiceProto, bentoml.depends(InferenceService))
        self.logger = get_logger()
        self.ctx = bentoml.Context()
        self.db_healthy = False
        self.background_tasks: set[asyncio.Task[Any]] = set()
        asyncio.run_coroutine_threadsafe(self._init_postgres(), asyncio.get_event_loop())

    async def _init_postgres(self) -> None:
        try:
            AIOPostgres(url=settings.postgres.url)
            self._migrate_database()
        except Exception as e:
            self.logger.exception(f"Failed to initialize Postgres client: {e}")
            raise
        self.db_healthy = True

    def _migrate_database(self) -> None:
        """Run alembic migrations to head revision."""
        if os.getenv("CI"):
            return
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")

    async def _save_and_notify(
        self,
        image_pil: PILImage.Image,
        duplicate_status: ImageDuplicateStatus,
        image_model: Image,
        gallery_embedding_model: GalleryEmbedding,
        most_similar_image_uuid: UUID | None = None,
    ) -> None:
        if duplicate_status is not ImageDuplicateStatus.OK:
            patch_request = BackendPatchRequest(
                image_uuid=image_model.image_uuid,
                closest_match_uuid=most_similar_image_uuid,
                image_duplicate_status=duplicate_status,
                modified_image_uuid=None,
                image=None,
                ai_generated_status=image_model.generated_status,
                metadata=None,
            )

            async with aiohttp.ClientSession() as session, session.patch(
                settings.eva_backend.backend_image_patch_route,
                json=patch_request.model_dump_json(),
            ) as response:
                if response.status not in (200, 201):
                    self.logger.error(f"Failed to patch backend: {await response.text()}")

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
        image_bytes = image_bytes.getvalue()

        # Send PATCH request to backend with full model
        patch_request = BackendPatchRequest(
            image_uuid=image_model.original_image_uuid,
            image_duplicate_status=ImageDuplicateStatus.OK,
            closest_match_uuid=None,
            modified_image_uuid=image_model.image_uuid,
            image=image_bytes,
            ai_generated_status=image_model.generated_status,
            metadata=image_model.image_metadata,
        )

        async with aiohttp.ClientSession() as session, session.patch(
            settings.eva_backend.backend_image_patch_route,
            json=patch_request.model_dump_json(),
        ) as response:
            if response.status not in (200, 201):
                self.logger.error(f"Failed to patch backend: {await response.text()}")

    async def _process_image_data(
        self,
        image_pil: PILImage.Image,
        artwork_uuid: str,
        ai_generated_status: AIGeneratedStatus,
        metadata: dict[str, Any],
    ) -> None:
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
                        select(Image)
                        .filter_by(image_hash=watermark),  # Changed from md5_hash to image_hash
                    )
                    if result.scalar_one_or_none():
                        result = await conn.execute(
                            select(GalleryEmbedding.image_embedding, GalleryEmbedding.image_id)
                            .order_by(GalleryEmbedding.image_embedding_distance_to(image_embed)),
                        )

                        float_vec, image_id = (await (await result.scalars()).one())  # type: ignore

                        result = await conn.execute(
                            select(Image.image_uuid)
                            .where(Image.id == image_id),
                        )

                        most_similar_image_uuid = await (await result.scalars()).one()  # type: ignore

                        # check if dot product is more than 95% similar
                        if np.dot(image_embed, float_vec) > settings.model.detection.threshold:
                            duplicate_status = ImageDuplicateStatus.EXISTS

        if duplicate_status is not ImageDuplicateStatus.OK:
            await self._save_and_notify(
                duplicate_status=duplicate_status,
                most_similar_image_uuid=most_similar_image_uuid,
                image_pil=image_pil,
                image_model=Image(
                    original_image_uuid=artwork_uuid,
                    generated_status=ai_generated_status,
                    image_metadata=metadata,
                    image_hash=original_hash,
                ),
                gallery_embedding_model=GalleryEmbedding(
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

        watermarked_image_pil: PILImage.Image = (await self.embedding_service.add_watermark([
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
            image_model=Image(
                original_image_id=artwork_uuid,
                image_metadata=metadata,
                image_hash=original_hash,
            ),
            gallery_embedding_model=GalleryEmbedding(
                image_embedding=image_embed,
                watermarked_image_embedding=watermarked_image_embed,
                metadata_embedding=metadata_embedding,
                user_caption_embedding=user_caption_embed,
                generated_caption_embedding=generated_caption_embed,
            ),
        )

    @lru_cache(maxsize=4096)
    @staticmethod
    async def _get_and_cache_query_embeddings(query: str, embedding_service: InferenceServiceProto) -> tuple[float, ...]:
        return tuple((await embedding_service.embed_text([query]))[0])

    @bentoml.api(
        route="/image/search_query",
        input_spec=JSON(pydantic_model=SearchRequest),  # type: ignore
        output_spec=JSON(pydantic_model=SearchResponse),  # type: ignore
    )
    async def search_query(self, query_request: SearchRequest) -> SearchResponse:
        embedded_text: tuple[float, ...] = await self._get_and_cache_query_embeddings(
            query_request.query,
            self.embedding_service,
        )

        async with AIOPostgres() as conn:
            stmt = (
                select(GalleryEmbedding.image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(embedded_text))
                .limit(query_request.count)
                .offset(query_request.page * query_request.count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore

        return SearchResponse(image_id=list(results))

    @lru_cache(maxsize=4096)
    @staticmethod
    async def _search_image_id_cache(image_id: int) -> tuple[float] | None:
        async with AIOPostgres() as conn:
            # First get the embedding for the input image_id
            embedding_stmt = select(GalleryEmbedding).where(GalleryEmbedding.image_id == image_id)
            embedding_result = await conn.execute(embedding_stmt)
            embedding = await embedding_result.scalar_one_or_none()  # type: ignore
            return tuple(embedding.image_embedding) if embedding else None

    @bentoml.api(
        route="/image/search_image",
        input_spec=JSON(pydantic_model=ImageSearchRequest),  # type: ignore
        output_spec=JSON(pydantic_model=ImageSearchResponse),  # type: ignore
    )
    async def search_image(self, search_request: ImageSearchRequest) -> ImageSearchResponse:
        async with AIOPostgres() as conn:
            # First get the image ID from the UUID
            image_id_stmt = select(Image.id).where(Image.image_uuid == search_request.image_uuid)
            image_id_result = await conn.execute(image_id_stmt)
            image_id = await image_id_result.scalar_one_or_none()  # type: ignore

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
                select(Image.image_uuid)
                .join(GalleryEmbedding, GalleryEmbedding.image_id == Image.id)
                .where(Image.id != image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(embedding))
                .limit(search_request.count)
                .offset(search_request.page * search_request.count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore

        return ImageSearchResponse(image_uuid=list(results))

    @bentoml.api(
        route="/image/search_image_raw",
        input_spec=Multipart(
            image=File(),
            request=JSON(pydantic_model=RawImageSearchRequest),  # type: ignore
        ),
        output_spec=JSON(pydantic_model=ImageSearchResponse),  # type: ignore
    )
    async def search_image_raw(
        self,
        image: bytes,
        request: RawImageSearchRequest,
        ctx: bentoml.Context,
    ) -> ImageSearchResponse:
        if ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            ctx.response.status_code = 503
            return ImageSearchResponse(image_uuid=[])  # type: ignore

        ctx.state["queued_processing"] = ctx.state.get("queued_processing", 0) + 1

        image_pil = PILImage.open(io.BytesIO(image)).convert("RGB")
        image_embed = (await self.embedding_service.embed_image([image_pil]))[0]

        ctx.state["queued_processing"] -= 1

        async with AIOPostgres() as conn:
            stmt = (
                select(GalleryEmbedding.image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(image_embed))
                .limit(request.count)
                .offset(request.page * request.count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore

        return ImageSearchResponse(image_uuid=list(results))  # type: ignore

    @bentoml.api(
        route="/image/process",
        input_spec=Multipart(
            image=File(),
            request=JSON(pydantic_model=ProcessImageRequest),  # type: ignore
        ),
    )
    async def process_image(
        self,
        image: bytes,
        request: ProcessImageRequest,
        ctx: bentoml.Context,
    ) -> None:
        if ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            ctx.response.status_code = 503
            return

        image_pil = PILImage.open(io.BytesIO(image)).convert("RGB")

        task = asyncio.create_task(self._process_image_data(
            image_pil=image_pil,
            artwork_uuid=str(request.image_uuid),
            ai_generated_status=request.ai_generated_status,
            metadata=request.metadata or {},
        ))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        ctx.state["queued_processing"] = ctx.state.get("queued_processing", 0) + 1
        ctx.response.status_code = 201

    @bentoml.api(route="/healthz")
    async def healthz(self, ctx: bentoml.Context) -> dict[str, str]:
        """Check whether the service is healthy."""
        if not self.db_healthy:
            ctx.response.status_code = 503
            return {"status": "unhealthy"}

        ctx.response.status_code = 200
        return {"status": "healthy"}

    @bentoml.api(route="/readyz")
    async def readyz(self, ctx: bentoml.Context) -> dict[str, str]:
        """Readiness check for the service."""
        if ctx.state.get("queued_processing", 0) >= settings.bentoml.inference.slow_batched_op_max_batch_size:
            ctx.response.status_code = 503
            return {"status": "not ready"}

        try:
            # Test database connection
            async with AIOPostgres().session() as conn:
                await conn.execute(select(1))

            # Test embedding service with timeout
            try:
                await asyncio.wait_for(self.embedding_service.readyz(), timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.error("Embedding service readiness check timed out")
                raise
        except Exception as e:
            self.logger.exception(f"Readiness check failed: {e}")
            ctx.response.status_code = 503
            raise
        else:
            ctx.response.status_code = 200
            return {"status": "ready"}
