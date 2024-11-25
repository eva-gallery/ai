from __future__ import annotations

import hashlib
import io
import os
import asyncio
from typing import cast, Optional, TYPE_CHECKING

import aiohttp
import bentoml
from PIL import Image as PILImage
from sqlalchemy import select
import numpy as np

from . import settings
from .database import AIOPostgres
from .model import ImageDuplicateStatus, SearchRequest, SearchResponse, ImageSearchRequest, ImageSearchResponse, BackendPatchRequest, AIGeneratedStatus, ProcessImageRequest
from .orm import Annotation, GalleryEmbedding, Image, ModifiedImageData
from .util import get_logger
from .services import InferenceService


@bentoml.service(
    name="evagallery_ai_api",
    **settings.bentoml.service.api.to_dict()
)
class APIService:
    def __init__(self):
        self.embedding_service = bentoml.depends(InferenceService)
        self.logger = get_logger()
        self.db_healthy = False
        asyncio.run_coroutine_threadsafe(self._init_postgres(), asyncio.get_event_loop())
        
    async def _init_postgres(self) -> None:
        try:
            AIOPostgres(url=settings.postgres.url)
            self._migrate_database()
        except Exception as e:
            self.logger.error(f"Failed to initialize Postgres client: {e}")
            raise e
        self.db_healthy = True
    
    def _migrate_database(self) -> None:
        """Run alembic migrations to head revision"""
        if os.getenv("CI"):
            return
        from alembic.config import Config
        from alembic import command
        
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    
    async def _save_and_notify(self,
                               image_id: int,
                               duplicate_status: ImageDuplicateStatus,
                               most_similar_image_id: Optional[int] = None,
                               image_data_model: Optional[ModifiedImageData] = None,
                               gallery_embedding_model: Optional[GalleryEmbedding] = None,
                               annotation_model: Optional[Annotation] = None) -> None:
        
        if duplicate_status is not ImageDuplicateStatus.OK:
            patch_request = BackendPatchRequest(
                image_id=image_id,
                closest_match_id=most_similar_image_id,
                image_duplicate_status=duplicate_status
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    settings.eva_backend.backend_image_patch_route,
                    json=patch_request.model_dump_json()
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to patch backend: {await response.text()}")
                        raise Exception(f"Failed to patch backend for image id {image_id}")
        
        if TYPE_CHECKING:
            image_data_model = cast(ModifiedImageData, image_data_model)
            annotation_model = cast(Annotation, annotation_model)
            gallery_embedding_model = cast(GalleryEmbedding, gallery_embedding_model)
        
        async with AIOPostgres() as conn:
            conn.add(image_data_model)
            conn.add(annotation_model)
            await conn.flush()  # Flush to get IDs
            
            # Create and add Image model to establish relationships
            image = Image(
                image_data_id=image_data_model.id,
                annotation_id=annotation_model.id
            )
            conn.add(image)
            await conn.flush()
            
            # Set image_id on gallery embedding
            gallery_embedding_model.image_id = image.id
            conn.add(gallery_embedding_model)

        # Send PATCH request to backend
        patch_request = BackendPatchRequest(
            image_id=image_data_model.original_image_id,
            metadata=image_data_model.image_metadata
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                settings.eva_backend.backend_image_patch_route,
                json=patch_request.model_dump_json()
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to patch backend: {await response.text()}")
                    raise Exception(f"Failed to patch backend for image id {image_data_model.original_image_id}")
    
    async def _process_image_data(
        self,
        image_pil: PILImage.Image,
        artwork_id: int,
        ai_generated_status: AIGeneratedStatus,
        metadata: dict
    ) -> None:
        # Generate a shorter hash that will fit in 32 bits
        original_hash = hashlib.sha256(image_pil.tobytes()).hexdigest()[:8]  # First 32 bits (8 hex chars)
        
        image_embed = (await self.embedding_service.embed_image(image_pil))[0]
        
        duplicate_status = ImageDuplicateStatus.OK
        most_similar_image_id = None
        
        if metadata.get("ignore_duplicate_check", False) in (False, None):
            # Check for duplicates
            watermark_result, ai_watermark = await asyncio.gather(
                self.embedding_service.check_watermark([image_pil]),
                self.embedding_service.check_ai_watermark([image_pil])
            )
            has_watermark, watermark = watermark_result[0]
            ai_watermark = ai_watermark[0]
            
            if ai_watermark:
                duplicate_status = ImageDuplicateStatus.PLAGIARIZED
            elif has_watermark and watermark:
                async with AIOPostgres().session() as conn:
                    result = await conn.execute(
                        select(ModifiedImageData)
                        .filter_by(image_hash=watermark)  # Changed from md5_hash to image_hash
                    )
                    if result.scalar_one_or_none():
                        result = await conn.execute(
                            select(GalleryEmbedding.image_embedding, GalleryEmbedding.image_id)
                            .order_by(GalleryEmbedding.image_embedding_distance_to(image_embed))
                        )
                        float_vec, most_similar_image_id = (await (await result.scalars()).one())  # type: ignore
                        # check if dot product is more than 95% similar
                        if np.dot(image_embed, float_vec) > settings.model.detection.threshold:
                            duplicate_status = ImageDuplicateStatus.EXISTS
        
        if duplicate_status is not ImageDuplicateStatus.OK:
            await self._save_and_notify(
                duplicate_status=duplicate_status,
                image_id=artwork_id,
                most_similar_image_id=most_similar_image_id
            )
            return
        
        # Get embeddings and caption
        generated_image_caption, metadata_embedding = await asyncio.gather(
            self.embedding_service.generate_caption(image_pil),
            self.embedding_service.embed_text([str(metadata)])
        )
        generated_image_caption = generated_image_caption[0]
        metadata_embedding = metadata_embedding[0]
        
        # Detect if AI generated
        watermarked_image_pil = image_pil
        watermarked_image_embed = None
        if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
            ai_generated_status = (await self.embedding_service.detect_ai_generation(image_pil))[0]
            if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
                watermarked_image_pil = (await self.embedding_service.add_ai_watermark(image_pil))[0]
        
        watermarked_image_pil = (await self.embedding_service.add_watermark(watermarked_image_pil, original_hash))[0]
        watermarked_image_embed, generated_caption_embed = await asyncio.gather(
            self.embedding_service.embed_image([watermarked_image_pil]),
            self.embedding_service.embed_text([generated_image_caption])
        )
        watermarked_image_embed = watermarked_image_embed[0]
        generated_caption_embed = generated_caption_embed[0]
        
        user_caption = metadata.get('caption')
        user_caption_embed = None
        if user_caption:
            user_caption_embed = (await self.embedding_service.embed_text([user_caption]))[0]

        metadata['ai_generated'] = ai_generated_status
        metadata['duplicate_status'] = duplicate_status

        # Save to database and send to backend
        await self._save_and_notify(
            duplicate_status=duplicate_status,
            image_id=artwork_id,
            image_data_model=ModifiedImageData(
                original_image_id=artwork_id,
                image_metadata=metadata,
                image_hash=original_hash
            ),
            gallery_embedding_model=GalleryEmbedding(
                image_embedding=image_embed,
                watermarked_image_embedding=watermarked_image_embed,
                metadata_embedding=metadata_embedding,
                user_caption_embedding=user_caption_embed,
                generated_caption_embedding=generated_caption_embed,
            ),
            annotation_model=Annotation(
                user_annotation=user_caption,
                generated_annotation=generated_image_caption
            )
        )

    @bentoml.api(route="/image/search_query", input_spec=SearchRequest, output_spec=SearchResponse)  # type: ignore
    async def search_query(self, query: str, count: int, page: int) -> SearchResponse:
        embedded_text: np.ndarray = (await self.embedding_service.embed_text([query]))[0]

        async with AIOPostgres() as conn:
            stmt = (
                select(GalleryEmbedding.image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(embedded_text))
                .limit(count)
                .offset(page * count)
            )
            
            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore

        return SearchResponse(image_id=list(results))
    
    @bentoml.api(route="/image/search_image", input_spec=ImageSearchRequest, output_spec=ImageSearchResponse)  # type: ignore
    async def search_image(self, image_id: int, count: int, page: int) -> ImageSearchResponse:
        embedded_image: np.ndarray = (await self.embedding_service.embed_image([image_id]))[0]
        
        async with AIOPostgres() as conn:
            stmt = (
                select(GalleryEmbedding.image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(embedded_image))
                .limit(count)
                .offset(page * count)
            )

            result = await conn.execute(stmt)
            results = await (await result.scalars()).all()  # type: ignore

        return ImageSearchResponse(image_id=list(results))

    @bentoml.api(route="/image/process", input_spec=ProcessImageRequest)  # type: ignore
    async def process_image(
        self,
        image: bytes,
        artwork_id: int,
        ai_generated_status: AIGeneratedStatus,
        metadata: dict,
        ctx: bentoml.Context
    ) -> None:
        image_pil = PILImage.open(io.BytesIO(image)).convert("RGB")
        
        # Start processing in background
        asyncio.create_task(self._process_image_data(
            image_pil=image_pil,
            artwork_id=artwork_id,
            ai_generated_status=ai_generated_status,
            metadata=metadata
        ))
        
        ctx.response.status_code = 201

    @bentoml.api(route="/healthz")  # type: ignore
    async def healthz(self, ctx: bentoml.Context) -> dict[str, str]:
        """Basic health check endpoint."""
        if not self.db_healthy:
            ctx.response.status_code = 503
            return {"status": "unhealthy"}
        
        ctx.response.status_code = 200
        return {"status": "healthy"}

    @bentoml.api(route="/readyz")  # type: ignore
    async def readyz(self, ctx: bentoml.Context) -> dict[str, str]:
        """
        Readiness check endpoint that verifies all dependencies are available.
        Checks database connection and embedding service availability.
        """
        try:
            # Test database connection
            async with AIOPostgres().session() as conn:
                await conn.execute(select(1))
            
            # Test embedding service
            await self.embedding_service.readyz()
            
            ctx.response.status_code = 200
            return {"status": "ready"}
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            ctx.response.status_code = 503
            raise bentoml.exceptions.ServiceUnavailable("Service is not ready")
