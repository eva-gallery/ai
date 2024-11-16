from __future__ import annotations

import hashlib
import io
import asyncio
from typing import cast, Optional, TYPE_CHECKING

import aiohttp
import bentoml
from PIL import Image as PILImage
from sqlalchemy import select
import numpy as np

from ai_api import settings
from ai_api.database.postgres_client import AIOPostgres
from ai_api.model.api.status import ImageDuplicateStatus
from ai_api.model.api.query_search import SearchRequest, SearchResponse
from ai_api.model.api.image_search import ImageSearchRequest, ImageSearchResponse
from ai_api.model.api.process import BackendPatchRequest, AIGeneratedStatus, ProcessImageRequest
from ai_api.orm.annotation import Annotation
from ai_api.orm.gallery_embedding import GalleryEmbedding
from ai_api.orm.image import Image
from ai_api.util.logger import get_logger
from ai_api.services.embedding_service import EmbeddingService
from ai_api.orm import ModifiedImageData


@bentoml.service(
    name="evagallery_ai_api",
    workers=settings.bentoml.api.workers,
    resources=settings.bentoml.api.resources.to_dict()
)
class APIService:
    def __init__(self):
        self.embedding_service = bentoml.depends(EmbeddingService)
        self.logger = get_logger()
        
        try:
            AIOPostgres(url=settings.postgres.url)
        except Exception as e:
            self.logger.error(f"Failed to initialize Postgres client: {e}")
            raise e
    
    async def _save_and_notify(self,
                               image_id: int,
                               duplicate_status: ImageDuplicateStatus,
                               image_embedding: list[float],
                               image_data_model: Optional[ModifiedImageData] = None,
                               gallery_embedding_model: Optional[GalleryEmbedding] = None,
                               annotation_model: Optional[Annotation] = None) -> None:
        
        if duplicate_status is not ImageDuplicateStatus.OK:
            async with AIOPostgres().session() as conn:
                stmt = select(GalleryEmbedding.image_id).order_by(GalleryEmbedding.image_embedding_distance_to(image_embedding))
                result = (await conn.execute(stmt)).scalar_one_or_none()
                
            patch_request = BackendPatchRequest(
                image_id=image_id,
                closest_match_id=result,
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

    @bentoml.api(route="/image/search_query", input_spec=SearchRequest, output_spec=SearchResponse)  # type: ignore
    async def search_query(self, query: str, count: int, page: int) -> SearchResponse:
        embedded_text: np.ndarray = (await self.embedding_service.embed_text.async_run([query]))[0]

        async with AIOPostgres() as conn:
            stmt = (
                select(GalleryEmbedding.image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(embedded_text))
                .limit(count)
                .offset(page * count)
            )
            
            results = (await conn.execute(stmt)).scalars().all()

        return SearchResponse(image_id=cast(list[int], results))
    
    @bentoml.api(route="/image/search_image", input_spec=ImageSearchRequest, output_spec=ImageSearchResponse)  # type: ignore
    async def search_image(self, image_id: int, count: int, page: int) -> ImageSearchResponse:
        embedded_image: np.ndarray = (await self.embedding_service.embed_image.async_run([image_id]))[0]
        
        async with AIOPostgres() as conn:
            stmt = (
                select(GalleryEmbedding.image_id)
                .order_by(GalleryEmbedding.image_embedding_distance_to(embedded_image))
                .limit(count)
                .offset(page * count)
            )

            results = (await conn.execute(stmt)).scalars().all()

        return ImageSearchResponse(image_id=cast(list[int], results))

    async def _process_image_data(
        self,
        image_pil: PILImage.Image,
        artwork_id: int,
        ai_generated_status: AIGeneratedStatus,
        metadata: dict
    ) -> None:
        
        duplicate_status = ImageDuplicateStatus.OK
        if metadata.get("ignore_duplicate_check", False) in (False, None):
            # Check for duplicates
            watermark_result, ai_watermark = await asyncio.gather(
                self.embedding_service.check_watermark.async_run(image_pil),
                self.embedding_service.check_ai_watermark.async_run(image_pil)
            )
            has_watermark, watermark = watermark_result[0]
            ai_watermark = ai_watermark[0]
            
            if has_watermark or ai_watermark:
                async with AIOPostgres().session() as conn:
                    result = await conn.execute(
                        select(ModifiedImageData)
                        .filter_by(md5_hash=watermark)
                    )
                    if result.scalar_one_or_none():
                        duplicate_status = ImageDuplicateStatus.EXISTS
                if ai_watermark:
                    duplicate_status = ImageDuplicateStatus.PLAGIARIZED
                    
        image_embed = await self.embedding_service.embed_image.async_run([image_pil])
        image_embed = image_embed[0]
        
        if duplicate_status is not ImageDuplicateStatus.OK:
            await self._save_and_notify(
                duplicate_status=duplicate_status,
                image_id=artwork_id,
                image_embedding=image_embed
            )
            return
        
        # Get embeddings and caption
        original_md5 = hashlib.md5(image_pil.tobytes()).hexdigest()
        generated_image_caption, metadata_embedding = await asyncio.gather(
            self.embedding_service.generate_caption.async_run(image_pil),
            self.embedding_service.embed_text.async_run([str(metadata)])
        )
        generated_image_caption = generated_image_caption[0]
        metadata_embedding = metadata_embedding[0]
        
        # Detect if AI generated
        watermarked_image_pil = image_pil
        watermarked_image_embed = None
        if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
            ai_generated_status, _ = await self.embedding_service.detect_ai_generation.async_run(image_pil)
            if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
                watermarked_image_pil = await self.embedding_service.add_ai_watermark.async_run(image_pil)
        
        watermarked_image_pil = await self.embedding_service.add_watermark.async_run(watermarked_image_pil, original_md5)
        watermarked_image_embed, generated_caption_embed = await asyncio.gather(
            self.embedding_service.embed_image.async_run([watermarked_image_pil]),
            self.embedding_service.embed_text.async_run([generated_image_caption])
        )
        watermarked_image_embed = watermarked_image_embed[0]
        generated_caption_embed = generated_caption_embed[0]
        
        user_caption = metadata.get('caption')
        user_caption_embed = None
        if user_caption:
            user_caption_embed = await self.embedding_service.embed_text.async_run([user_caption])

        metadata['ai_generated'] = ai_generated_status
        metadata['duplicate_status'] = duplicate_status

        # Save to database and send to backend
        await self._save_and_notify(
            duplicate_status=duplicate_status,
            image_id=artwork_id,
            image_embedding=image_embed,
            image_data_model=ModifiedImageData(
                original_image_id=artwork_id,
                image_metadata=metadata,
                md5_hash=original_md5
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
            await self.embedding_service.readyz.async_run()
            
            ctx.response.status_code = 200
            return {"status": "ready"}
        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            ctx.response.status_code = 503
            raise bentoml.exceptions.ServiceUnavailable("Service is not ready")
