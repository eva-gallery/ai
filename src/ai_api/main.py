from __future__ import annotations

import io
import asyncio
import uuid
from typing import Optional

import bentoml
import httpx
from PIL import Image as PILImage
from sqlalchemy import text, select

from ai_api import settings
from ai_api.database.postgres_client import AIOPostgres
from ai_api.model.api.status import ResponseStatus, ResponseType
from ai_api.model.api.search import SearchRequest, SearchResponse
from ai_api.model.api.process import BackendPatchRequest, ProcessImageRequest, AIGeneratedStatus
from ai_api.util.logger import get_logger
from ai_api.runners.embedding_runner import EmbeddingRunner
from ai_api.orm import ImageData, GalleryEmbedding

# Initialize shared resources
runner = bentoml.Runner(EmbeddingRunner, name="embedding_runner")
pg = AIOPostgres(url=settings.postgres.url)
logger = get_logger()
svc = bentoml.Service(
    "embedding-search", 
    runners=[runner], 
    workers=settings.bentoml.workers,  # type: ignore
    resources={"gpu": settings.bentoml.gpus}  # type: ignore
)


@svc.on_startup
async def startup(ctx):
    # Run alembic migrations on startup
    import os
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from pathlib import Path

    alembic_cfg = Config(Path(__file__).parent.parent.parent / "alembic.ini")
    script = ScriptDirectory.from_config(alembic_cfg)
    heads = script.get_revisions("head")
    if heads:
        command.upgrade(alembic_cfg, "head")

@svc.api(
    route="/search",
    input=SearchRequest,  # type: ignore
    output=SearchResponse,  # type: ignore
)
async def get_similar(queries: list[str], count: Optional[int] = 50, page: Optional[int] = 0):
    embedded_texts = await runner.embed_text.async_run(queries)
    
    # embedded_texts is a list of lists of floats from the embedding model
    if not isinstance(embedded_texts, list) or not all(isinstance(x, list) for x in embedded_texts):
        logger.error("Failed to embed queries")
        raise ValueError("Failed to embed queries")

    async with pg as conn:
        stmt = text("""
            SELECT image_id
            FROM gallery_embedding
            ORDER BY text_embedding <=> :query 
            LIMIT :count OFFSET :offset
        """)
        results = []

        for embedded_text in embedded_texts:
            result = await conn.execute(
                stmt,
                {"query": embedded_text, "count": count, "offset": page * count}  # type: ignore
            )  
            results.append([x.image_id for x in result.fetchall()])

    return SearchResponse(id_list=results).model_dump()

async def _process_image_data(
    image_pil: PILImage.Image,
    artwork_id: str,
    ai_generated_status: AIGeneratedStatus,
    metadata: dict
) -> None:
    # Get embeddings and caption
    image_embed = await runner.embed_image.async_run([image_pil])
    image_caption = metadata.get('caption', await runner.generate_caption.async_run(image_pil))
    metadata_embedding = await runner.embed_text.async_run([str(metadata)])
    caption_embedding = await runner.embed_text.async_run([image_caption])

    # Generate watermark
    new_uuid = str(uuid.uuid4())
    watermarked_image_pil = await runner.add_watermark.async_run(image_pil, new_uuid)
    
    # Detect if AI generated
    watermarked_image_embed = None
    if ai_generated_status == AIGeneratedStatus.NOT_GENERATED:
        ai_generated_status, score = await runner.detect_ai_generation.async_run(image_pil)
        if ai_generated_status == AIGeneratedStatus.GENERATED:
            watermarked_image_embed = await runner.embed_image.async_run([watermarked_image_pil])

    # Save to database
    async with pg as conn:
        async with conn.begin():
            try:
                # Save image data and embeddings
                image_data_record = ImageData(
                    original_image_uuid=uuid.UUID(artwork_id),
                    modified_image_uuid=new_uuid,
                    image_metadata=metadata
                )
                conn.add(image_data_record)
                
                gallery_embedding_record = GalleryEmbedding(
                    image_embedding_model_id=settings.model.embedding.image.name,
                    text_embedding_model_id=settings.model.embedding.text.name,
                    captioning_model_id=settings.model.captioning.name,
                    image_embedding=image_embed[0],
                    watermarked_image_embedding=watermarked_image_embed[0] if watermarked_image_embed else None,
                    text_embedding=caption_embedding[0],
                    metadata_embedding=metadata_embedding[0],
                    image_id=image_data_record.id
                )
                conn.add(gallery_embedding_record)
                
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")
                raise

    # Send to eva backend endpoint
    async with httpx.AsyncClient() as client:
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        
        await client.patch(
            f"{settings.eva_backend.url}{settings.eva_backend.patch_endpoint}",
            json=BackendPatchRequest(
                image_uuid=artwork_id,
                modified_image_uuid=new_uuid,
                image=img_byte_arr.getvalue(),
                ai_generated_status=ai_generated_status,
                metadata=metadata
            ).model_dump(by_alias=True)
        )

@svc.api(
    route="/image/process",
    input=ProcessImageRequest,  # type: ignore
    output=ResponseStatus,  # type: ignore
)
async def process_image(
    image: bytes,
    artwork_id: str,
    ai_generated_status: AIGeneratedStatus,
    metadata: dict
) -> dict:
    image_pil = PILImage.open(io.BytesIO(image)).convert("RGB")
    
    # Check for duplicates
    has_watermark, watermark = await runner.check_watermark.async_run(image_pil)
    if has_watermark:
        async with pg as conn:
            result = await conn.execute(
                select(ImageData).filter_by(modified_image_uuid=watermark)
            )
            if result.scalar():
                return ResponseStatus(status=ResponseType.EXISTS).model_dump()
    
    if await runner.check_ai_watermark.async_run(image_pil):
        return ResponseStatus(status=ResponseType.EXISTS).model_dump()
    
    # Start processing in background
    asyncio.create_task(_process_image_data(
        image_pil=image_pil,
        artwork_id=artwork_id,
        ai_generated_status=ai_generated_status,
        metadata=metadata
    ))
    
    return ResponseStatus(status=ResponseType.OK).model_dump(by_alias=True)
