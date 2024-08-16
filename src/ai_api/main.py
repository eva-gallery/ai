from __future__ import annotations

import os
from typing import Optional

import bentoml
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from torch.cuda import is_available
from transformers import pipeline

from ai_api.database.postgres_client import AIOPostgres
from ai_api.orm.image import Image
from ai_api.util.logger import get_logger


@bentoml.service(workers=os.environ.get("BENTOML_WORKERS", 1), resources={"gpu": os.environ.get("BENTOML_GPUS", 1)})
class EmbeddingSearchService:
    def __init__(self) -> None:
        device = "cuda" if is_available() else "cpu"
        self.model_img = SentenceTransformer(os.environ.get("IMAGE_EMBEDDING_MODEL", "clip-ViT-B-32"), device=device)
        self.model_text = SentenceTransformer(os.environ.get("TEXT_EMBEDDING_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1"),
                                              device=device)
        self.model_caption = pipeline("image-to-text", model=os.environ.get("IMAGE_CAPTION_MODEL", "nlpconnect/vit-gpt2-image-captioning"),
                                      device=device)
        # self.model_tags = ???  # Add logit-based tag-model with mutlimodal capabilities to make use of caption
        self.logger = get_logger()
        pg_url = os.environ.get("POSTGRES_USER", "postgres") + ":" + os.environ.get("POSTGRES_PASSWORD", "password") + "@" + \
                 os.environ.get("POSTGRES_HOST", "localhost") + ":" + os.environ.get("POSTGRES_PORT", "5432") + "/" + \
                 os.environ.get("POSTGRES_DB", "ai_api")  # Will remake into a pydantic model incl all other configurables
        # self.pg = AIOPostgres(url=pg_url)
        self.pg = AIOPostgres()  # Uses sqlite for now

    @bentoml.api(batchable=True)
    async def embed(self, images: list[str], captions: list[str]):  # TODO Remake to accept images as blobs with metadata
        # image_data = get_from_server() ??? # TODO
        image_embeds = [self.model_img.encode(image) for image in images]
        image_captions = [self.model_caption(image)['generated_text'] if not caption else caption
                          for caption, image in zip(captions, images)]
        # save to pg
        async with self.pg as conn_init:
            async with conn_init.begin() as conn:
                try:
                    conn.session.add_all([Image(id=image, image_vector=vector, image_caption=caption)
                                          for image, vector, caption in zip(images, image_embeds, image_captions)])
                except Exception as e:  # TODO Add API custom exceptions and OTL tracing
                    await conn.rollback()
                    raise e
                else:
                    await conn.commit()

    @bentoml.api(batchable=True, batch_dim=(0, 0), max_batch_size=32, max_latency_ms=1000)
    async def get_similar(self, queries: list[str], count: Optional[int] = 50, page: Optional[int] = 0):
        embedded_texts = self.model_text.encode(queries)

        async with self.pg as conn:
            stmt = text("""
                SELECT id
                FROM image
                ORDER BY image_vector <=> :query 
                LIMIT :count OFFSET :offset
            """)
            results = []

            for embedded_text in embedded_texts:
                result = await conn.execute(stmt, {"query": embedded_text, "count": count, "offset": page * count})
                results.append([x.id for x in result.fetchall()])

        return results
