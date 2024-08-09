from __future__ import annotations

import bentoml
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from transformers import pipeline

from ai_api.database.postgres_client import AIOPostgres
from ai_api.orm.image import Image
from ai_api.util.logger import get_logger


@bentoml.service(workers=1, resources={"gpu": 1})
class EmbeddingSearchService:
    def __init__(self) -> None:
        self.model_img = SentenceTransformer('clip-ViT-B-32')
        self.model_text = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        self.model_caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        self.logger = get_logger()
        self.pg = AIOPostgres(url="postgresql+asyncpg://postgres:password@localhost:5432/ai_api")

    @bentoml.api(batchable=True)
    async def embed(self, images: list[str]):
        # image_data = get_from_server()  # TODO
        image_embeds = [self.model_img.encode(image) for image in images]
        image_captions = [self.model_caption(image)['generated_text'] for image in images]
        # save to pg
        async with self.pg as conn_init:
            async with conn_init.begin() as conn:
                try:
                    conn.session.add_all([Image(id=image, image_vector=vector, image_caption=caption)
                                          for image, vector, caption in zip(images, image_embeds, image_captions)])
                except Exception as e:
                    await conn.rollback()
                    raise e
                else:
                    await conn.commit()

    @bentoml.api(batchable=True)
    async def get_similar(self, queries: list[str], count: int = 50):
        embedded_texts = self.model_text.encode(queries)

        async with self.pg as conn:
            stmt = text("SELECT id FROM image ORDER BY image_vector <=> :query LIMIT :count")
            results = []
            for embedded_text in embedded_texts:
                result = await conn.execute(stmt, {"query": embedded_text, "count": count})
                results.append([x.id for x in result.fetchall()])

        return results
