from __future__ import annotations

import bentoml
from sqlalchemy import text

from ai_api.database.postgres_client import AIOPostgres
from ai_api.orm.image import Image
from ai_api.util.logger import get_logger


@bentoml.service(workers=1, resources={"gpu": 1})
class EmbeddingSearchService:
    def __init__(self) -> None:
        from autodistill_metaclip import MetaCLIP
        self.model = MetaCLIP(None)  # noqa
        self.logger = get_logger()
        self.pg = AIOPostgres(url="postgresql+asyncpg://postgres:password@localhost:5432/ai_api")

    @bentoml.api(batchable=True)
    async def embed(self, images: list[str]):
        # image_data = get_from_server()  # TODO
        image_embeds = [self.model.embed_image(image) for image in images]
        # save to pg
        async with self.pg as conn_init:
            async with conn_init.begin() as conn:
                try:
                    conn.session.add_all([Image(id=image, image_vector=vector) for image, vector in zip(images, image_embeds)])
                except Exception as e:
                    await conn.rollback()
                    raise e
                else:
                    await conn.commit()

    @bentoml.api(batchable=True)
    async def get_similar(self, query: str):
        embedded_text = self.model.embed_text(query)
        async with self.pg as conn:
            stmt = text("SELECT id FROM image ORDER BY image_vector <=> :query LIMIT 3")
            results = await conn.execute(stmt, {"query": embedded_text})

        results = results.fetchall()
        results = [result.id for result in results]
        return results
