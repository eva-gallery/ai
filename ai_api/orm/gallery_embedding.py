from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import Float, ForeignKey, ColumnElement, custom_op
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base
from ai_api import settings


if TYPE_CHECKING:
    from .image import Image


class GalleryEmbedding(Base):
    __tablename__ = "gallery_embedding"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    
    image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    watermarked_image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    metadata_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    user_caption_embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.model.embedding.dimension), nullable=True)
    generated_caption_embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.model.embedding.dimension), nullable=True)
    
    image: Mapped["Image"] = relationship("Image", back_populates="gallery_embeddings")

    @classmethod
    def image_embedding_distance_to(cls, vector: list[float] | np.ndarray) -> ColumnElement[float]:
        return custom_op('<#>', return_type=Float)(cls.image_embedding, vector) * -1
    
    @classmethod
    def watermarked_image_embedding_distance_to(cls, vector: list[float] | np.ndarray) -> ColumnElement[float]:
        return custom_op('<#>', return_type=Float)(cls.watermarked_image_embedding, vector) * -1
    
    @classmethod
    def metadata_embedding_distance_to(cls, vector: list[float] | np.ndarray) -> ColumnElement[float]:
        return custom_op('<#>', return_type=Float)(cls.metadata_embedding, vector) * -1
    
    @classmethod
    def user_caption_embedding_distance_to(cls, vector: list[float] | np.ndarray) -> ColumnElement[float]:
        return custom_op('<#>', return_type=Float)(cls.user_caption_embedding, vector) * -1
    
    @classmethod
    def generated_caption_embedding_distance_to(cls, vector: list[float] | np.ndarray) -> ColumnElement[float]:
        return custom_op('<#>', return_type=Float)(cls.generated_caption_embedding, vector) * -1