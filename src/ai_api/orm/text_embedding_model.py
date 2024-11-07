from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base
from ai_api import settings


if TYPE_CHECKING:
    from .gallery_embedding import GalleryEmbedding


class TextEmbeddingModel(Base):
    __tablename__ = "text_embedding_model"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    vector_length: Mapped[int] = mapped_column(Vector(settings.model.embedding.text.vector_length))

    gallery_embeddings: Mapped[list["GalleryEmbedding"]] = relationship("GalleryEmbedding", back_populates="embedding_model")
