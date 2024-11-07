from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .base import Base
from ai_api import settings


if TYPE_CHECKING:
    from .image_embedding_model import ImageEmbeddingModel
    from .text_embedding_model import TextEmbeddingModel
    from .captioning_model import CaptioningModel
    from .image import Image


class GalleryEmbedding(Base):
    __tablename__ = "gallery_embedding"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_embedding_model_id: Mapped[int] = mapped_column(ForeignKey("image_embedding_model.id"))
    text_embedding_model_id: Mapped[int] = mapped_column(ForeignKey("text_embedding_model.id"))
    captioning_model_id: Mapped[int] = mapped_column(ForeignKey("captioning_model.id"))
    image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.image.vector_length))
    watermarked_image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.image.vector_length))
    text_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.text.vector_length))
    metadata_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.metadata.vector_length))
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))

    image_embedding_model: Mapped["ImageEmbeddingModel"] = relationship("ImageEmbeddingModel", back_populates="gallery_embeddings")
    text_embedding_model: Mapped["TextEmbeddingModel"] = relationship("TextEmbeddingModel", back_populates="gallery_embeddings")
    captioning_model: Mapped["CaptioningModel"] = relationship("CaptioningModel", back_populates="gallery_embeddings")
    image: Mapped["Image"] = relationship("Image", back_populates="gallery_embeddings")
