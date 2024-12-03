# type: ignore[import-cycle]
"""The ORM model for the gallery embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ColumnElement, Float, ForeignKey, custom_op
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_api import settings

from .base import Base

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .image import Image


class GalleryEmbedding(Base):
    """The ORM model for the gallery embedding."""

    __tablename__ = "gallery_embedding"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))

    image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    watermarked_image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    metadata_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    user_caption_embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.model.embedding.dimension), nullable=True)
    generated_caption_embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.model.embedding.dimension), nullable=True)

    image: Mapped[Image] = relationship("image", back_populates=__tablename__)

    @classmethod
    def image_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate the distance between the image embedding and a vector."""
        return custom_op("<#>", return_type=Float)(cls.image_embedding, vector) * -1

    @classmethod
    def watermarked_image_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate the distance between the watermarked image embedding and a vector."""
        return custom_op("<#>", return_type=Float)(cls.watermarked_image_embedding, vector) * -1

    @classmethod
    def metadata_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate the distance between the metadata embedding and a vector."""
        return custom_op("<#>", return_type=Float)(cls.metadata_embedding, vector) * -1

    @classmethod
    def user_caption_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate the distance between the user caption embedding and a vector."""
        return custom_op("<#>", return_type=Float)(cls.user_caption_embedding, vector) * -1

    @classmethod
    def generated_caption_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate the distance between the generated caption embedding and a vector."""
        return custom_op("<#>", return_type=Float)(cls.generated_caption_embedding, vector) * -1
