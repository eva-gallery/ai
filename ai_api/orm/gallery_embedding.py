# type: ignore[import-cycle]
"""Module containing the ORM model for image embeddings and similarity search.

This module defines the database schema for storing various types of embeddings
associated with images, including raw image embeddings, watermarked image embeddings,
metadata embeddings, and caption embeddings. It also provides methods for calculating
similarity distances between embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ColumnElement, Float, ForeignKey, custom_op
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_api import settings
from ai_api.orm.base import Base

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_api.orm.image import Image


class GalleryEmbedding(Base):
    """SQLAlchemy ORM model for storing image embeddings and similarity search data.

    This class represents the database schema for storing various types of embeddings
    used in similarity search operations. It includes embeddings for the original image,
    watermarked version, metadata, and captions.

    Attributes:
        id: Primary key auto-incrementing ID.
        image_id: Foreign key reference to the associated image.
        image_embedding: Vector embedding of the original image.
        watermarked_image_embedding: Vector embedding of the watermarked image.
        metadata_embedding: Vector embedding of the image metadata.
        user_caption_embedding: Optional vector embedding of user-provided caption.
        generated_caption_embedding: Optional vector embedding of AI-generated caption.
        image: Relationship to the associated Image model.

    """

    __tablename__ = "gallery_embedding"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))

    image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    watermarked_image_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    metadata_embedding: Mapped[list[float]] = mapped_column(Vector(settings.model.embedding.dimension))
    user_caption_embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.model.embedding.dimension), nullable=True)
    generated_caption_embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.model.embedding.dimension), nullable=True)

    image: Mapped[Image] = relationship(
        "Image",
        back_populates="gallery_embedding",
    )

    @classmethod
    def image_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate cosine similarity between image embedding and input vector.

        :param vector: Input vector to compare against image embedding.
        :type vector: Sequence[float]
        :returns: Negative cosine distance (higher value means more similar).
        :rtype: ColumnElement[float]
        """
        return custom_op("<#>", return_type=Float)(cls.image_embedding, vector) * -1

    @classmethod
    def watermarked_image_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate cosine similarity between watermarked image embedding and input vector.

        :param vector: Input vector to compare against watermarked image embedding.
        :type vector: Sequence[float]
        :returns: Negative cosine distance (higher value means more similar).
        :rtype: ColumnElement[float]
        """
        return custom_op("<#>", return_type=Float)(cls.watermarked_image_embedding, vector) * -1

    @classmethod
    def metadata_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate cosine similarity between metadata embedding and input vector.

        :param vector: Input vector to compare against metadata embedding.
        :type vector: Sequence[float]
        :returns: Negative cosine distance (higher value means more similar).
        :rtype: ColumnElement[float]
        """
        return custom_op("<#>", return_type=Float)(cls.metadata_embedding, vector) * -1

    @classmethod
    def user_caption_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate cosine similarity between user caption embedding and input vector.

        :param vector: Input vector to compare against user caption embedding.
        :type vector: Sequence[float]
        :returns: Negative cosine distance (higher value means more similar).
        :rtype: ColumnElement[float]
        """
        return custom_op("<#>", return_type=Float)(cls.user_caption_embedding, vector) * -1

    @classmethod
    def generated_caption_embedding_distance_to(cls, vector: Sequence[float]) -> ColumnElement[float]:
        """Calculate cosine similarity between generated caption embedding and input vector.

        :param vector: Input vector to compare against generated caption embedding.
        :type vector: Sequence[float]
        :returns: Negative cosine distance (higher value means more similar).
        :rtype: ColumnElement[float]
        """
        return custom_op("<#>", return_type=Float)(cls.generated_caption_embedding, vector) * -1
