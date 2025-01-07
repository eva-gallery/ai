"""Module containing the SQLAlchemy ORM model for image data.

This module defines the database schema and relationships for storing image data,
including metadata, embeddings, and various status flags.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import UUID, Enum, Float, Function, Index, String, func
from sqlalchemy.dialects.postgresql import BOOLEAN, BYTEA, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.sqltypes import Numeric
from sqlalchemy.types import TypeDecorator

from ai_api.model.api import AIGeneratedStatus
from ai_api.orm.base import Base

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ColumnElement

    from ai_api.orm.gallery_embedding import GalleryEmbedding


class ImageHash(TypeDecorator[bytes]):
    """Custom type for image hash that adds hamming distance comparison.

    Extends the BYTEA type to add hamming distance operator support.
    """

    impl = BYTEA
    cache_ok = True

    @property
    def comparator_factory(self) -> type[BYTEA.Comparator[bytes]]:
        """Get the comparator factory for this type.

        :returns: A comparator class with hamming distance support.
        """
        class ImageHashComparator(self.impl.Comparator[bytes]):
            """Comparator class that adds hamming distance and similarity operators."""

            def hamming_distance(self, other: bytes) -> Function[int]:
                """Calculate hamming distance between two binary hashes.

                :param other: The other hash to compare against.
                :returns: SQL expression for hamming distance calculation.
                """
                return func.bit_count(func.bit_xor(self.expr, other))

            def hamming_similarity(self, other: bytes) -> ColumnElement[Numeric[float]]:
                """Calculate similarity between two binary hashes based on hamming distance.

                :param other: The other hash to compare against.
                :returns: Float between 0 and 1 representing similarity (1 = identical).
                """
                return func.cast(1.0 - (func.cast(self.hamming_distance(other), Float) / 64.0), Numeric(10, 2))

        return ImageHashComparator


class Image(Base):
    """SQLAlchemy ORM model for storing image data and metadata.

    This class represents the database schema for storing images, including their
    metadata, AI generation status, and relationships to embeddings.

    :param id: Primary key auto-incrementing ID.
    :type id: int
    :param image_uuid: Unique UUID for the image.
    :type image_uuid: uuid.UUID
    :param original_image_uuid: UUID of the original image if this is a derivative.
    :type original_image_uuid: uuid.UUID
    :param generated_status: Status indicating if and how the image was AI-generated.
    :type generated_status: AIGeneratedStatus
    :param image_metadata: JSON blob containing image metadata.
    :type image_metadata: dict[str, Any]
    :param image_hash: Perceptual hash of the image content.
    :type image_hash: bytes
    :param user_annotation: Optional user-provided annotation for the image.
    :type user_annotation: str | None
    :param generated_annotation: Optional AI-generated annotation for the image.
    :type generated_annotation: str | None
    :param gallery_embedding: Related embedding vector for similarity search.
    :type gallery_embedding: GalleryEmbedding
    """

    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    image_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    original_image_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)
    generated_status: Mapped[AIGeneratedStatus] = mapped_column(Enum(AIGeneratedStatus))
    public: Mapped[bool] = mapped_column(BOOLEAN, default=False)

    image_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
    image_hash: Mapped[bytes] = mapped_column(ImageHash)

    user_annotation: Mapped[str | None] = mapped_column(String, nullable=True)
    generated_annotation: Mapped[str | None] = mapped_column(String, nullable=True)

    gallery_embedding: Mapped[GalleryEmbedding] = relationship("gallery_embedding", back_populates=__tablename__)

    __table_args__ = (
        Index("idx_image_uuid", "image_uuid"),
        Index("idx_image_hash", "image_hash"),
    )
