"""Module containing the SQLAlchemy ORM model for image data.

This module defines the database schema and relationships for storing image data,
including metadata, embeddings, and various status flags.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import UUID, Enum, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_api.model.api import AIGeneratedStatus
from ai_api.orm.base import Base

if TYPE_CHECKING:
    from ai_api.orm.gallery_embedding import GalleryEmbedding


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
    :type image_hash: str
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

    image_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB)
    image_hash: Mapped[str] = mapped_column(String(8))

    user_annotation: Mapped[str | None] = mapped_column(String, nullable=True)
    generated_annotation: Mapped[str | None] = mapped_column(String, nullable=True)

    gallery_embedding: Mapped[GalleryEmbedding] = relationship("gallery_embedding", back_populates=__tablename__)

    __table_args__ = (
        Index("idx_image_uuid", "image_uuid"),
        Index("idx_image_hash", "image_hash"),
    )
