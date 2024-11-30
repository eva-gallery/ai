"""The ORM model for the image."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import UUID, Enum, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..model.api import AIGeneratedStatus  # noqa: TID252
from .base import Base

if TYPE_CHECKING:
    from .gallery_embedding import GalleryEmbedding


class Image(Base):
    """The ORM model for the image."""

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
