from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


if TYPE_CHECKING:
    from .image import Image


class ImageData(Base):
    __tablename__ = "image_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    original_image_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    modified_image_uuid: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    image_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    md5_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)

    images: Mapped[list["Image"]] = relationship("Image", back_populates="image_data")
