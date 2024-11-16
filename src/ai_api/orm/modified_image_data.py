from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from sqlalchemy import String, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


if TYPE_CHECKING:
    from .image import Image


class ModifiedImageData(Base):
    __tablename__ = "image_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    original_image_id: Mapped[int] = mapped_column(Integer)
    image_metadata: Mapped[dict] = mapped_column(JSONB)
    md5_hash: Mapped[str] = mapped_column(String(32))

    image: Mapped["Image"] = relationship("Image", back_populates="image_data")
