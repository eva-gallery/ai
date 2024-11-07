from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base


if TYPE_CHECKING:
    from .image import Image


class Annotation(Base):
    __tablename__ = "annotation"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_annotation: Mapped[str | None] = mapped_column(String, nullable=True)
    generated_annotation: Mapped[str] = mapped_column(String)

    images: Mapped[list["Image"]] = relationship("Image", back_populates="annotation")
