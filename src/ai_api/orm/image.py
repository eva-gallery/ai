from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


if TYPE_CHECKING:
    from .image_data import ImageData
    from .annotation import Annotation
    from .gallery_embedding import GalleryEmbedding


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_data_id: Mapped[int] = mapped_column(ForeignKey("image_data.id"))
    annotation_id: Mapped[int] = mapped_column(ForeignKey("annotation.id"))

    image_data: Mapped[ImageData] = relationship("ImageData", back_populates="images")
    annotation: Mapped[Annotation] = relationship("Annotation", back_populates="images")
    gallery_embeddings: Mapped[list[GalleryEmbedding]] = relationship("GalleryEmbedding", back_populates="image")
