from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


if TYPE_CHECKING:
    from .modified_image_data import ModifiedImageData
    from .annotation import Annotation
    from .gallery_embedding import GalleryEmbedding


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_data_id: Mapped[int] = mapped_column(ForeignKey("image_data.id"))
    annotation_id: Mapped[int] = mapped_column(ForeignKey("annotation.id"))

    image_data: Mapped[ModifiedImageData] = relationship("ModifiedImageData", back_populates="images")
    annotation: Mapped[Annotation] = relationship("Annotation", back_populates="image")
    gallery_embedding: Mapped[GalleryEmbedding] = relationship("GalleryEmbedding", back_populates="image")
