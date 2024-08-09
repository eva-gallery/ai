import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import types
from sqlalchemy.orm import mapped_column, Mapped

from ai_api.orm import Base


class Image(Base):
    __tablename__ = "image"

    id: Mapped[uuid.UUID] = mapped_column(types.Uuid, primary_key=True, nullable=False, init=False)
    image_vector: Mapped[list[float]] = mapped_column(Vector(1000), nullable=False)
    image_caption: Mapped[str] = mapped_column(types.VARCHAR(512), nullable=False)
