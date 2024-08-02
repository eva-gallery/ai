from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import UUID, VARCHAR

from ai_api.orm import base


class Image(base):
    __tablename__ = "image"

    id = Column(UUID, primary_key=True, nullable=False)
    image_vector = Column(Vector(1000), nullable=False)
    image_caption = Column(VARCHAR(500), nullable=False)
