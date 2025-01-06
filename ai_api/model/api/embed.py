"""Module containing models for image embedding generation.

This module provides request models for generating embeddings from images,
which can be used for similarity search and other ML tasks.
"""

from bentoml import Field, IODescriptor
from PIL.Image import Image as PILImage


class EmbedRequest(IODescriptor):
    """Request model for generating image embeddings.

    This class represents a request to generate embeddings for a batch of images,
    which can be used for similarity search and other ML tasks.

    :param image: List of images to generate embeddings for.
    :type image: list[PILImage]
    :returns: The embeddings will be generated for each provided image.
    """

    image: list[PILImage] = Field(..., description="List of images to embed")

    class Config:
        arbitrary_types_allowed = True
