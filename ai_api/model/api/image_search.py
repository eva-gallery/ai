"""The module with Pydantic models for the image search API."""

from __future__ import annotations

import uuid

from bentoml import Field, IODescriptor
from PIL import Image as PILImage


class ImageSearchRequest(IODescriptor):
    """The request to search for similar images."""

    image_uuid: uuid.UUID = Field(..., description="UUID of image to find similar images to")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")


class ImageSearchResponse(IODescriptor):
    """The response to search for similar images."""

    image_uuid: list[uuid.UUID] = Field(..., description="UUIDs of the most similar images")


class RawImageSearchRequest(IODescriptor):
    """The request to search for similar images."""

    image: PILImage.Image = Field(..., description="Raw image bytes")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")
