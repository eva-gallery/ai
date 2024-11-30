"""The module with Pydantic models for the embed API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bentoml import Field, IODescriptor

if TYPE_CHECKING:
    from PIL import Image as PILImage


class EmbedRequest(IODescriptor):
    """The request to embed images."""

    image: list[PILImage.Image] = Field(..., description="List of images to embed")
    caption: list[str] = Field(..., description="List of captions for the images")
    metadata: list[dict[str, Any]] = Field(..., description="List of metadata for the images")


class EmbedResponse(IODescriptor):
    """The response to embed images."""

    image_id: list[int] = Field(..., description="List of image IDs")
