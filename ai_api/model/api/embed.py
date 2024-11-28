from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    image: list[bytes] = Field(..., description="List of images to embed")
    caption: list[str] = Field(..., description="List of captions for the images")
    metadata: list[dict[str, Any]] = Field(..., description="List of metadata for the images")


class EmbedResponse(BaseModel):
    image_id: list[int] = Field(..., description="List of image IDs")
