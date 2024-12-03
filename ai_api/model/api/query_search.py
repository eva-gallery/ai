"""The module with Pydantic models for the query search API."""

from __future__ import annotations

from bentoml import Field, IODescriptor
from pydantic import UUID4


class SearchRequest(IODescriptor):
    """The request to search for images."""

    query: str = Field(..., description="Text query to search for")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")


class SearchResponse(IODescriptor):
    """The response to search for images."""

    image_uuid: list[UUID4] = Field(..., description="UUIDs of the most similar images")
