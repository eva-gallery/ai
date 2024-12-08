"""Module containing models for text-based image search functionality.

This module provides request and response models for performing text-based
searches against the image database, supporting pagination and result count control.
"""

from __future__ import annotations

from bentoml import Field, IODescriptor
from pydantic import UUID4


class SearchRequest(IODescriptor):
    """Request model for text-based image search.
    
    This class represents a search request that allows users to find images
    using text queries with pagination support.
    
    :param query: The text query to search for matching images.
    :type query: str
    :param count: Number of results to return per page.
    :type count: int
    :param page: Zero-based page number for pagination.
    :type page: int
    """

    query: str = Field(..., description="Text query to search for")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")


class SearchResponse(IODescriptor):
    """Response model for text-based image search.
    
    This class represents the search results containing UUIDs of images
    that match the search criteria.
    
    :param image_uuid: List of UUIDs of images matching the search query.
    :type image_uuid: list[UUID4]
    :returns: A list of image UUIDs matching the search criteria.
    """

    image_uuid: list[UUID4] = Field(..., description="UUIDs of the most similar images")
