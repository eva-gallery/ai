"""Module containing models for image-based similarity search functionality.

This module provides request and response models for performing image similarity searches,
supporting both UUID-based and raw image-based queries with pagination.
"""

from __future__ import annotations

from bentoml import Field, IODescriptor
from PIL import Image as PILImage
from pydantic import UUID4


class ImageSearchRequest(IODescriptor):
    """Request model for UUID-based image similarity search.
    
    This class represents a search request that finds similar images based on
    a reference image identified by its UUID.
    
    :param image_uuid: UUID of the reference image to find similar images to.
    :type image_uuid: UUID4
    :param count: Number of results to return per page.
    :type count: int
    :param page: Zero-based page number for pagination.
    :type page: int
    """

    image_uuid: UUID4 = Field(..., description="UUID of image to find similar images to")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")


class ImageSearchResponse(IODescriptor):
    """Response model for image similarity search.
    
    This class represents the search results containing UUIDs of images
    that are visually similar to the query image.
    
    :param image_uuid: List of UUIDs of the most similar images found.
    :type image_uuid: list[UUID4]
    :returns: A list of image UUIDs ordered by similarity to the query image.
    """

    image_uuid: list[UUID4] = Field(..., description="UUIDs of the most similar images")


class RawImageSearchRequest(IODescriptor):
    """Request model for raw image-based similarity search.
    
    This class represents a search request that finds similar images based on
    a provided raw image file.
    
    :param image: Raw image data to find similar images to.
    :type image: PILImage.Image
    :param count: Number of results to return per page.
    :type count: int
    :param page: Zero-based page number for pagination.
    :type page: int
    """

    image: PILImage.Image = Field(..., description="Raw image bytes")
    count: int = Field(default=50, description="Number of results to return")
    page: int = Field(default=0, description="Page number to return")

    class Config:
        arbitrary_types_allowed = True
