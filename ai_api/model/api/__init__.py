"""Module containing Pydantic models for the API endpoints.

This module provides a collection of Pydantic models used across the API endpoints for:
- Image processing and watermarking
- Image and text-based search operations
- Duplicate detection and status tracking
- Service protocols and interfaces

The models ensure type safety and validation for API requests and responses.
"""

from ai_api.model.api.embed import EmbedRequest
from ai_api.model.api.image_search import ImageSearchRequest, ImageSearchResponse, RawImageSearchRequest
from ai_api.model.api.process import AddWatermarkRequest, AIGeneratedStatus, BackendPatchRequest, ProcessImageRequest
from ai_api.model.api.protocols import APIServiceProto, InferenceServiceProto
from ai_api.model.api.query_search import SearchRequest, SearchResponse
from ai_api.model.api.status import ImageDuplicateStatus

__all__ = [
    "AIGeneratedStatus",
    "APIServiceProto",
    "AddWatermarkRequest",
    "BackendPatchRequest",
    "EmbedRequest",
    "ImageDuplicateStatus",
    "ImageSearchRequest",
    "ImageSearchResponse",
    "InferenceServiceProto",
    "ProcessImageRequest",
    "RawImageSearchRequest",
    "SearchRequest",
    "SearchResponse",
]
