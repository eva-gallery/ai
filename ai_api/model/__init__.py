"""Module containing data models for API requests and responses.

This module provides a collection of data models used throughout the API, including:
- Request and response models for all API endpoints
- Status enums and type definitions
- Service protocols and interfaces
- Data validation and serialization logic

The models are built using Pydantic to ensure type safety and validation
at runtime, with clear error messages for invalid data.
"""

from ai_api.model.api import (
    AddWatermarkRequest,
    AIGeneratedStatus,
    APIServiceProto,
    BackendPatchRequest,
    ImageDuplicateStatus,
    ImageSearchRequest,
    ImageSearchResponse,
    InferenceServiceProto,
    ProcessImageRequest,
    RawImageSearchRequest,
    SearchRequest,
    SearchResponse,
)

__all__ = [
    "AIGeneratedStatus",
    "APIServiceProto",
    "AddWatermarkRequest",
    "BackendPatchRequest",
    "ImageDuplicateStatus",
    "ImageSearchRequest",
    "ImageSearchResponse",
    "InferenceServiceProto",
    "ProcessImageRequest",
    "RawImageSearchRequest",
    "SearchRequest",
    "SearchResponse",
]
