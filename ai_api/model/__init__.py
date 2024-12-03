"""The module with Pydantic models for the API's requests and responses."""

from .api import (
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
