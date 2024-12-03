"""The module with Pydantic models for the API."""

from .image_search import ImageSearchRequest, ImageSearchResponse, RawImageSearchRequest
from .process import AddWatermarkRequest, AIGeneratedStatus, BackendPatchRequest, ProcessImageRequest
from .protocols import APIServiceProto, InferenceServiceProto
from .query_search import SearchRequest, SearchResponse
from .status import ImageDuplicateStatus

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
