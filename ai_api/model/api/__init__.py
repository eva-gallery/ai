"""The module with Pydantic models for the API."""

from .embed import EmbedRequest, EmbedResponse
from .image_search import ImageSearchRequest, ImageSearchResponse, RawImageSearchRequest
from .process import AIGeneratedStatus, BackendPatchRequest, ProcessImageRequest
from .query_search import SearchRequest, SearchResponse
from .service import APIServiceProto, InferenceServiceProto
from .status import ImageDuplicateStatus

__all__ = [
    "AIGeneratedStatus",
    "APIServiceProto",
    "BackendPatchRequest",
    "EmbedRequest",
    "EmbedResponse",
    "ImageDuplicateStatus",
    "ImageSearchRequest",
    "ImageSearchResponse",
    "InferenceServiceProto",
    "ProcessImageRequest",
    "RawImageSearchRequest",
    "SearchRequest",
    "SearchResponse",
]
