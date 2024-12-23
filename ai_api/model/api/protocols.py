"""Module containing protocol definitions for API service interfaces.

This module defines the protocol interfaces that API services must implement,
ensuring type safety and consistent API contracts. It includes protocols for:
- Inference service operations (embedding, detection, watermarking)
- Main API service operations (search, processing, health checks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from _bentoml_sdk.method import APIMethod
    from bentoml import Context
    from PIL import Image as PILImage

    from .image_search import ImageSearchResponse
    from .process import AddWatermarkRequest, AIGeneratedStatus
    from .query_search import SearchResponse


class InferenceServiceProto(Protocol):
    """Protocol defining the interface for ML inference operations.

    This protocol defines the required methods for the inference service,
    including operations for:
    - Text and image embedding generation
    - Image captioning
    - AI generation detection
    - Watermark operations

    All methods are asynchronous and support batch operations for efficiency.
    """

    readyz: APIMethod[[], Coroutine[Any, Any, dict[str, str]]]
    embed_text: APIMethod[
        [list[str]],
        Coroutine[Any, Any, list[list[float]]],
    ]
    embed_image: APIMethod[
        [list[PILImage.Image]],
        Coroutine[Any, Any, list[list[float]]],
    ]
    generate_caption: APIMethod[
        [list[PILImage.Image]],
        Coroutine[Any, Any, list[str]],
    ]
    detect_ai_generation: APIMethod[
        [list[PILImage.Image]],
        Coroutine[Any, Any, list[AIGeneratedStatus]],
    ]
    check_watermark: APIMethod[
        [list[PILImage.Image]],
        Coroutine[Any, Any, list[tuple[bool, str | None]]],
    ]
    check_ai_watermark: APIMethod[
        [list[PILImage.Image]],
        Coroutine[Any, Any, list[bool]],
    ]
    add_watermark: APIMethod[
        [list[AddWatermarkRequest]],
        Coroutine[Any, Any, list[PILImage.Image]],
    ]
    add_ai_watermark: APIMethod[
        [list[PILImage.Image]],
        Coroutine[Any, Any, list[PILImage.Image]],
    ]


class APIServiceProto(Protocol):
    """Protocol defining the interface for the main API service.

    This protocol defines the required attributes and methods for the main API
    service, including:
    - Health and readiness checks
    - Search operations (text and image-based)
    - Image processing operations

    The service coordinates between the inference service, database, and external
    services to provide the complete API functionality.

    Attributes:
        db_healthy: Flag indicating database health status.
        embedding_service: Reference to the inference service.
        ctx: BentoML request context.

    """

    db_healthy: bool
    embedding_service: InferenceServiceProto
    ctx: Context

    healthz: Callable[[], Coroutine[Any, Any, dict[str, str]]]

    readyz: Callable[[], Coroutine[Any, Any, dict[str, str]]]

    search_query: Callable[
        [Any, str, int, int],
        Coroutine[Any, Any, SearchResponse],
    ]

    search_image: Callable[
        [Any, str, int, int],
        Coroutine[Any, Any, ImageSearchResponse],
    ]

    process_image: APIMethod[
        [PILImage.Image, str, str, dict[str, Any], Context],
        Coroutine[Any, Any, None],
    ]
