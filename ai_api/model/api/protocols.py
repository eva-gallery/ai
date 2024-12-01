"""Protocol definitions for API services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import uuid
    from collections.abc import Coroutine

    from _bentoml_sdk.method import APIMethod
    from bentoml import Context
    from PIL import Image as PILImage

    from .image_search import ImageSearchResponse
    from .process import AddWatermarkRequest, AIGeneratedStatus
    from .query_search import SearchResponse


class InferenceServiceProto(Protocol):
    """Protocol defining the interface for the inference service."""

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
    """Protocol defining the interface for the API service."""

    db_healthy: bool
    embedding_service: InferenceServiceProto
    ctx: Context

    healthz: APIMethod[
        [Context],
        Coroutine[Any, Any, dict[str, str]],
    ]

    readyz: APIMethod[
        [Context],
        Coroutine[Any, Any, dict[str, str]],
    ]

    search_query: APIMethod[
        [str, int, int],
        Coroutine[Any, Any, SearchResponse],
    ]

    search_image: APIMethod[
        [uuid.UUID, int, int],
        Coroutine[Any, Any, ImageSearchResponse],
    ]

    process_image: APIMethod[
        [PILImage.Image, str, str, dict[str, Any], Context],
        Coroutine[Any, Any, None],
    ]
