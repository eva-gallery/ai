from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus
from ai_api.services.inference_service import InferenceService

if TYPE_CHECKING:
    from ai_api.model.api.service import InferenceServiceProto


# Skip all tests in this module if CI=true
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Tests skipped in CI environment",
)


@pytest.fixture(scope="module")
def service() -> InferenceServiceProto:
    with patch("bentoml.server_context") as mock_context:
        mock_context.worker_index = 1  # Mock worker index to be 1
        return InferenceService()

@pytest.mark.asyncio
async def test_readyz(service: InferenceServiceProto) -> None:
    result = await service.readyz()
    assert result == {"status": "ready"}, f"Expected {'status': 'ready'} but got {result}"

@pytest.mark.asyncio
async def test_embed_text(service: InferenceServiceProto) -> None:
    texts = ["test text"]
    result = (await service.embed_text(texts))[0]
    assert isinstance(result, list), f"Expected list but got {type(result)}"
    assert len(result) == settings.model.embedding.dimension, f"Expected length {settings.model.embedding.dimension} but got {len(result)}"

# Using sample_image from conftest.py
@pytest.mark.asyncio
async def test_embed_image(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    images = [sample_image]
    result = (await service.embed_image(images))[0]
    assert isinstance(result, list), f"Expected list but got {type(result)}"
    assert len(result) == settings.model.embedding.dimension, f"Expected length {settings.model.embedding.dimension} but got {len(result)}"

@pytest.mark.asyncio
async def test_generate_caption(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    images = [sample_image]
    result = (await service.generate_caption(images))[0]
    assert isinstance(result, str), f"Expected str but got {type(result)}"

@pytest.mark.asyncio
async def test_detect_ai_generation(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    images = [sample_image]
    result = (await service.detect_ai_generation(images))[0]
    assert result in AIGeneratedStatus, f"Expected {AIGeneratedStatus} but got {result}"

@pytest.mark.asyncio
async def test_detect_ai_generation_not_generated(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    images = [sample_image]
    result = (await service.detect_ai_generation(images))[0]
    assert result in AIGeneratedStatus, f"Expected {AIGeneratedStatus} but got {result}"

@pytest.mark.asyncio
async def test_check_watermark(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    images = [sample_image]
    result = (await service.check_watermark(images))[0]
    assert isinstance(result, tuple), f"Expected tuple but got {type(result)}"
    assert len(result) == 2, f"Expected tuple of length 2 but got length {len(result)}"
    assert isinstance(result[0], bool), f"Expected bool but got {type(result[0])}"
    assert isinstance(result[1], (str, type(None))), f"Expected str or None but got {type(result[1])}"

@pytest.mark.asyncio
async def test_check_ai_watermark(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    images = [sample_image]
    result = (await service.check_ai_watermark(images))[0]
    assert isinstance(result, bool), f"Expected bool but got {type(result)}"
