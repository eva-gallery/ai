from unittest.mock import Mock, patch

import torch
import pytest

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus
from ai_api.services.inference_service import InferenceService

@pytest.fixture(scope="module")
def service():
    return InferenceService()

@pytest.mark.asyncio
async def test_readyz(service, mock_context):
    result = await service.readyz(mock_context)
    assert result == {"status": "ready"}, f"Expected {'status': 'ready'} but got {result}"

@pytest.mark.asyncio
async def test_embed_text(service, mock_numpy_array):
    texts = ["test text"]
    result = (await service.embed_text(texts))[0]  # type: ignore
    assert isinstance(result, list), f"Expected list but got {type(result)}"
    assert len(result) == settings.model.embedding.dimension, f"Expected length {settings.model.embedding.dimension} but got {len(result)}"

# Using sample_image from conftest.py
@pytest.mark.asyncio
async def test_embed_image(service, sample_image):
    images = [sample_image]
    result = (await service.embed_image(images))[0]  # type: ignore
    assert isinstance(result, list), f"Expected list but got {type(result)}"
    assert len(result) == settings.model.embedding.dimension, f"Expected length {settings.model.embedding.dimension} but got {len(result)}"

@pytest.mark.asyncio
async def test_generate_caption(service, sample_image):
    images = [sample_image]
    result = (await service.generate_caption(images))[0]  # type: ignore
    assert isinstance(result, str), f"Expected str but got {type(result)}"

@pytest.mark.asyncio
async def test_detect_ai_generation(service, sample_image):
    images = [sample_image]
    result = (await service.detect_ai_generation(images))[0]  # type: ignore
    assert result in AIGeneratedStatus, f"Expected {AIGeneratedStatus} but got {result}"

@pytest.mark.asyncio
async def test_detect_ai_generation_not_generated(service, sample_image):
    images = [sample_image]
    result = (await service.detect_ai_generation(images))[0]  # type: ignore
    assert result in AIGeneratedStatus, f"Expected {AIGeneratedStatus} but got {result}"

@pytest.mark.asyncio
async def test_check_watermark(service, sample_image):
    images = [sample_image]
    result = (await service.check_watermark(images))[0]  # type: ignore
    assert isinstance(result, tuple), f"Expected tuple but got {type(result)}"
    assert len(result) == 2, f"Expected tuple of length 2 but got length {len(result)}"
    assert isinstance(result[0], bool), f"Expected bool but got {type(result[0])}"
    assert isinstance(result[1], (str, type(None))), f"Expected str or None but got {type(result[1])}"

@pytest.mark.asyncio
async def test_check_ai_watermark(service, sample_image):    
    images = [sample_image]
    result = (await service.check_ai_watermark(images))[0]  # type: ignore
    assert isinstance(result, bool), f"Expected bool but got {type(result)}"
