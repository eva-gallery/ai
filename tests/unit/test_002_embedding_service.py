"""Tests for the embedding service."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Generator
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from ai_api import settings
from ai_api.model.api.process import AIGeneratedStatus
from ai_api.services.inference_service import InferenceService

if TYPE_CHECKING:
    from ai_api.model.api.protocols import InferenceServiceProto


# Skip all tests in this module if CI=true
pytestmark = pytest.mark.skipif(
    bool(os.getenv("CI", None)),
    reason="Tests skipped in CI environment",
)


@pytest.fixture(scope="module")
def service() -> Generator[InferenceServiceProto, None, None]:
    """Create a test instance of the inference service.
    
    :returns: Test instance of InferenceService.
    """
    with patch("bentoml.server_context") as mock_context:
        mock_context.worker_index = 1  # Mock worker index to be 1
        yield InferenceService()


@pytest.mark.asyncio
async def test_readyz(service: InferenceServiceProto) -> None:
    """Test readiness check endpoint."""
    result = await service.readyz()
    assert result == {"status": "ready"}, f"Expected {'status': 'ready'} but got {result}"


@pytest.mark.asyncio
async def test_embed_text(service: InferenceServiceProto) -> None:
    """Test text embedding generation."""
    texts = ["test text"]
    result = (await service.embed_text(texts))[0]
    assert isinstance(result, list), f"Expected list but got {type(result)}"
    assert len(result) == settings.model.embedding.dimension, f"Expected length {settings.model.embedding.dimension} but got {len(result)}"


@pytest.mark.asyncio
async def test_embed_image(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test image embedding generation."""
    images = [sample_image]
    result = (await service.embed_image(images))[0]
    assert isinstance(result, list), f"Expected list but got {type(result)}"
    assert len(result) == settings.model.embedding.dimension, f"Expected length {settings.model.embedding.dimension} but got {len(result)}"


@pytest.mark.asyncio
async def test_generate_caption(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test image caption generation."""
    images = [sample_image]
    result = (await service.generate_caption(images))[0]
    assert isinstance(result, str), f"Expected str but got {type(result)}"


@pytest.mark.asyncio
async def test_detect_ai_generation(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test AI generation detection."""
    images = [sample_image]
    result = (await service.detect_ai_generation(images))[0]
    assert result in AIGeneratedStatus, f"Expected {AIGeneratedStatus} but got {result}"


@pytest.mark.asyncio
async def test_detect_ai_generation_not_generated(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test AI generation detection for non-AI images."""
    images = [sample_image]
    result = (await service.detect_ai_generation(images))[0]
    assert result in AIGeneratedStatus, f"Expected {AIGeneratedStatus} but got {result}"


@pytest.mark.asyncio
async def test_check_ai_watermark(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test AI watermark detection."""
    images = [sample_image]
    result = (await service.check_ai_watermark(images))[0]
    assert isinstance(result, bool), f"Expected bool but got {type(result)}"


@pytest.mark.asyncio
async def test_add_ai_watermark(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test adding AI watermark to images."""
    images = [sample_image]
    # Generate caption for the prompt
    prompts = await service.generate_caption(images)
    result = (await service.add_ai_watermark(images, prompts))[0]
    assert isinstance(result, PILImage.Image), f"Expected PIL.Image but got {type(result)}"
    assert result.size == sample_image.size, f"Expected size {sample_image.size} but got {result.size}"


@pytest.mark.asyncio
async def test_ai_watermark_round_trip(service: InferenceServiceProto, sample_image: PILImage.Image) -> None:
    """Test that an added AI watermark can be detected.
    
    This test verifies that when we add an AI watermark to an image,
    we can then detect that watermark in the resulting image.
    
    :param service: The inference service instance.
    :param sample_image: Sample test image.
    """
    images = [sample_image]
    # Generate caption for the prompt
    prompts = await service.generate_caption(images)
    
    # First verify the original image has no watermark
    original_has_watermark = (await service.check_ai_watermark(images))[0]
    assert not original_has_watermark, "Original image should not have a watermark"

    # Add watermark to the image using the caption as prompt
    watermarked_images = await service.add_ai_watermark(images, prompts)
    assert len(watermarked_images) == 1, "Should get one watermarked image back"
    watermarked_image = watermarked_images[0]

    # Verify the watermark can be detected
    has_watermark = (await service.check_ai_watermark([watermarked_image]))[0]
    assert has_watermark, "Watermarked image should be detected as having a watermark"


@pytest.mark.asyncio
async def test_ai_watermark_image_quality(service: InferenceServiceProto) -> None:
    """Test that watermarking doesn't significantly degrade image quality.
    
    This test verifies that the watermarked image remains visually similar
    to the original image using PSNR and SSIM metrics.
    """
    # Get a real test image from picsum
    import requests
    from io import BytesIO
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    response = requests.get("https://picsum.photos/200")
    test_image = PILImage.open(BytesIO(response.content))
    
    # Generate caption for the image
    prompts = await service.generate_caption([test_image])
    
    # Add watermark
    watermarked_image = (await service.add_ai_watermark([test_image], prompts))[0]
    
    # Verify basic image properties remain intact
    assert watermarked_image.size == test_image.size, "Image size should not change"
    assert watermarked_image.mode == test_image.mode, "Image mode should not change"
    
    # Convert images to numpy arrays for quality metrics
    img1 = np.array(test_image.convert("RGB"))
    img2 = np.array(watermarked_image.convert("RGB"))
    
    # Calculate PSNR - higher is better, typical good values are 25+ for watermarked images
    psnr_value = psnr(img1, img2)
    assert psnr_value > 18, f"PSNR too low ({psnr_value}), indicating significant image degradation"
    
    # Calculate SSIM - ranges from -1 to 1, higher is better
    ssim_value = ssim(img1, img2, channel_axis=2)  # channel_axis=2 for RGB images
    assert ssim_value > 0.75, f"SSIM too low ({ssim_value}), indicating significant structural changes"


@pytest.mark.asyncio
async def test_ai_watermark_no_false_positives(service: InferenceServiceProto) -> None:
    """Test that watermark detection doesn't have false positives.
    
    This test verifies that various real images are correctly identified 
    as not having a watermark.
    """
    import requests
    from io import BytesIO
    
    # Test with multiple real images
    test_images = []
    for _ in range(3):  # Test 3 different random images
        response = requests.get("https://picsum.photos/200")
        test_images.append(PILImage.open(BytesIO(response.content)))
    
    # Check each image
    results = await service.check_ai_watermark(test_images)
    for i, has_watermark in enumerate(results):
        assert not has_watermark, f"Real image {i} falsely detected as having watermark"
