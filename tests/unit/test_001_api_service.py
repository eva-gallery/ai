"""Tests for the API service."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import bentoml
import pytest
from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession

from ai_api.main import APIService, APIServiceProto
from ai_api.model.api.process import AIGeneratedStatus

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(scope="module")
def mock_context() -> bentoml.Context:
    """Create a mock BentoML context for testing.
    
    :returns: Mock context with response and state attributes.
    """
    context = MagicMock()
    context.response = MagicMock()
    context.response.status_code = 201
    context.state = {"queued_processing": 0}
    context.request = MagicMock()
    context.request.headers = {}
    return context


@pytest.fixture(scope="module")
def sample_image() -> PILImage.Image:
    """Create a sample PIL Image for testing.
    
    :returns: Sample RGB image.
    """
    return PILImage.new("RGB", (100, 100))


@pytest.fixture(scope="module")
def mock_embedding_service() -> MagicMock:
    """Create a mock embedding service for testing.
    
    :returns: A mock embedding service.
    """
    service = MagicMock()
    service.embed_text.return_value = [0.1, 0.2, 0.3]
    service.embed_image.return_value = [0.1, 0.2, 0.3]
    service.check_ai_watermark.return_value = [False]
    service.detect_ai_generation.return_value = [AIGeneratedStatus.NOT_GENERATED]
    return service


@pytest.fixture(scope="module")
def mock_db_session() -> AsyncMock:
    """Create a mock database session for testing.
    
    :returns: A mock database session.
    """
    session = AsyncMock(spec=AsyncSession)
    
    # Create test UUIDs
    test_uuids = [str(uuid.uuid4()) for _ in range(3)]
    
    # Mock the execute result for search queries
    mock_result = AsyncMock()
    mock_result.scalars.return_value.all.return_value = test_uuids
    
    # Set up the session's execute to return our mock result
    session.execute.return_value = mock_result
    
    return session


@pytest.fixture(scope="module")
def mock_aiohttp_session() -> AsyncMock:
    """Create a mock aiohttp session for testing.
    
    :returns: A mock aiohttp session.
    """
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.patch.return_value.__aenter__.return_value.status = 200
    return session


@pytest.fixture(scope="module")
async def api_service(
    mock_embedding_service: MagicMock,
    mock_db_session: AsyncMock,
    mock_aiohttp_session: AsyncMock,
) -> AsyncGenerator[APIServiceProto, None]:
    """Create an API service instance for testing.
    
    :param mock_embedding_service: Mock embedding service.
    :param mock_db_session: Mock database session.
    :param mock_aiohttp_session: Mock aiohttp session.
    :returns: An API service instance.
    """
    # Create a patched version of APIService
    with patch("ai_api.main.AIOPostgres") as mock_postgres:
        mock_postgres.return_value.__aenter__.return_value = mock_db_session
        
        # Create and configure service
        service = APIService()
        service.embedding_service = mock_embedding_service  # Directly inject mock service
        service._db_ready = True  # type: ignore[attr-defined]
        
        # Mock the image cache function
        async def mock_image_cache(image_id: int) -> tuple[float, ...]:
            return (0.1, 0.2, 0.3)
        
        service._get_image_embedding = mock_image_cache  # type: ignore[attr-defined]
        
        yield service


@pytest.mark.asyncio
async def test_search_image(api_service: APIServiceProto) -> None:
    """Test searching by image.
    
    :param api_service: The API service instance.
    """
    test_uuid = uuid.uuid4()
    result = await api_service.search_image(str(test_uuid), 10, 0)  # type: ignore[call-arg]
    
    assert isinstance(result.image_uuid, list)


@pytest.mark.asyncio
async def test_process_image(api_service: APIServiceProto, mock_context: bentoml.Context, sample_image: PILImage.Image) -> None:
    """Test processing a new image."""
    # Mock context state
    mock_context.state = {"queued_processing": 0}

    await api_service.process_image(
        sample_image,
        str(uuid.uuid4()),
        AIGeneratedStatus.NOT_GENERATED.value,
        {"test": "metadata"},
        mock_context,
    )

    assert mock_context.response.status_code == 201


@pytest.mark.asyncio
async def test_process_image_with_duplicate(api_service: APIServiceProto, mock_context: bentoml.Context) -> None:
    """Test processing a duplicate image using perceptual hashing."""
    # Mock context state
    mock_context.state = {"queued_processing": 0}

    # Mock perceptual hash similarity check
    api_service.embedding_service.embed_image = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    test_image = PILImage.new("RGB", (100, 100))

    await api_service.process_image(
        test_image,
        str(uuid.uuid4()),
        AIGeneratedStatus.NOT_GENERATED.value,
        {"test": "metadata"},
        mock_context,
    )

    assert mock_context.response.status_code == 201


@pytest.mark.asyncio
async def test_process_image_with_ai_watermark(api_service: APIServiceProto, mock_context: bentoml.Context) -> None:
    """Test processing an image with AI watermark."""
    # Mock context state
    mock_context.state = {"queued_processing": 0}

    # Mock AI watermark detection
    api_service.embedding_service.check_ai_watermark = AsyncMock(return_value=[True])

    test_image = PILImage.new("RGB", (100, 100))

    await api_service.process_image(
        test_image,
        str(uuid.uuid4()),
        AIGeneratedStatus.NOT_GENERATED.value,
        {"test": "metadata"},
        mock_context,
    )

    assert mock_context.response.status_code == 201
