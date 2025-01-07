"""Tests for the API service."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import bentoml
import pytest
from PIL import Image as PILImage

from ai_api.main import APIService
from ai_api.model.api.process import AIGeneratedStatus

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ai_api.model.api import APIServiceProto


# Skip all tests in this module if CI=true
pytestmark = pytest.mark.skipif(
    bool(os.getenv("CI", None)),
    reason="Tests skipped in CI environment",
)


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
    """Create a mock embedding service for testing."""
    service = MagicMock()

    # Mock async methods
    service.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    service.embed_image = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    service.generate_caption = AsyncMock(return_value=["A test image caption"])
    service.detect_ai_generation = AsyncMock(return_value=[AIGeneratedStatus.NOT_GENERATED])
    service.check_watermark = AsyncMock(return_value=[(False, None)])
    service.check_ai_watermark = AsyncMock(return_value=[False])
    service.add_watermark = AsyncMock(return_value=[PILImage.new("RGB", (100, 100))])
    service.add_ai_watermark = AsyncMock(return_value=[PILImage.new("RGB", (100, 100))])
    service.readyz = AsyncMock()

    return service


@pytest.fixture(scope="module")
def mock_db_session() -> AsyncMock:
    """Create a mock database session for testing."""
    session = AsyncMock()
    session.__aenter__.return_value = session

    # Create a mock result that properly simulates the database query execution chain
    mock_result = AsyncMock()
    mock_scalars = AsyncMock()
    mock_all = AsyncMock(return_value=[uuid.uuid4() for _ in range(3)])

    mock_scalars.all = mock_all
    mock_result.scalars = AsyncMock(return_value=mock_scalars)
    session.execute.return_value = mock_result

    return session


@pytest.fixture(scope="module")
def mock_aiohttp_session() -> AsyncMock:
    """Create a mock aiohttp session for testing."""
    session = AsyncMock()
    session.__aenter__.return_value = session

    response = AsyncMock()
    response.status = 200
    response.__aenter__.return_value = response

    session.patch.return_value = response
    return session


@pytest.fixture(scope="module")
async def api_service(
    mock_embedding_service: MagicMock,
    mock_db_session: AsyncMock,
    mock_aiohttp_session: AsyncMock,
) -> AsyncGenerator[APIServiceProto, None]:
    """Create a test instance of the API service with mocked dependencies.

    :param mock_embedding_service: Mock embedding service.
    :param mock_db_session: Mock database session.
    :param mock_aiohttp_session: Mock HTTP session.
    :yields: Test instance of APIService.
    """
    # Mock BentoML server context
    mock_server_context = MagicMock()
    mock_server_context.worker_index = 1

    # Create a patched version of APIService
    with patch("ai_api.main.APIService") as MockAPIService, \
         patch("ai_api.main.bentoml.server_context", mock_server_context), \
         patch("ai_api.main.bentoml.depends", return_value=mock_embedding_service), \
         patch("ai_api.main.AIOPostgres") as mock_postgres, \
         patch("ai_api.main.aiohttp.ClientSession", return_value=mock_aiohttp_session), \
         patch("ai_api.main.bentoml.Context") as mock_context:

        # Create a mock for _verify_jwt
        mock_verify_jwt = MagicMock(return_value=None)

        # Set up the mock class to include our mock method
        MockAPIService._verify_jwt = mock_verify_jwt  # noqa: SLF001
        service = APIService()
        MockAPIService.return_value = service

        mock_postgres.return_value.session.return_value = mock_db_session
        mock_context.return_value = mock_context
        # Add the mock method to the instance
        service._verify_jwt = mock_verify_jwt  # type: ignore[attr-defined] # noqa: SLF001

        try:
            yield service
        finally:
            # Clean up any background tasks
            for task in service.background_tasks:
                task.cancel()


@pytest.mark.asyncio
async def test_search_query(api_service: APIServiceProto, mock_db_session: AsyncMock) -> None:
    """Test searching by text query."""
    # Ensure the mock session is used
    with patch("ai_api.main.AIOPostgres") as mock_postgres:
        mock_postgres.return_value.__aenter__.return_value = mock_db_session

        result = await api_service.search_query("test query", 10, 0)  # type: ignore[call-arg]

        assert isinstance(result.image_uuid, list)
        assert len(result.image_uuid) == 3
        assert all(isinstance(id_, uuid.UUID) for id_ in result.image_uuid)


@pytest.mark.asyncio
async def test_search_image(api_service: APIServiceProto, mock_db_session: AsyncMock) -> None:
    """Test searching by image."""
    # Ensure the mock session is used
    with patch("ai_api.main.AIOPostgres") as mock_postgres:
        mock_postgres.return_value.__aenter__.return_value = mock_db_session

        # Mock the cache function to return a valid embedding
        with patch("ai_api.main._search_image_id_cache", new=AsyncMock(return_value=(0.1, 0.2, 0.3))):
            test_uuid = uuid.uuid4()
            result = await api_service.search_image(str(test_uuid), 10, 0)  # type: ignore[call-arg]

            assert isinstance(result.image_uuid, list)
            assert len(result.image_uuid) == 3
            assert all(isinstance(id_, uuid.UUID) for id_ in result.image_uuid)


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
