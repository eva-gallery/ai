from __future__ import annotations

import io
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage

from ai_api.main import APIService
from ai_api.model.api.process import AIGeneratedStatus


@pytest.fixture(scope="module")
def mock_embedding_service() -> MagicMock:
    service = MagicMock()
    
    # Mock async methods
    service.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    service.embed_image = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    service.generate_caption = AsyncMock(return_value=["A test image caption"])
    service.detect_ai_generation = AsyncMock(return_value=[AIGeneratedStatus.NOT_GENERATED])
    service.check_watermark = AsyncMock(return_value=[(False, None)])
    service.check_ai_watermark = AsyncMock(return_value=[False])
    service.add_watermark = AsyncMock(return_value=[PILImage.new('RGB', (100, 100))])
    service.add_ai_watermark = AsyncMock(return_value=[PILImage.new('RGB', (100, 100))])
    service.readyz = AsyncMock()
    
    return service

@pytest.fixture(scope="module")
def mock_db_session() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__.return_value = session
    
    # Create a mock result that properly simulates the database query execution chain
    mock_result = AsyncMock()
    mock_scalars = AsyncMock()
    mock_all = AsyncMock(return_value=[1, 2, 3])
    
    mock_scalars.all = mock_all
    mock_result.scalars = AsyncMock(return_value=mock_scalars)
    session.execute.return_value = mock_result
    
    return session

@pytest.fixture(scope="module")
def mock_aiohttp_session() -> AsyncMock:
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
    mock_aiohttp_session: AsyncMock
) -> AsyncGenerator[APIService, None]:  
    with patch('ai_api.main.bentoml.depends', return_value=mock_embedding_service), \
         patch('ai_api.main.AIOPostgres') as mock_postgres, \
         patch('ai_api.main.aiohttp.ClientSession', return_value=mock_aiohttp_session):
        
        mock_postgres.return_value.session.return_value = mock_db_session
        service = APIService()
        yield service

@pytest.mark.asyncio
async def test_healthz_healthy(api_service: APIService, mock_context: MagicMock) -> None:  
    api_service.db_healthy = True
    result = await api_service.healthz(mock_context)
    
    assert result == {"status": "healthy"}
    assert mock_context.response.status_code == 200

@pytest.mark.asyncio
async def test_healthz_unhealthy(api_service: APIService, mock_context: MagicMock) -> None:  
    api_service.db_healthy = False
    result = await api_service.healthz(mock_context)
    
    assert result == {"status": "unhealthy"}
    assert mock_context.response.status_code == 503

@pytest.mark.asyncio
async def test_readyz(api_service: APIService, mock_context: MagicMock) -> None:  
    # Set up the mock context state to return an integer
    mock_context.state.get.return_value = 0
    
    result = await api_service.readyz(mock_context)
    
    assert result == {"status": "ready"}
    assert mock_context.response.status_code == 200

@pytest.mark.asyncio
async def test_readyz_not_ready(api_service: APIService, mock_context: MagicMock) -> None:  
    # Mock a high number of queued processes
    mock_context.state.get.return_value = 1000
    
    result = await api_service.readyz(mock_context)
    
    assert result == {"status": "not ready"}
    assert mock_context.response.status_code == 503

@pytest.mark.asyncio
async def test_search_query(api_service: APIService, mock_db_session: AsyncMock) -> None:  
    # Ensure the mock session is used
    with patch('ai_api.main.AIOPostgres') as mock_postgres:
        mock_postgres.return_value.__aenter__.return_value = mock_db_session
        
        result = await api_service.search_query(query="test query", count=10, page=0)
        
        assert isinstance(result.image_id, list)
        assert result.image_id == [1, 2, 3]
        assert all(isinstance(id_, int) for id_ in result.image_id)

@pytest.mark.asyncio
async def test_search_image(api_service: APIService, mock_db_session: AsyncMock) -> None:  
    # Ensure the mock session is used
    with patch('ai_api.main.AIOPostgres') as mock_postgres:
        mock_postgres.return_value.__aenter__.return_value = mock_db_session
        
        # Mock the cache function to return a valid embedding
        with patch.object(api_service, '_search_image_id_cache', new=AsyncMock(return_value=(0.1, 0.2, 0.3))):
            result = await api_service.search_image(image_id=1, count=10, page=0)
            
            assert isinstance(result.image_id, list)
            assert result.image_id == [1, 2, 3]
            assert all(isinstance(id_, int) for id_ in result.image_id)

@pytest.mark.asyncio
async def test_process_image(api_service: APIService, mock_context: MagicMock, sample_image: PILImage.Image) -> None:  
    # Mock context state
    mock_context.state = {'queued_processing': 0}
    
    image_bytes = io.BytesIO()
    sample_image.save(image_bytes, format='PNG')
    
    await api_service.process_image(
        image=image_bytes.getvalue(),
        artwork_id=1,
        ai_generated_status=AIGeneratedStatus.NOT_GENERATED,
        metadata={"test": "metadata"},
        ctx=mock_context
    )
    
    assert mock_context.response.status_code == 201

@pytest.mark.asyncio
async def test_process_image_with_duplicate(api_service: APIService, mock_context: MagicMock) -> None:  
    # Mock context state
    mock_context.state = {'queued_processing': 0}
    
    # Mock duplicate detection
    api_service.embedding_service.check_watermark.return_value = [(True, "test_hash")]
    
    test_image = PILImage.new('RGB', (100, 100))
    image_bytes = io.BytesIO()
    test_image.save(image_bytes, format='PNG')
    
    await api_service.process_image(
        image=image_bytes.getvalue(),
        artwork_id=1,
        ai_generated_status=AIGeneratedStatus.NOT_GENERATED,
        metadata={"test": "metadata"},
        ctx=mock_context
    )
    
    assert mock_context.response.status_code == 201

@pytest.mark.asyncio
async def test_process_image_with_ai_watermark(api_service: APIService, mock_context: MagicMock) -> None:  
    # Mock context state
    mock_context.state = {'queued_processing': 0}
    
    # Mock AI watermark detection
    api_service.embedding_service.check_ai_watermark.return_value = [True]
    
    test_image = PILImage.new('RGB', (100, 100))
    image_bytes = io.BytesIO()
    test_image.save(image_bytes, format='PNG')
    
    await api_service.process_image(
        image=image_bytes.getvalue(),
        artwork_id=1,
        ai_generated_status=AIGeneratedStatus.NOT_GENERATED,
        metadata={"test": "metadata"},
        ctx=mock_context
    )
    
    assert mock_context.response.status_code == 201
