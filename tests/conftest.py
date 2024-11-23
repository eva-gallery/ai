from __future__ import annotations

import os
import sys
from typing import Generator, Any
from unittest.mock import Mock, patch

import pytest
import numpy as np
from PIL import Image

# Set environment before any tests run
os.environ["ENV_FOR_DYNACONF"] = "testing"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

@pytest.fixture(scope="session", autouse=True)
def setup_test_env() -> Generator[None, None, None]:
    """Setup any test environment variables and patch logger."""
    with patch.dict(os.environ, {"ENV_FOR_DYNACONF": "testing"}), \
         patch('ai_api.util.logger.get_logger') as mock_logger, \
         patch('loguru.logger.debug'), \
         patch('loguru.logger.info'), \
         patch('loguru.logger.warning'), \
         patch('loguru.logger.error'):
        # Create a simple print logger
        def print_log(*args, **kwargs):
            print(*args, file=sys.stdout)
        mock_logger.return_value.debug = print_log
        mock_logger.return_value.info = print_log
        mock_logger.return_value.warning = print_log
        mock_logger.return_value.error = print_log
        
        # Verify logging works before running any tests
        try:
            mock_logger.return_value.info("Test logging setup successful")
            from loguru import logger
            logger.info("Test loguru logging setup successful")
        except Exception as e:
            pytest.fail(f"Logging setup failed: {str(e)}")
            
        yield

@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    """Create a sample PIL Image for testing."""
    return Image.new('RGB', (256, 256))

@pytest.fixture(scope="session")
def mock_torch_cuda() -> Generator[Mock, None, None]:
    """Mock torch.cuda.is_available to always return False for CPU testing."""
    with patch('torch.cuda.is_available', return_value=False) as mock:
        yield mock

@pytest.fixture(scope="session")
def mock_numpy_array() -> np.ndarray:
    """Create a sample numpy array for embeddings."""
    return np.array([[0.1, 0.2, 0.3]])

@pytest.fixture(scope="session")
def mock_context() -> Mock:
    """Create a mock context with response for API testing."""
    context = Mock()
    context.response = Mock()
    context.response.status_code = None
    return context

def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", 
        "asyncio: mark test as async/await test"
    )
