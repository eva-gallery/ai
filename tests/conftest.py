"""Test configuration and fixtures."""

from __future__ import annotations

import os
import sys
from typing import Generator, Any
from unittest.mock import Mock, patch

import pytest
import numpy as np
import numpy.typing as npt
from PIL import Image
import requests
from io import BytesIO

# Set environment before any tests run
os.environ["ENV_FOR_DYNACONF"] = "testing"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

@pytest.fixture(scope="session", autouse=True)
def setup_test_env() -> Generator[None, None, None]:
    """Setup any test environment variables and patch logger.
    
    This fixture patches all logger methods to print to stdout for better
    visibility during testing.
    
    :yields: None after setting up the test environment.
    """
    def print_log(*args: Any, **kwargs: Any) -> None:
        """Print log messages with their kwargs.
        
        :param args: Positional arguments to print.
        :param kwargs: Keyword arguments to print.
        """
        if kwargs:
            # Format the message with kwargs if they exist
            if len(args) > 0 and isinstance(args[0], str):
                try:
                    print(args[0].format(**kwargs), file=sys.stdout)
                except KeyError:
                    print(*args, kwargs, file=sys.stdout)
            else:
                print(*args, kwargs, file=sys.stdout)
        else:
            print(*args, file=sys.stdout)

    with patch.dict(os.environ, {"ENV_FOR_DYNACONF": "testing"}), \
         patch("ai_api.util.logger.get_logger") as mock_logger, \
         patch("loguru.logger.debug"), \
         patch("loguru.logger.info"), \
         patch("loguru.logger.warning"), \
         patch("loguru.logger.error"):
        
        # Set up all logger methods to use print_log
        mock_logger.return_value.debug = print_log
        mock_logger.return_value.info = print_log
        mock_logger.return_value.warning = print_log
        mock_logger.return_value.error = print_log
        
        # Also patch loguru logger
        from loguru import logger
        logger.debug = print_log
        logger.info = print_log
        logger.warning = print_log
        logger.error = print_log
        
        print_log("Test logging setup successful")
        yield

@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    """Grab a sample image from lorem picsum for testing.
    
    :returns: Sample image from lorem picsum.
    """
    response = requests.get("https://picsum.photos/200")
    return Image.open(BytesIO(response.content))

@pytest.fixture(scope="session")
def mock_torch_cuda() -> Generator[Mock, None, None]:
    """Mock torch.cuda.is_available to always return False for CPU testing.
    
    :yields: Mock object for torch.cuda.is_available.
    """
    with patch("torch.cuda.is_available", return_value=False) as mock:
        yield mock

@pytest.fixture(scope="session")
def mock_numpy_array() -> npt.NDArray[np.float64]:
    """Create a sample numpy array for embeddings.
    
    :returns: Sample embedding array.
    """
    return np.array([[0.1, 0.2, 0.3]], dtype=np.float64)

@pytest.fixture(scope="session")
def mock_context() -> Mock:
    """Create a mock context with response for API testing.
    
    :returns: Mock context object.
    """
    context = Mock()
    context.response = Mock()
    context.response.status_code = None
    return context

def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers.
    
    :param config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers", 
        "asyncio: mark test as async/await test",
    )
