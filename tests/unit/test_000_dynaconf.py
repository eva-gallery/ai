import os
from unittest.mock import patch

import pytest

from ai_api import settings

@pytest.mark.order(2)
def test_dynaconf_env_switching():
    """Test that dynaconf correctly switches environments and handles defaults"""
    
    # Test default environment (testing)
    assert settings.current_env == "testing"
    assert settings.test == 0
    
    # Test switching to development environment
    with patch.dict(os.environ, {"ENV_FOR_DYNACONF": "development"}):
        settings.reload()
        assert settings.current_env == "development"
        assert settings.test == 1  # Overridden in development
        
    # Test environment variable override
    with patch.dict(os.environ, {
        "ENV_FOR_DYNACONF": "testing",
        "EVA_AI_TEST": "1",
    }):
        settings.reload()
        assert settings.current_env == "testing"
        assert settings.test == 1  # Overridden by env var

        # Check that non-overridden values remain from settings.yaml
        assert settings.model.cache_dir == "/tmp/cache"

@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings to testing environment after each test"""
    yield
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    settings.reload()
