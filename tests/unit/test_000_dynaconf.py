import os
from unittest.mock import patch

import pytest

from ai_api import settings

@pytest.mark.order(2)
def test_dynaconf_env_switching():
    """Test that dynaconf correctly switches environments and handles defaults"""
    
    # Test default environment (testing)
    assert settings.current_env == "testing"
    assert settings.postgres.host == "localhost"
    assert settings.model.cache_dir == "./models"  # From default env
    
    # Test switching to development environment
    with patch.dict(os.environ, {"ENV_FOR_DYNACONF": "development"}):
        settings.reload()
        assert settings.current_env == "development"
        assert settings.postgres.host == "localhost"
        assert settings.model.cache_dir == "/models"  # Overridden in development
        
    # Test environment variable override
    with patch.dict(os.environ, {
        "ENV_FOR_DYNACONF": "testing",
        "EVA_AI_POSTGRES__HOST": "test-db",
        "EVA_AI_MODEL__CACHE_DIR": "/override/path"
    }):
        settings.reload()
        assert settings.current_env == "testing"
        assert settings.postgres.host == "test-db"  # Overridden by env var
        assert settings.model.cache_dir == "/override/path"  # Overridden by env var
        
        # Check that non-overridden values remain from settings.yaml
        assert settings.postgres.port == 5432
        assert settings.model.embedding.image.name == "sentence-transformers/clip-ViT-B-32"

@pytest.mark.order(1)
def test_dynaconf_production_env():
    """Test production environment specific settings"""
    
    with patch.dict(os.environ, {"ENV_FOR_DYNACONF": "production"}):
        settings.reload()
        assert settings.current_env == "production"
        assert settings.bentoml.api.workers == 64
        assert settings.model.cache_dir == "./models"  # From default env
        
        # Test that production specific settings are loaded
        assert settings.bentoml.inference.fast_batched_op_max_batch_size == 64
        assert settings.bentoml.inference.slow_batched_op_max_batch_size == 32

@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings to testing environment after each test"""
    yield
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    settings.reload()
