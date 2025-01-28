"""Test dynaconf configuration."""

import os
from unittest.mock import patch

import pytest
from _bentoml_sdk.service.config import validate

from ai_api import settings


@pytest.mark.order(2)
def test_dynaconf_env_switching():
    """Test that dynaconf correctly switches environments and handles defaults."""
    # Test default environment (testing)
    assert settings.current_env == "testing"
    assert settings.test == 1  # testing env value

    # Test switching to development environment
    with patch.dict(os.environ, {"ENV_FOR_DYNACONF": "development"}):
        settings.reload()
        assert settings.current_env == "development"
        assert settings.test == 1  # development env value
        assert settings.debug is False  # default value

    # Test switching to production environment with env var override
    with patch.dict(os.environ, {
        "ENV_FOR_DYNACONF": "production",
        "EVA_AI_TEST": "1",
    }):
        settings.reload()
        assert settings.current_env == "production"
        assert settings.test == 1  # Overridden by env var
        assert settings.debug is False  # default value

        # Check that non-overridden values remain from settings.yaml
        assert settings.model.cache_dir == "./tmp/cache"


def test_validate_settings_for_bentoml_in_all_envs():
    """Test that settings are valid for BentoML in all environments."""
    # Test each environment's BentoML service settings
    for env in ["testing", "development", "production"]:
        with patch.dict(os.environ, {"ENV_FOR_DYNACONF": env}):
            settings.reload()

            # Validate API service config
            api_config = settings.bentoml.service.api.to_dict()
            validate(api_config)


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings to testing environment after each test."""
    yield
    os.environ["ENV_FOR_DYNACONF"] = "testing"
    settings.reload()
