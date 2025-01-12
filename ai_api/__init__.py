"""Main initialization module for the AI API service.

This module handles the initialization of the AI API service, including:
- Configuration loading using Dynaconf
- Environment variable setup
- Safe globals configuration for sentence transformers
- Model cache directory setup

The module uses Dynaconf for configuration management, supporting multiple
configuration files and environment-based settings.
"""

import os
from pathlib import Path
from typing import Any

import torch.serialization
from dynaconf import Dynaconf
from sentence_transformers.models import Dense
from torch import Tensor, nn
from transformers import modeling_utils


def filter_none_from_dict_recursive(d: dict[str, Any]) -> dict[str, Any]:
    return {k: filter_none_from_dict_recursive(v) if isinstance(v, dict) else v for k, v in d.items() if v is not None}


# Add safe globals for sentence transformers
torch.serialization.add_safe_globals([
    Dense,
    nn,
    Tensor,
    modeling_utils,
    dict,
    list,
    tuple,
    set,
    str,
    int,
    float,
    bool,
])

settings = Dynaconf(
    envvar_prefix="EVA_AI",
    settings_files=[
        "settings.yaml",
        "settings.local.yaml",
        ".secrets.yaml",
    ],
    environments=True,
    load_dotenv=True,
    env=os.getenv("ENV_FOR_DYNACONF", "testing"),
    loaders=["dynaconf.loaders.env_loader", "dynaconf.loaders.yaml_loader"],
)

if os.getenv("HF_HOME") is None:
    os.environ["HF_HOME"] = settings.model.cache_dir or str(Path.cwd() / "cache")

if os.getenv("HF_OFFLINE") is None:
    os.environ["HF_OFFLINE"] = settings.bentoml.hf_offline or "false"

if settings.debug:
    os.environ["CI"] = "true"


API_SERVICE_KWARGS = filter_none_from_dict_recursive(settings.bentoml.service.api.to_dict())
INFERENCE_SERVICE_KWARGS = filter_none_from_dict_recursive(settings.bentoml.service.embedding.to_dict())


__all__ = ["API_SERVICE_KWARGS", "INFERENCE_SERVICE_KWARGS", "settings"]
