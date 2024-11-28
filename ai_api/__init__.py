import os
from pathlib import Path

import torch.serialization
from dynaconf import Dynaconf
from sentence_transformers.models import Dense
from torch import Tensor, nn
from transformers import modeling_utils

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


__all__ = ["settings"]
