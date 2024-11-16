from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="EVA_AI",
    settings_files=["settings.yaml", ".secrets.yaml"],
    loaders=["dynaconf.loaders.env_loader", "dynaconf.loaders.yaml_loader"]
)
