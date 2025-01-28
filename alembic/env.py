from logging.config import fileConfig
from typing import Any, cast

from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.engine.url import URL

from ai_api import settings
from ai_api.orm import Base
from ai_api.orm.gallery_embedding import GalleryEmbedding # type: ignore[unused-import]
from ai_api.orm.image import Image # type: ignore[unused-import]

if settings.debug:
    exit(0)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
global_url = URL.create("postgresql", settings.postgres.user, settings.postgres.password, settings.postgres.host, settings.postgres.port, settings.postgres.db)
print("Using connection string: ", global_url.render_as_string(hide_password=True))
global_url = global_url.render_as_string(hide_password=False).replace("%", "%%")
config = context.config
config.set_main_option("sqlalchemy.url", global_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = global_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration: dict[str, Any] = cast(dict[str, Any], config.get_section(config.config_ini_section))
    configuration["sqlalchemy.url"] = global_url
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
