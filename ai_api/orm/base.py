"""Module containing the base SQLAlchemy ORM model.

This module provides the base declarative model for SQLAlchemy ORM models,
incorporating async support through AsyncAttrs. All ORM models in the application
should inherit from this base class to ensure consistent behavior and async support.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase


class Base(AsyncAttrs, DeclarativeBase):
    """Base SQLAlchemy ORM model with async support.

    This class combines SQLAlchemy's DeclarativeBase with AsyncAttrs to provide
    async operation support. It serves as the foundation for all ORM models in
    the application, ensuring consistent behavior and proper async capabilities.

    All ORM models should inherit from this base class rather than directly from
    SQLAlchemy's DeclarativeBase.
    """

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Convert the model instance to a dictionary.

        Excludes auto-incrementing columns (typically primary keys) from the output.

        :returns: Dictionary containing all non-auto-incrementing model columns except relationships.
        """
        if exclude is None:
            exclude = set()

        return {
            column.key: getattr(self, column.key)
            for column in self.__table__.columns
            if not column.autoincrement and column.key not in exclude
        }
