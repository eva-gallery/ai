"""Singleton metaclass for the application."""

from threading import Lock
from typing import Any, ClassVar


class Singleton(type):
    """A metaclass that implements the Singleton pattern.

    This ensures only one instance of a class is created. All subsequent
    instantiations return the same instance. This implementation is thread-safe.
    """

    _instances: ClassVar[dict[type, Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Create or return the singleton instance of the class."""
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
