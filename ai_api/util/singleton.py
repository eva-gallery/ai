"""Module providing thread-safe singleton pattern implementation.

This module implements a thread-safe singleton metaclass that ensures only one
instance of a class is ever created, with proper handling of concurrent access
through double-checked locking pattern.
"""

from threading import Lock
from typing import Any, ClassVar


class Singleton(type):
    """Thread-safe singleton metaclass implementation.

    This metaclass implements the singleton pattern ensuring only one instance
    of a class is ever created. It uses a double-checked locking pattern to
    handle concurrent instantiation attempts safely.

    Attributes:
        _instances: Class-level dictionary storing singleton instances.
        _lock: Thread lock for synchronizing instance creation.

    """

    _instances: ClassVar[dict[type, Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Create or return the singleton instance of the class.

        This method implements double-checked locking to ensure thread-safe
        singleton instance creation. If an instance already exists, it is
        returned; otherwise, a new instance is created in a thread-safe manner.

        :param args: Positional arguments for class instantiation.
        :type args: Any
        :param kwargs: Keyword arguments for class instantiation.
        :type kwargs: Any
        :returns: The singleton instance of the class.
        :rtype: Any
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
