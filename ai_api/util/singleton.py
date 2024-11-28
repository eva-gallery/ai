from typing import Any


class Singleton(type):
    """A metaclass that implements the Singleton pattern.
    
    This ensures only one instance of a class is created. All subsequent
    instantiations return the same instance.
    """

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
