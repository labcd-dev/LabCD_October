import threading
from typing import Any, Callable, Dict, Optional

_registry: Dict[int, Callable[[Dict[str, Any]], None]] = {}
_lock = threading.Lock()


def register_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Register *callback* for the calling thread."""
    with _lock:
        _registry[threading.get_ident()] = callback


def unregister_callback() -> None:
    """Remove the callback registered for the calling thread (if any)."""
    with _lock:
        _registry.pop(threading.get_ident(), None)


def get_callback() -> Optional[Callable[[Dict[str, Any]], None]]:
    """Return the callback registered for the calling thread, or *None*."""
    with _lock:
        return _registry.get(threading.get_ident())
