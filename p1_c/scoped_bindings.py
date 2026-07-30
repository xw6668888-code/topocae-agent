"""Exception-safe, serialized temporary module bindings for generator revisions."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from types import ModuleType
from typing import Any, Iterator, Mapping


_BINDING_LOCK = RLock()


@contextmanager
def scoped_module_bindings(
    bindings: Mapping[ModuleType, Mapping[str, Any]],
) -> Iterator[None]:
    """Temporarily bind module attributes and restore them after success/error."""

    with _BINDING_LOCK:
        previous = {
            (module, name): getattr(module, name)
            for module, attributes in bindings.items()
            for name in attributes
        }
        try:
            for module, attributes in bindings.items():
                for name, value in attributes.items():
                    setattr(module, name, value)
            yield
        finally:
            for (module, name), value in reversed(tuple(previous.items())):
                setattr(module, name, value)


__all__ = ["scoped_module_bindings"]
