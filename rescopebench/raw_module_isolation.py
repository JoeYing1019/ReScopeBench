"""Load raw environment packages without leaking their common module names.

Many source environments expose a top-level ``mcp_servers`` package.  Loading
two of them in one evaluator process would otherwise reuse whichever package
entered ``sys.modules`` first.  This helper temporarily owns that namespace,
loads and initializes one environment, then restores the caller's modules.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import Callable, Iterator, Sequence, TypeVar


_IMPORT_LOCK = RLock()
_T = TypeVar("_T")


def _owned_names(prefix: str) -> list[str]:
    return [
        name
        for name in tuple(sys.modules)
        if name == prefix or name.startswith(f"{prefix}.")
    ]


@contextmanager
def isolated_package_namespace(
    source_root: Path | str,
    *,
    package: str = "mcp_servers",
) -> Iterator[None]:
    """Temporarily import one source tree's package under its canonical name.

    The lock is required because both ``sys.path`` and ``sys.modules`` are
    process-global.  Imported module objects remain usable after the context;
    only their ambiguous canonical aliases are removed and restored.
    """

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    with _IMPORT_LOCK:
        original_path = list(sys.path)
        saved = {name: sys.modules[name] for name in _owned_names(package)}
        for name in saved:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        try:
            yield
        finally:
            for name in _owned_names(package):
                sys.modules.pop(name, None)
            sys.modules.update(saved)
            sys.path[:] = original_path
            importlib.invalidate_caches()


def load_isolated_modules(
    source_root: Path | str,
    module_names: Sequence[str],
    *,
    package: str = "mcp_servers",
    initialize: Callable[[tuple[ModuleType, ...]], _T] | None = None,
) -> tuple[tuple[ModuleType, ...], _T | None]:
    """Import modules from one raw environment and optionally initialize them."""

    root = Path(source_root).resolve()
    with isolated_package_namespace(root, package=package):
        modules = tuple(importlib.import_module(name) for name in module_names)
        for module in modules:
            module_path = Path(str(getattr(module, "__file__", ""))).resolve()
            if not module_path.is_relative_to(root):
                raise ImportError(
                    f"isolated import escaped source root: {module.__name__} -> {module_path}"
                )
        initialized = initialize(modules) if initialize is not None else None
    return modules, initialized


__all__ = ["isolated_package_namespace", "load_isolated_modules"]
