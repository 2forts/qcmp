from __future__ import annotations
from typing import Dict
from qcmp.comparators.base import Comparator

_COMPARATORS: Dict[str, Comparator] = {}

def register(comparator: Comparator) -> None:
    key = comparator.meta.key
    if key in _COMPARATORS:
        raise KeyError(f"Comparator '{key}' already registered")
    _COMPARATORS[key] = comparator

def get(key: str) -> Comparator:
    if key not in _COMPARATORS:
        avail = ", ".join(sorted(_COMPARATORS))
        raise KeyError(f"Unknown comparator '{key}'. Available: {avail}")
    return _COMPARATORS[key]

def list_all() -> list[str]:
    return sorted(_COMPARATORS)
