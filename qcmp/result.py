from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class RunResult:
    key: str
    n: int
    A: int
    B: int
    counts: dict[str,int]
    decoded: dict[str,int]
    circuit: Any
    transpiled: Any
    metrics: dict[str,Any] | None = None
