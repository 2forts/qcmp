from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Protocol, Callable, Optional
from qiskit import QuantumCircuit

@dataclass(frozen=True)
class ComparatorMeta:
    key: str
    name: str
    paper: str
    kind: str
    signed: bool
    outputs: tuple[str, ...]
    ancilla_mode: str
    notes: str = ""

@dataclass
class BuildOutput:
    circuit: QuantumCircuit
    regs: Dict[str, Any]
    decode_counts: Optional[Callable[[dict[str,int]], dict[str,int]]] = None

class Comparator(Protocol):
    meta: ComparatorMeta
    def build(self, n: int, **kwargs) -> BuildOutput: ...
