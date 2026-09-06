from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

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
    # Operands whose values are compiled into the circuit instead of loaded
    # into quantum registers. The runner maps "a"/"b" to a_value/b_value.
    # Empty by default, so existing comparator plugins keep their behaviour.
    constant_operands: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceLayout:
    """Classifies comparator qubits for structural resource metrics.

    Operand registers hold the two inputs. Solution qubits contain the final
    comparator result. Restored-zero qubits are auxiliary inputs returned to
    |0>. Every remaining circuit qubit is counted as a garbage output.
    """

    operand_registers: tuple[str, ...]
    solution_qubits: tuple[Any, ...]
    restored_zero_qubits: tuple[Any, ...] = ()


@dataclass
class BuildOutput:
    circuit: QuantumCircuit
    regs: Dict[str, Any]
    decode_counts: Optional[Callable[[dict[str, int]], dict[str, int]]] = None
    resources: ResourceLayout | None = None


class Comparator(Protocol):
    meta: ComparatorMeta

    def build(self, n: int, **kwargs) -> BuildOutput: ...
