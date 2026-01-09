from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister

def load_int(qc: QuantumCircuit, reg: QuantumRegister, value: int) -> None:
    if value < 0 or value >= 2 ** len(reg):
        raise ValueError(f"value {value} doesn't fit in {len(reg)} bits")
    for i in range(len(reg)):
        if (value >> i) & 1:
            qc.x(reg[i])
