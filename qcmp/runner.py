from __future__ import annotations
from typing import Optional, Any
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qcmp.registry import get
from qcmp.encoding import load_int
from qcmp.result import RunResult


def run(
    key: str,
    n: int,
    A: int,
    B: int,
    shots: int = 1024,
    seed: int = 1,
    basis_gates: Optional[list[str]] = None,
    optimization_level: int = 1,
    compute_metrics: bool = False,
    metric_keys: Optional[list[str]] = None,
    **build_kwargs: Any,
) -> RunResult:
    comp = get(key)
    built = comp.build(n, **build_kwargs)

    qc = built.circuit
    regs = built.regs

    # --- IMPORTANT: prepare inputs BEFORE the comparator circuit ---
    # We build a prefix circuit with the same registers, then compose it in front.
    prefix = QuantumCircuit(*qc.qregs, *qc.cregs, name="prep")
    load_int(prefix, regs["a"], A)
    load_int(prefix, regs["b"], B)

    qc_prepared = prefix.compose(qc)

    backend = AerSimulator(seed_simulator=seed)
    tqc = transpile(
        qc_prepared,
        backend=backend,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    counts = backend.run(tqc, shots=shots).result().get_counts()

    decoded = (
        built.decode_counts(counts)
        if built.decode_counts
        else {"bitstring": max(counts, key=counts.get)}
    )

    metrics = None
    if compute_metrics:
        from qcmp.metrics import compute  # lazy import (optional module usage)
        metrics = compute(tqc, keys=metric_keys)

    return RunResult(
        key=key,
        n=n,
        A=A,
        B=B,
        counts=counts,
        decoded=decoded,
        circuit=qc_prepared,      # <-- return the prepared circuit (what we actually ran)
        transpiled=tqc,
        metrics=metrics,
    )