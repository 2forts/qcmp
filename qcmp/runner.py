from __future__ import annotations
from typing import Optional, Any
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qcmp.registry import get
from qcmp.encoding import load_int
from qcmp.result import RunResult


_CLI_OPERANDS = ("a", "b")


def _simulation_backend(
    num_qubits: int,
    *,
    seed: int,
    simulation_method: str,
    statevector_qubit_limit: int,
) -> AerSimulator:
    """Choose a simulator before transpilation validates the target width.

    Aer 'automatic' advertises a dense-state width derived from host memory.
    Large comparator circuits may be cheap in MPS but exceed that width. In
    automatic mode, prefer MPS above the smaller of the advertised width and
    a conservative dense-state limit (24 qubits = 256 MiB for amplitudes alone).
    MPS can still be expensive for highly entangled inputs; callers can select
    an explicit method. Never enlarge a dense simulator's advertised width.
    """
    if statevector_qubit_limit < 1:
        raise ValueError("statevector_qubit_limit must be >= 1")
    backend = AerSimulator(seed_simulator=seed, method=simulation_method)
    if simulation_method == "automatic":
        dense_limit = min(statevector_qubit_limit, backend.num_qubits)
        if num_qubits > dense_limit:
            backend = AerSimulator(seed_simulator=seed, method="matrix_product_state")

    if backend.options.method == "matrix_product_state" and num_qubits > backend.num_qubits:
        # Aer normally advertises 63 MPS qubits. This updates the transpiler
        # target, not a dense allocation or the state's bond dimension.
        backend.set_max_qubits(num_qubits)
    return backend


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
    gate_profiles: Any = None,
    simulation_method: str = "automatic",
    statevector_qubit_limit: int = 24,
    **build_kwargs: Any,
) -> RunResult:
    comp = get(key)
    input_values = {"a": A, "b": B}
    constant_operands = tuple(getattr(comp.meta, "constant_operands", ()))
    if len(set(constant_operands)) != len(constant_operands):
        raise ValueError(f"Comparator {key!r} declares duplicate constant operands")
    unknown_constants = sorted(set(constant_operands) - set(_CLI_OPERANDS))
    if unknown_constants:
        raise ValueError(
            f"Comparator {key!r} declares unsupported constant operands: "
            f"{', '.join(unknown_constants)}"
        )

    effective_build_kwargs = dict(build_kwargs)
    for operand in constant_operands:
        parameter = f"{operand}_value"
        if parameter in effective_build_kwargs:
            raise TypeError(
                f"{parameter} is supplied automatically from the runner's "
                f"--{operand.upper()} input"
            )
        effective_build_kwargs[parameter] = input_values[operand]

    built = comp.build(n, **effective_build_kwargs)

    qc = built.circuit
    regs = built.regs

    # --- IMPORTANT: prepare inputs BEFORE the comparator circuit ---
    # We build a prefix circuit with the same registers, then compose it in front.
    prefix = QuantumCircuit(*qc.qregs, *qc.cregs, name="prep")
    for operand in _CLI_OPERANDS:
        if operand in constant_operands:
            continue
        if operand not in regs:
            raise KeyError(
                f"Comparator {key!r} has no register for runtime operand {operand!r}"
            )
        load_int(prefix, regs[operand], input_values[operand])

    qc_prepared = prefix.compose(qc)

    backend = _simulation_backend(
        qc_prepared.num_qubits,
        seed=seed,
        simulation_method=simulation_method,
        statevector_qubit_limit=statevector_qubit_limit,
    )
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
        metrics = compute(built, keys=metric_keys, gate_profiles=gate_profiles)

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