from __future__ import annotations

from typing import Any, Optional

from qiskit import QuantumCircuit

from qcmp.comparators.base import BuildOutput
from qcmp.metrics.equations import EquationReport, MetricEquation, generate_equations
from qcmp.metrics.estimator import (
    NON_GATE_OPS,
    RESOURCE_METRIC_ALIASES,
    RESOURCE_METRICS,
    CircuitResources,
    MissingGateProfileError,
    estimate_resources,
)
from qcmp.metrics.profiles import (
    DEFAULT_GATE_PROFILES,
    GateProfile,
    GateProfileRegistry,
    load_gate_profiles,
)


LEGACY_METRICS = (
    "depth",
    "size",
    "op_counts",
    "gate_counts",
    "total_gates",
    "weighted_cost",
)


def available() -> list[str]:
    return list(RESOURCE_METRICS + LEGACY_METRICS)


def _only_quantum_gates(circuit: QuantumCircuit) -> dict[str, int]:
    return {
        op: int(count)
        for op, count in circuit.count_ops().items()
        if op not in NON_GATE_OPS
    }


def compute(
    subject: BuildOutput | QuantumCircuit,
    keys: Optional[list[str]] = None,
    *,
    gate_profiles: GateProfileRegistry | None = None,
) -> dict[str, Any]:
    """Compute concrete metrics.

    Passing ``BuildOutput`` enables all resource metrics. Passing a bare
    ``QuantumCircuit`` remains supported for the legacy syntactic metrics.
    """

    selected = available() if not keys else keys
    accepted = set(available()) | set(RESOURCE_METRIC_ALIASES)
    unknown = sorted(set(selected) - accepted)
    if unknown:
        raise KeyError(", ".join(unknown))

    built = subject if isinstance(subject, BuildOutput) else None
    circuit = subject.circuit if built is not None else subject
    resources = (
        estimate_resources(built, gate_profiles=gate_profiles).to_dict()
        if built is not None and any(
            RESOURCE_METRIC_ALIASES.get(key, key) in RESOURCE_METRICS
            for key in selected
        )
        else None
    )

    output: dict[str, Any] = {}
    for key in selected:
        canonical_key = RESOURCE_METRIC_ALIASES.get(key, key)
        if canonical_key in RESOURCE_METRICS:
            if resources is None:
                if canonical_key == "num_qubits":
                    output[key] = circuit.num_qubits
                    continue
                raise ValueError(
                    f"Metric {key!r} requires BuildOutput with ResourceLayout"
                )
            output[key] = resources[canonical_key]
        elif key == "depth":
            output[key] = circuit.depth()
        elif key == "size":
            output[key] = circuit.size()
        elif key == "op_counts":
            output[key] = dict(circuit.count_ops())
        elif key == "gate_counts":
            output[key] = _only_quantum_gates(circuit)
        elif key == "total_gates":
            output[key] = sum(_only_quantum_gates(circuit).values())
        elif key == "weighted_cost":
            weights = {
                "x": 1,
                "h": 1,
                "s": 1,
                "sdg": 1,
                "t": 1,
                "tdg": 1,
                "cx": 2,
                "cz": 2,
                "ccx": 5,
                "measure": 0,
                "reset": 0,
            }
            output[key] = sum(
                weights.get(op, 1) * int(count)
                for op, count in circuit.count_ops().items()
            )

    return output


__all__ = [
    "CircuitResources",
    "DEFAULT_GATE_PROFILES",
    "EquationReport",
    "GateProfile",
    "GateProfileRegistry",
    "MetricEquation",
    "MissingGateProfileError",
    "RESOURCE_METRIC_ALIASES",
    "RESOURCE_METRICS",
    "available",
    "compute",
    "estimate_resources",
    "generate_equations",
    "load_gate_profiles",
]
