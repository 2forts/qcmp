from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qiskit import QuantumCircuit

from qcmp.comparators.base import BuildOutput, ResourceLayout
from qcmp.metrics.profiles import DEFAULT_GATE_PROFILES, GateProfile, GateProfileRegistry


NON_GATE_OPS = {"measure", "reset", "barrier", "delay"}
T_GATES = {"t", "tdg"}
RESOURCE_METRICS = (
    "quantum_cost",
    "delay",
    "num_qubits",
    "num_auxiliaries",
    "num_garbage_outputs",
    "t_count",
    "t_depth",
)

RESOURCE_METRIC_ALIASES = {
    "qubits": "num_qubits",
    "ancilla_inputs": "num_auxiliaries",
    "garbage_outputs": "num_garbage_outputs",
}


class MissingGateProfileError(ValueError):
    pass


@dataclass(frozen=True)
class CircuitResources:
    quantum_cost: int
    delay: int
    num_qubits: int
    num_auxiliaries: int
    num_garbage_outputs: int
    t_count: int
    t_depth: int

    def to_dict(self) -> dict[str, int]:
        return {
            "quantum_cost": self.quantum_cost,
            "delay": self.delay,
            "num_qubits": self.num_qubits,
            "num_auxiliaries": self.num_auxiliaries,
            "num_garbage_outputs": self.num_garbage_outputs,
            "t_count": self.t_count,
            "t_depth": self.t_depth,
        }


@dataclass(frozen=True)
class _LocalResources:
    quantum_cost: int = 0
    delay: int = 0
    t_count: int = 0
    t_depth: int = 0

    @classmethod
    def from_profile(cls, profile: GateProfile) -> "_LocalResources":
        return cls(
            quantum_cost=profile.quantum_cost,
            delay=profile.delay,
            t_count=profile.t_count,
            t_depth=profile.t_depth,
        )


def _resource_bits(layout: ResourceLayout, built: BuildOutput) -> tuple[set[Any], set[Any], set[Any]]:
    operands: set[Any] = set()
    for register_name in layout.operand_registers:
        if register_name not in built.regs:
            raise KeyError(f"Unknown operand register {register_name!r}")
        operands.update(built.regs[register_name])

    solutions = set(layout.solution_qubits)
    restored_zero = set(layout.restored_zero_qubits)

    all_qubits = set(built.circuit.qubits)
    for label, bits in (
        ("operand", operands),
        ("solution", solutions),
        ("restored-zero", restored_zero),
    ):
        unknown = bits - all_qubits
        if unknown:
            raise ValueError(f"{label} layout contains qubits outside the circuit")

    overlaps = (
        ("operand/solution", operands & solutions),
        ("operand/restored-zero", operands & restored_zero),
        ("solution/restored-zero", solutions & restored_zero),
    )
    for label, bits in overlaps:
        if bits:
            raise ValueError(f"Resource layout overlaps in {label}: {bits}")

    return operands, solutions, restored_zero


def _estimate_circuit_local(
    circuit: QuantumCircuit,
    registry: GateProfileRegistry,
    *,
    decomposition_stack: tuple[tuple[int, str], ...] = (),
) -> _LocalResources:
    delay_clock = {qubit: 0 for qubit in circuit.qubits}
    t_clock = {qubit: 0 for qubit in circuit.qubits}
    classical_delay_clock = {clbit: 0 for clbit in circuit.clbits}
    classical_t_clock = {clbit: 0 for clbit in circuit.clbits}
    quantum_cost = 0
    t_count = 0

    def synchronize(clocks: dict[Any, int], bits: tuple[Any, ...]) -> None:
        if not bits:
            return
        value = max(clocks[bit] for bit in bits)
        for bit in bits:
            clocks[bit] = value

    for instruction in circuit.data:
        operation = instruction.operation
        qargs = tuple(instruction.qubits)
        cargs = tuple(instruction.clbits)

        if operation.name == "measure":
            # Measurement has no gate cost or delay in this model, but the
            # classical result cannot be used before the measured qubit's
            # preceding quantum path has completed.
            for qubit, clbit in zip(qargs, cargs):
                classical_delay_clock[clbit] = max(
                    classical_delay_clock[clbit], delay_clock[qubit]
                )
                classical_t_clock[clbit] = max(
                    classical_t_clock[clbit], t_clock[qubit]
                )
            continue

        if operation.name == "barrier":
            synchronize(delay_clock, qargs)
            synchronize(t_clock, qargs)
            continue

        if operation.name in {"reset", "delay"}:
            continue

        local = _operation_resources(
            operation,
            registry,
            decomposition_stack=decomposition_stack,
        )
        quantum_cost += local.quantum_cost
        t_count += local.t_count

        delay_predecessors = [delay_clock[q] for q in qargs]
        delay_predecessors.extend(classical_delay_clock[c] for c in cargs)
        start = max(delay_predecessors, default=0)
        end = start + local.delay
        for qubit in qargs:
            delay_clock[qubit] = end

        t_predecessors = [t_clock[q] for q in qargs]
        t_predecessors.extend(classical_t_clock[c] for c in cargs)
        t_start = max(t_predecessors, default=0)
        t_end = t_start + local.t_depth
        for qubit in qargs:
            t_clock[qubit] = t_end

    return _LocalResources(
        quantum_cost=quantum_cost,
        delay=max(
            [*delay_clock.values(), *classical_delay_clock.values()], default=0
        ),
        t_count=t_count,
        t_depth=max([*t_clock.values(), *classical_t_clock.values()], default=0),
    )


def _operation_resources(
    operation: Any,
    registry: GateProfileRegistry,
    *,
    decomposition_stack: tuple[tuple[int, str], ...],
) -> _LocalResources:
    blocks: Iterable[QuantumCircuit] = getattr(operation, "blocks", ()) or ()
    blocks = tuple(blocks)
    if blocks:
        # A control-flow instruction executes at most one branch. Measurements
        # and classical evaluation are outside the quantum-gate resource model.
        branch_resources = [
            _estimate_circuit_local(
                block,
                registry,
                decomposition_stack=decomposition_stack + ((id(operation), operation.name),),
            )
            for block in blocks
        ]
        return _LocalResources(
            quantum_cost=max((r.quantum_cost for r in branch_resources), default=0),
            delay=max((r.delay for r in branch_resources), default=0),
            t_count=max((r.t_count for r in branch_resources), default=0),
            t_depth=max((r.t_depth for r in branch_resources), default=0),
        )

    if operation.num_qubits <= 2:
        is_t = operation.name in T_GATES
        return _LocalResources(
            quantum_cost=1,
            delay=1,
            t_count=int(is_t),
            t_depth=int(is_t),
        )

    profile = registry.get(operation.name, operation.num_qubits)
    if profile is not None:
        return _LocalResources.from_profile(profile)

    definition = getattr(operation, "definition", None)
    if definition is not None:
        signature = f"{operation.name}/{operation.num_qubits}"
        if any(operation_id == id(operation) for operation_id, _ in decomposition_stack):
            chain = " -> ".join(
                [label for _, label in decomposition_stack] + [signature]
            )
            raise MissingGateProfileError(f"Recursive gate definition detected: {chain}")
        return _estimate_circuit_local(
            definition,
            registry,
            decomposition_stack=decomposition_stack + ((id(operation), signature),),
        )

    raise MissingGateProfileError(
        f"No resource profile or decomposition for gate "
        f"{operation.name!r} acting on {operation.num_qubits} qubits"
    )


def estimate_resources(
    built: BuildOutput,
    *,
    gate_profiles: GateProfileRegistry | None = None,
) -> CircuitResources:
    """Compute structural resources for a comparator build.

    Input preparation, measurements, resets, barriers and backend-specific
    transpilation are intentionally excluded. Conditional quantum branches use
    their worst-case branch resource.
    """

    if built.resources is None:
        raise ValueError(
            "BuildOutput.resources is required for ancilla and garbage metrics"
        )

    registry = gate_profiles or DEFAULT_GATE_PROFILES
    local = _estimate_circuit_local(built.circuit, registry)
    operands, solutions, restored_zero = _resource_bits(built.resources, built)
    all_qubits = set(built.circuit.qubits)
    garbage = all_qubits - operands - solutions - restored_zero

    return CircuitResources(
        quantum_cost=local.quantum_cost,
        delay=local.delay,
        num_qubits=built.circuit.num_qubits,
        num_auxiliaries=built.circuit.num_qubits - len(operands),
        num_garbage_outputs=len(garbage),
        t_count=local.t_count,
        t_depth=local.t_depth,
    )
