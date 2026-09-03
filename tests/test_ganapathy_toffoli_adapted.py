"""Paper primitives, Fig. 3 tracing, coherent cleanup and qcmp integration."""

from itertools import product
import json
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

import qcmp.comparators
from qcmp.__main__ import main
from qcmp.comparators.cmp_ganapathy_toffoli_adapted import (
    CMP_Ganapathy_Toffoli_Adapted, c3x_via_clean_ancilla, gt_bit_into,
    lt_bit_into, paper_one_bit_comparator, xor_into, xnor_into,
)
from qcmp.encoding import load_int
from qcmp.metrics import estimate_resources, generate_equations
from qcmp.registry import get
from qcmp.runner import run


KEY = "cmp_ganapathy_toffoli_adapted"


def _flags(a, b):
    return {"gt": int(a > b), "lt": int(a < b), "eq": int(a == b)}


def _flag_bits(a, b):
    return int(a > b) | (int(a < b) << 1) | (int(a == b) << 2)


def _operations(qc):
    operations = []
    for instruction in qc.data:
        name = instruction.operation.name
        assert name in {"x", "cx", "ccx"}, "No opaque or relative-phase gates"
        operations.append((name, tuple(qc.find_bit(q).index for q in instruction.qubits)))
    return operations


def _evaluate(operations, state):
    """Independent interpreter of the ACTUAL emitted X/CX/CCX sequence."""
    for name, q in operations:
        if name == "x":
            state ^= 1 << q[0]
        elif name == "cx":
            state ^= ((state >> q[0]) & 1) << q[1]
        else:
            state ^= (((state >> q[0]) & 1) & ((state >> q[1]) & 1)) << q[2]
    return state


@pytest.mark.parametrize("gate,expression", [
    (xor_into, lambda a, b: a ^ b),
    (xnor_into, lambda a, b: 1 ^ a ^ b),
    (gt_bit_into, lambda a, b: a & (1 ^ b)),
    (lt_bit_into, lambda a, b: (1 ^ a) & b),
])
def test_primitive_complete_matrix_including_nonzero_targets(gate, expression):
    qc = QuantumCircuit(3)
    gate(qc, 0, 1, 2)
    reference = np.zeros((8, 8), dtype=complex)
    for source in range(8):
        a, b = source & 1, (source >> 1) & 1
        target = source ^ (expression(a, b) << 2)
        reference[target, source] = 1
    np.testing.assert_allclose(Operator(qc).data, reference, atol=1e-12)


def test_literal_one_bit_text_complete_unitary():
    qc = QuantumCircuit(5)
    paper_one_bit_comparator(qc, 0, 1, 2, 3, 4)
    reference = np.zeros((32, 32), dtype=complex)
    for source in range(32):
        a, b = source & 1, (source >> 1) & 1
        reference[source ^ (_flag_bits(a, b) << 2), source] = 1
    np.testing.assert_allclose(Operator(qc).data, reference, atol=1e-12)
    assert dict(qc.count_ops()) == {"x": 5, "cx": 2, "ccx": 2}


def test_c3x_synthesis_exact_on_clean_scratch_subspace():
    qc = QuantumCircuit(5)
    c3x_via_clean_ancilla(qc, 0, 1, 2, 3, 4)
    reference = np.zeros((32, 16), dtype=complex)
    for source in range(16):
        target = source ^ (8 if source & 7 == 7 else 0)
        reference[target, source] = 1
    np.testing.assert_allclose(Operator(qc).data[:, :16], reference, atol=1e-12)
    assert dict(qc.count_ops()) == {"ccx": 3}


def _figure_3_reading():
    """Independent literal tracing of Fig. 3, with data preparation omitted.

    Includes its redundant input X pairs and native three-control gates.
    It deliberately RETAINS garbage, unlike the adapted clean plugin.
    q0,q1 = A (LSB first), q2,q3 = B; outputs GT=q11, LT=q12, EQ=q4.
    """
    qc = QuantumCircuit(13)
    qc.x(6)
    for control, target in [(1,10),(0,9),(3,10),(2,9),(1,8),(0,7),(3,8),(2,7)]:
        qc.cx(control, target)
    qc.x(8)
    qc.x(7)
    qc.x(2)
    qc.x(3)
    qc.x(2)
    qc.x(3)
    qc.ccx(6, 8, 5)
    qc.ccx(5, 7, 4)
    qc.mcx([1, 6, 10], 11)
    qc.x(1)
    qc.mcx([3, 6, 10], 12)
    qc.x(1)
    qc.mcx([0, 5, 9], 11)
    qc.x(0)
    qc.mcx([2, 5, 9], 12)
    qc.x(0)
    return qc


def test_figure_3_tracing_all_16_pairs():
    qc = _figure_3_reading()
    for a, b in product(range(4), repeat=2):
        state = Statevector.from_int(a | (b << 2), 1 << 13).evolve(qc)
        index = int(np.argmax(np.abs(state.data)))
        assert np.isclose(state.data[index], 1)
        assert index & 15 == a | (b << 2)
        assert {name: (index >> q) & 1 for name, q in zip(("gt", "lt", "eq"), (11, 12, 4))} == _flags(a, b)


@pytest.mark.parametrize("n", range(1, 8))
def test_exhaustive_emitted_gates_all_inputs_and_clean_work(n):
    built = get(KEY).build(n, measure=False)
    ops = _operations(built.circuit)
    for a, b in product(range(1 << n), repeat=2):
        initial = a | (b << n)
        expected = initial | (_flag_bits(a, b) << (2*n))
        # Equality of the FULL basis index also verifies unchanged inputs
        # and that every single work qubit has been returned to zero.
        assert _evaluate(ops, initial) == expected


@pytest.mark.parametrize("n", range(1, 5))
def test_xor_output_semantics_with_every_initial_output(n):
    ops = _operations(get(KEY).build(n, measure=False).circuit)
    for a, b, flags in product(range(1 << n), range(1 << n), range(8)):
        initial = a | (b << n) | (flags << (2*n))
        assert _evaluate(ops, initial) == initial ^ (_flag_bits(a, b) << (2*n))


@pytest.mark.parametrize("n", range(1, 5))
def test_exhaustive_aer_mps_measuring_inputs_outputs_and_work(n):
    built = get(KEY).build(n, measure=False)
    circuits, expected = [], []
    for a, b in product(range(1 << n), repeat=2):
        prep = QuantumCircuit(*built.circuit.qregs)
        load_int(prep, built.regs["a"], a)
        load_int(prep, built.regs["b"], b)
        circuit = prep.compose(built.circuit)
        circuit.measure_all()
        circuits.append(circuit)
        expected.append(a | (b << n) | (_flag_bits(a, b) << (2*n)))
    backend = AerSimulator(method="matrix_product_state", max_parallel_threads=1)
    compiled = transpile(circuits, backend, optimization_level=0, num_processes=1)
    result = backend.run(compiled, shots=4, seed_simulator=2025).result()
    for i, index in enumerate(expected):
        assert result.get_counts(i) == {format(index, f"0{built.circuit.num_qubits}b"): 4}


@pytest.mark.parametrize("n", [1, 2])
def test_complex_superpositions_of_operands_and_outputs_without_relative_phases(n):
    built = get(KEY).build(n, measure=False)
    assert built.circuit.num_clbits == 0
    assert built.decode_counts is None
    rng = np.random.default_rng(2025)
    size = 1 << (2*n + 3)
    amplitudes = rng.normal(size=size) + 1j * rng.normal(size=size)
    amplitudes /= np.linalg.norm(amplitudes)
    initial = np.zeros(1 << built.circuit.num_qubits, dtype=complex)
    reference = np.zeros_like(initial)
    initial[:size] = amplitudes
    for source, amplitude in enumerate(amplitudes):
        a = source & ((1 << n) - 1)
        b = (source >> n) & ((1 << n) - 1)
        reference[source ^ (_flag_bits(a, b) << (2*n))] = amplitude
    actual = Statevector(initial).evolve(built.circuit)
    np.testing.assert_allclose(actual.data, reference, atol=1e-12)


@pytest.mark.parametrize("n", [5, 8, 64])
def test_public_runner_large_widths_and_msb_priority(n):
    maximum = (1 << n) - 1
    half = 1 << (n - 1)
    for a, b in [(0,0), (0,maximum), (maximum,0), (maximum,maximum),
                 (half-1,half), (half,half-1)]:
        # All three sizes exceed the automatic dense-state threshold.
        result = run(KEY, n=n, A=a, B=b, shots=8)
        assert result.decoded == _flags(a, b)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 64])
def test_resources_count_our_synthesis_and_uncomputation(n):
    built = get(KEY).build(n)
    resources = estimate_resources(built)
    expected_gates = {"x": 5, "cx": 2, "ccx": 2} if n == 1 else {
        "x": 2*n+2, "cx": 8*n, "ccx": 8*n-1,
    }
    assert dict(built.circuit.count_ops()) == {**expected_gates, "measure": 3}
    assert resources.quantum_cost == (17 if n == 1 else 50*n-3)
    assert resources.t_count == (14 if n == 1 else 56*n-7)
    assert resources.num_qubits == (5 if n == 1 else 5*n+4)
    assert resources.num_auxiliaries == (3 if n == 1 else 3*n+4)
    assert resources.num_garbage_outputs == 0
    assert len(built.resources.restored_zero_qubits) == (0 if n == 1 else 3*n+1)
    assert len(built.resources.solution_qubits) == 3


def test_all_resource_equations_and_exceptions():
    report = generate_equations(get(KEY))
    assert not report.warnings
    expected = {
        "quantum_cost": ("50*N - 3", 2, {1:17}),
        "delay": ("35*N - 3", 3, {1:15, 2:68}),
        "num_qubits": ("5*N + 4", 2, {1:5}),
        "num_auxiliaries": ("3*N + 4", 2, {1:3}),
        "num_garbage_outputs": ("0", 1, {}),
        "t_count": ("56*N - 7", 2, {1:14}),
        "t_depth": ("21*N - 6", 2, {1:6}),
    }
    for name, (expression, valid_from, exceptions) in expected.items():
        equation = report.equations[name]
        assert (equation.expression, equation.valid_from, equation.exceptions) == (
            expression, valid_from, exceptions,
        )
        assert equation.status == "inferred"


def test_exact_run_metrics_all_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["qcmp", "run", "--cmp", KEY, "--n", "4",
                                      "--A", "1", "--B", "2", "--metrics-all"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["decoded"] == {"gt": 0, "lt": 1, "eq": 0}
    assert result["counts"] == {"010": 1024}
    assert result["metrics"]["quantum_cost"] == 197
    assert result["metrics"]["num_qubits"] == 24
    assert result["metrics"]["num_garbage_outputs"] == 0


def test_exact_equations_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["qcmp", "equations", "--cmp", KEY])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["comparator"] == KEY
    assert len(result["equations"]) == 7
    assert all(item["status"] == "inferred" for item in result["equations"].values())


def test_registered_plugin_and_joint_decoder():
    comparator = get(KEY)
    assert isinstance(comparator, CMP_Ganapathy_Toffoli_Adapted)
    built = comparator.build(3)
    assert built.regs["out_qubit"] == built.regs["out_qubits"]["lt"]
    assert built.decode_counts({"11 010": 3, "00 010": 2, "00 001": 1}) == _flags(1, 2)
    with pytest.raises(ValueError):
        built.decode_counts({})


@pytest.mark.parametrize("n,error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_width(n, error):
    with pytest.raises(error):
        get(KEY).build(n)


def test_invalid_options_and_inputs():
    with pytest.raises(TypeError):
        get(KEY).build(2, sense="lt")
    with pytest.raises(TypeError):
        get(KEY).build(2, measure="False")
    with pytest.raises(ValueError):
        run(KEY, n=2, A=4, B=0)
    with pytest.raises(ValueError):
        run(KEY, n=2, A=0, B=-1)
