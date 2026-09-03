"""Regressions for the public CLI, simulator target width and tree equations."""

from fractions import Fraction
import json
import sys

import pytest
from qiskit_aer import AerSimulator

import qcmp.comparators  # registers plugins before importing the registry
from qcmp.__main__ import main
from qcmp.metrics import estimate_resources, generate_equations
from qcmp.metrics.equations import _infer_tail_logarithm, _infer_tail_polynomial
from qcmp.registry import get
import qcmp.runner as runner


KEY = "cmp_kumar_tree_adapted"


@pytest.fixture
def limited_dense_backend(monkeypatch):
    """Reproduce the user's 28-qubit target independently of host RAM."""
    methods = []

    def factory(**kwargs):
        backend = AerSimulator(**kwargs)
        method = kwargs.get("method", "automatic")
        methods.append(method)
        if method in {"automatic", "statevector"}:
            backend.set_max_qubits(28)
        return backend

    monkeypatch.setattr(runner, "AerSimulator", factory)
    return methods


def test_exact_run_cli_with_28_qubit_dense_target(monkeypatch, capsys, limited_dense_backend):
    monkeypatch.setattr(sys, "argv", [
        "qcmp", "run", "--cmp", KEY, "--n", "4", "--A", "1", "--B", "2",
        "--metrics-all",
    ])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["decoded"] == {"gt": 0, "lt": 1, "eq": 0}
    assert payload["counts"] == {"010": 1024}
    assert payload["metrics"]["num_qubits"] == 29
    assert payload["metrics"]["quantum_cost"] == 100
    assert payload["metrics"]["delay"] == 36
    assert payload["metrics"]["t_depth"] == 12
    assert limited_dense_backend == ["automatic", "matrix_product_state"]


def test_exact_equations_cli_does_not_abort(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["qcmp", "equations", "--cmp", KEY])
    main()
    payload = json.loads(capsys.readouterr().out)
    equations = payload["equations"]
    assert equations["quantum_cost"]["expression"] == "31*N - 24"
    assert equations["t_count"]["exceptions"] == {"1": 14}
    delay = equations["delay"]
    assert delay["expression"] is None
    assert delay["degree"] is None
    assert delay["status"] == "not_inferred"
    assert delay["model"] == "unresolved"
    assert delay["domain"] == "sampled N only"
    assert delay["samples"]["4"] == 36
    assert delay["validation_samples"] == {"13": 67, "17": 80}
    assert any(warning.startswith("delay:") for warning in payload["warnings"])
    depth = equations["t_depth"]
    assert depth["expression"] == "6*ceil(log2(N))"
    assert depth["domain"] == "N >= 2"
    assert depth["exceptions"] == {"1": 6}
    assert depth["model"] == "ceil_log2"
    assert depth["status"] == "inferred"
    assert depth["validation_samples"] == {"13": 24, "17": 30}


@pytest.mark.parametrize("width,limit,method", [
    (13, 24, "automatic"),
    (24, 24, "automatic"),
    (25, 24, "matrix_product_state"),
    (28, 50, "automatic"),
    (29, 50, "matrix_product_state"),
    (509, 24, "matrix_product_state"),
])
def test_automatic_method_respects_both_limits(width, limit, method, limited_dense_backend):
    backend = runner._simulation_backend(
        width, seed=1, simulation_method="automatic", statevector_qubit_limit=limit,
    )
    assert backend.options.method == method
    assert backend.num_qubits >= width
    assert backend.target.num_qubits >= width


def test_explicit_dense_method_is_not_silently_enlarged(limited_dense_backend):
    backend = runner._simulation_backend(
        29, seed=1, simulation_method="statevector", statevector_qubit_limit=24,
    )
    assert backend.options.method == "statevector"
    assert backend.num_qubits == 28
    assert limited_dense_backend == ["statevector"]


def test_explicit_mps_method_can_exceed_default_mps_target():
    backend = runner._simulation_backend(
        509, seed=1, simulation_method="matrix_product_state", statevector_qubit_limit=24,
    )
    assert backend.options.method == "matrix_product_state"
    assert backend.target.num_qubits >= 509


def test_invalid_dense_limit():
    with pytest.raises(ValueError, match="statevector_qubit_limit"):
        runner._simulation_backend(
            4, seed=1, simulation_method="automatic", statevector_qubit_limit=0,
        )


@pytest.mark.parametrize("n,a,b", [(4, 2, 1), (4, 7, 7), (8, 255, 0), (64, 1, 2)])
def test_public_runner_large_comparator(n, a, b):
    result = runner.run(KEY, n=n, A=a, B=b, shots=8)
    assert result.decoded == {"gt": int(a > b), "lt": int(a < b), "eq": int(a == b)}


def test_mps_runner_supports_existing_dynamic_comparator():
    result = runner.run("lt_orts_tand", n=9, A=255, B=256, shots=8)
    assert result.circuit.num_qubits > 24
    assert result.decoded == {"lt": 1}


def test_holdout_rejects_false_constant_on_last_plateau():
    points = [(n, 6 * (n - 1).bit_length()) for n in range(2, 13)]
    # This was accepted by the old fitter: constant 24 for every N >= 9.
    coefficients, degree, start = _infer_tail_polynomial(
        points, max_degree=4, validation_points=3,
    )
    assert coefficients == [Fraction(24)]
    assert degree == 0
    assert points[start][0] == 9
    with pytest.raises(ValueError):
        _infer_tail_polynomial(
            points, max_degree=4, validation_points=3, validation_samples=[(13, 24), (17, 30)],
        )


@pytest.mark.parametrize("transform,expression,model", [
    (lambda n: (n - 1).bit_length(), "3*ceil(log2(N)) + 2", "ceil_log2"),
    (lambda n: n.bit_length() - 1, "3*floor(log2(N)) + 2", "floor_log2"),
])
def test_generic_rounded_log_inference(transform, expression, model):
    points = [(n, 3 * transform(n) + 2) for n in range(1, 13)]
    probes = [(n, 3 * transform(n) + 2) for n in [13, 17]]
    actual, start, actual_model = _infer_tail_logarithm(
        points, validation_points=3, validation_samples=probes,
    )
    assert (actual, start, actual_model) == (expression, 0, model)


def test_log_fit_does_not_count_plateau_repetitions_as_independent_levels():
    points = [(n, 6 * (n - 1).bit_length()) for n in range(5, 13)]
    with pytest.raises(ValueError):
        _infer_tail_logarithm(points, validation_points=3, validation_samples=[(17, 30)])


def test_t_depth_model_against_larger_boundary_sizes():
    comparator = get(KEY)
    report = generate_equations(comparator, metrics=["t_depth"])
    assert report.equations["t_depth"].expression == "6*ceil(log2(N))"
    for n in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129]:
        expected = 6 if n == 1 else 6 * (n - 1).bit_length()
        assert estimate_resources(comparator.build(n)).t_depth == expected


@pytest.mark.parametrize("options", [
    {"sample_max": 3}, {"max_degree": -1}, {"validation_points": 0},
])
def test_invalid_equation_parameters_still_raise(options):
    with pytest.raises(ValueError):
        generate_equations(get(KEY), **options)


def test_unknown_metric_still_raises():
    with pytest.raises(KeyError, match="Unknown resource metrics"):
        generate_equations(get(KEY), metrics=["not_a_metric"])
