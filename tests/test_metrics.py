from __future__ import annotations

import json

import qcmp.comparators  # registers comparator plugins
from qcmp.metrics import (
    GateProfile,
    GateProfileRegistry,
    estimate_resources,
    generate_equations,
    load_gate_profiles,
)
from qcmp.registry import get


def test_structural_resources_xia_cgc() -> None:
    resources = estimate_resources(get("lt_xia_cgc").build(4)).to_dict()
    assert resources == {
        "quantum_cost": 73,
        "delay": 52,
        "num_qubits": 10,
        "num_auxiliaries": 2,
        "num_garbage_outputs": 0,
        "t_count": 56,
        "t_depth": 24,
    }


def test_equations_include_small_n_exception() -> None:
    report = generate_equations(
        get("lt_xia_qip"),
        sample_max=10,
        metrics=["quantum_cost", "num_qubits", "num_garbage_outputs"],
    )
    assert report.equations["quantum_cost"].expression == "18*N - 12"
    assert report.equations["quantum_cost"].valid_from == 2
    assert report.equations["quantum_cost"].exceptions == {1: 7}
    assert report.equations["num_qubits"].expression == "2*N + 2"
    assert report.equations["num_garbage_outputs"].expression == "0"


def test_custom_toffoli_profile_is_used() -> None:
    registry = GateProfileRegistry(
        [
            GateProfile(
                name="ccx",
                num_qubits=3,
                quantum_cost=10,
                delay=8,
                t_count=4,
                t_depth=2,
            ),
            # Profiles cannot override the mandatory unit cost/delay rule for
            # one- and two-qubit gates.
            GateProfile(
                name="x",
                num_qubits=1,
                quantum_cost=100,
                delay=100,
                t_count=100,
                t_depth=100,
            ),
        ]
    )
    resources = estimate_resources(
        get("lt_xia_cgc").build(1), gate_profiles=registry
    )
    assert resources.quantum_cost == 29  # nine 1/2-qubit gates + two CCX
    assert resources.delay == 22
    assert resources.t_count == 8
    assert resources.t_depth == 4


def test_measurement_dependency_is_kept_in_delay_path() -> None:
    resources = estimate_resources(get("lt_orts_tand").build(2))
    assert resources.delay == 26


def test_gate_profiles_can_be_loaded_from_json(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "ccx/3": {
                    "quantum_cost": 9,
                    "delay": 7,
                    "t_count": 4,
                    "t_depth": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_gate_profiles(path)
    profile = registry.get("ccx", 3)
    assert profile is not None
    assert profile.quantum_cost == 9
    assert profile.delay == 7
