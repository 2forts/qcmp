"""Register-driven Quantum Phase-based Comparator (QPC).

Paper
-----
A. Berti and A. Poggiali, "A Quantum Phase-based Comparator",
arXiv:2609.25262v1 [quant-ph] (2026).

PROVENANCE AND ADAPTER DECISIONS
--------------------------------

[PAPER P1] Theorem 1 places a phase

    phi_x - phi_y - pi/2

between two Hadamard gates.  It gives

    Pr(0) = 1/2 + sin(phi_x - phi_y)/2,

so ``Pr(0) < 1/2`` exactly when ``x < y``, provided the increasing phase
encoding spans less than pi.  Equality is explicitly excluded and the output
is statistical, not a deterministic comparison bit.

[PAPER P2] Figure 2 replaces the two data phase gates with binary-weighted
controlled-phase cascades.  In the paper's big-endian notation the weights are
``beta_i = angle_range / 2**(i+1)``.  Qiskit and qcmp load integers little
endian, so register qubit ``j`` instead receives weight ``2**j`` times the
smallest grid step.  This is only an endianness conversion; it implements the
same phases and the same 2*n controlled-phase gates.

[PAPER P3] Proposition 2 permits an integer phase power ``r`` by multiplying
every cascade angle by ``r``.  It adds no gate or qubit, but the order rule is
valid only while ``r * abs(phi_x - phi_y) < pi``.

[QCMP A1] qcmp supplies unsigned integers A and B in ``n``-qubit registers.
We map the complete integer range ``0 .. 2**n-1`` monotonically to phases
``0 .. phase_span``:

    phi(k) = k * phase_span / (2**n - 1).

This is exactly the paper's fixed-point grid with
``angle_range = phase_span / (1 - 2**-n)``.  The default phase span is one
radian, matching the paper's standing requirement ``phase_span < pi/2``.

[QCMP A2] The decoder implements the paper's empirical rule literally:
``lt = 1`` when fewer than half of the comparator measurements are zero.  A
finite-shot run may therefore misclassify close operands, and equality has no
defined statistical decision in the paper.  If the observed counts tie
exactly, qcmp returns ``lt = 0`` because the strict test ``p_hat(0) < 1/2`` is
false.  Increasing ``--shots`` reduces sampling error away from equality.

[QCMP A3] ``measure=False`` exposes the coherent Figure 2 circuit for use in a
larger algorithm.  The normal qcmp runner uses the default ``measure=True``.
"""

from __future__ import annotations

from math import isfinite, pi
from numbers import Integral, Real

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


DEFAULT_PHASE_SPAN = 1.0


def validate_phase_parameters(phase_span: float, power: int) -> tuple[float, int]:
    """Validate the paper's global no-wrapping condition.

    ``power * phase_span < pi`` guarantees correctness for every pair in the
    full qcmp input range.  The paper can use a larger power for a known subset
    of pairs, but a reusable comparator plugin cannot assume such a subset.
    """
    if isinstance(phase_span, bool) or not isinstance(phase_span, Real):
        raise TypeError("phase_span must be a real number")
    span = float(phase_span)
    if not isfinite(span) or span <= 0:
        raise ValueError("phase_span must be finite and > 0")

    if isinstance(power, bool) or not isinstance(power, Integral):
        raise TypeError("power must be a positive integer")
    integer_power = int(power)
    if integer_power < 1:
        raise ValueError("power must be >= 1")
    if integer_power * span >= pi:
        raise ValueError(
            "power * phase_span must be < pi to preserve the comparison "
            "for the complete input range"
        )
    return span, integer_power


def phase_grid_step(n: int, phase_span: float) -> float:
    """Return the phase between adjacent qcmp integer values.

    [PAPER P2, QCMP A1] The largest n-bit integer maps to ``phase_span``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return phase_span / ((1 << n) - 1)


def little_endian_phase_weight(
    n: int,
    qubit_index: int,
    phase_span: float,
) -> float:
    """Convert the paper's MSB-first beta_i to qcmp's little-endian order."""
    if qubit_index < 0 or qubit_index >= n:
        raise IndexError("qubit_index is outside the register")
    return phase_grid_step(n, phase_span) * (1 << qubit_index)


def apply_fixed_phase_offset(qc: QuantumCircuit, target) -> None:
    """[PAPER P1] Center equality at probability one half."""
    qc.p(-pi / 2, target)


def apply_controlled_phase_cascade(
    qc: QuantumCircuit,
    controls,
    target,
    *,
    phase_span: float,
    sign: int,
    power: int,
) -> None:
    """[PAPER P2-P3] Apply one binary-weighted controlled-P cascade.

    ``sign=+1`` supplies ``phi_x`` and ``sign=-1`` supplies ``-phi_y``.
    Multiplying every fixed angle by ``power`` implements Proposition 2.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")

    n = len(controls)
    for j, control in enumerate(controls):
        beta = little_endian_phase_weight(n, j, phase_span)
        qc.cp(sign * power * beta, control, target)


def apply_register_driven_qpc(
    qc: QuantumCircuit,
    a,
    b,
    output,
    *,
    phase_span: float,
    power: int,
) -> None:
    """[PAPER P1-P3, Figure 2] Apply the complete coherent QPC circuit."""
    if len(a) != len(b):
        raise ValueError("operand registers must have equal width")

    qc.h(output)
    apply_fixed_phase_offset(qc, output)
    apply_controlled_phase_cascade(
        qc,
        a,
        output,
        phase_span=phase_span,
        sign=+1,
        power=power,
    )
    apply_controlled_phase_cascade(
        qc,
        b,
        output,
        phase_span=phase_span,
        sign=-1,
        power=power,
    )
    qc.h(output)


def decode_phase_comparator_counts(
    counts: dict[str, int],
    *,
    output_index: int,
) -> dict[str, int]:
    """[PAPER P1, QCMP A2] Decide whether the empirical Pr(0) is below 1/2."""
    if not counts:
        raise ValueError("cannot decode empty counts")
    selected = marginal_counts(counts, indices=[output_index])
    shots = sum(selected.values())
    if shots == 0:
        raise ValueError("cannot decode empty counts")
    zero_count = selected.get("0", 0)
    return {"lt": int(2 * zero_count < shots)}


class LT_Berti_QPC:
    """Qcmp adapter for the register-driven QPC of Berti and Poggiali."""

    meta = ComparatorMeta(
        key="lt_berti_qpc",
        name="Register-driven Quantum Phase-based Comparator",
        paper=(
            'A. Berti and A. Poggiali, "A Quantum Phase-based Comparator", '
            "arXiv:2609.25262v1 [quant-ph] (2026)."
        ),
        kind="lt",
        signed=False,
        outputs=("lt",),
        ancilla_mode="clean",
        notes=(
            "Implements Figure 2. The output is statistical: lt=1 iff the "
            "observed frequency of zero is below one half. The paper excludes "
            "equality, and close values require more shots. Arbitrary-angle "
            "controlled-phase gates are counted as logical gates by qcmp."
        ),
    )

    def build(
        self,
        n: int,
        *,
        phase_span: float = DEFAULT_PHASE_SPAN,
        power: int = 1,
        measure: bool = True,
        **kwargs,
    ) -> BuildOutput:
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be an integer >= 1")
        span, integer_power = validate_phase_parameters(phase_span, power)

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        out = QuantumRegister(1, "out")

        if measure:
            c_lt = ClassicalRegister(1, "c_lt")
            qc = QuantumCircuit(a, b, out, c_lt, name=self.meta.key)
        else:
            c_lt = None
            qc = QuantumCircuit(a, b, out, name=self.meta.key)

        apply_register_driven_qpc(
            qc,
            a,
            b,
            out[0],
            phase_span=span,
            power=integer_power,
        )

        decode_counts = None
        if c_lt is not None:
            qc.measure(out[0], c_lt[0])
            output_index = qc.clbits.index(c_lt[0])

            def decode_counts(counts):
                return decode_phase_comparator_counts(
                    counts,
                    output_index=output_index,
                )

        regs = {
            "a": a,
            "b": b,
            "out": out,
            "out_qubit": out[0],
        }
        if c_lt is not None:
            regs["c_lt"] = c_lt

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0],),
        )
        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=resources,
        )
