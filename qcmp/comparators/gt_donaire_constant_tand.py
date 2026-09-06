"""N+2-qubit gate-encoded comparator from Donaire et al. (2025).

Paper
-----
L. M. Donaire, G. Ortega, E. M. Garzon, F. Orts, R. Paulavicius and
E. Filatovas, "Halving the Number of Qubits of Quantum Comparators",
PPAM 2024, LNCS 15580, pp. 290-300 (2025),
doi:10.1007/978-3-031-85700-3_21.  The supplied source is
``978-3-031-85700-3_21.pdf``.

PROVENANCE AND ASSUMPTIONS - search for [PAPER] and [IMPLEMENTATION].
-------------------------------------------------------------------
[PAPER P1] Section 3 and Fig. 3 define a half comparator with output one
exactly when A > B.  B occupies an N-qubit input register.  A must be known
when the circuit is built: each labelled ``a_i`` box is X when that bit is one
and no operation when it is zero.  Two work qubits are recycled for every N,
giving the advertised N+2 total qubits.

[PAPER P2] The comparator uses N temporary logical-AND blocks.  Figure 1
provides their complete four-T Clifford+T decomposition.  After every block
except the last, the previous work qubit is reset and the logical roles of the
two work wires are exchanged.  Consequently the paper reports T-count=4*N,
T-depth=2*N and explicitly requires a device with reset support.

[IMPLEMENTATION I1] ``prepare_t_state`` and ``temporary_logical_and`` spell
out Fig. 1 gate by gate.  Identity gates are omitted as the paper permits.
``reset_recycled_work`` remains separate so the non-unitary operation is
impossible to overlook in resource inspection.

[IMPLEMENTATION I2] The numbered steps do not explicitly repeat the a_i gates
on B at the end, although Fig. 3 draws them and labels the final wires b_i.
``restore_b_register`` implements those gates, so the runtime operand B is
restored and is not garbage.

[IMPLEMENTATION I3] Step 7 says a_i after the loop has incremented i to N.
Figure 3 unambiguously labels the final output gate a_(N-1); that is the index
used here.  The described construction starts with a_0 and a_1, so this module
supports N>=2.

[IMPLEMENTATION I4] A Qiskit measurement is added only after the final B
restoration.  This does not change basis-state results.  ``measure=False``
removes only that final measurement; the intermediate resets required by the
paper remain.

IMPORTANT LIMITATION
--------------------
Resetting a work qubit which can be correlated with B makes this construction
a non-unitary channel on superposed B inputs, not a coherent comparison oracle.
Qcmp's normal runner prepares classical basis-state integers, for which the
paper circuit is deterministic and all resets are safe.  This module does not
silently replace the published reset network with a different unitary design.

``build(n)`` without ``a_value`` compiles A=2**n-1 as the all-ones convention
for generic resource equations.  The runner supplies its normal ``A`` argument
as ``a_value`` because the metadata declares ``constant_operands=("a",)``.
The constant-operand support added for the previous comparators is sufficient;
no further library change is required.
"""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


# 1. Classical A bits reintroduced as the labelled gates in Fig. 3.

def encode_constant_bit(qc: QuantumCircuit, wire, bit: int) -> None:
    """[PAPER P1] XOR a classical bit into ``wire`` using X or omitted I."""
    if bit not in (0, 1):
        raise ValueError("A constant bit must be 0 or 1")
    if bit:
        qc.x(wire)


def _a_bit(a_value: int, index: int) -> int:
    return (a_value >> index) & 1


def encode_a_into_b(qc: QuantumCircuit, b, a_value: int) -> None:
    """[PAPER P1, step 2] Transform every b_i into a_i XOR b_i."""
    for i, wire in enumerate(b):
        encode_constant_bit(qc, wire, _a_bit(a_value, i))


def restore_b_register(qc: QuantumCircuit, b, a_value: int) -> None:
    """[PAPER Fig. 3 + IMPLEMENTATION I2] Restore every original b_i."""
    for i, wire in enumerate(b):
        encode_constant_bit(qc, wire, _a_bit(a_value, i))


# 2. Temporary logical-AND from Fig. 1.

def prepare_t_state(qc: QuantumCircuit, target) -> None:
    """[PAPER P2] Prepare T|+> on a clean temporary-AND target."""
    qc.h(target)
    qc.t(target)


def temporary_logical_and(
    qc: QuantumCircuit, first_control, second_control, target,
) -> None:
    """[PAPER P2, Fig. 1] Compute control_1 AND control_2 into target.

    ``target`` must be zero before ``prepare_t_state`` is called.  Together the
    preparation and this function use exactly four T/Tdg instructions.
    """
    qc.cx(first_control, target)
    qc.cx(second_control, target)
    qc.cx(target, first_control)
    qc.cx(target, second_control)
    qc.tdg(first_control)
    qc.tdg(second_control)
    qc.t(target)
    qc.cx(target, first_control)
    qc.cx(target, second_control)
    qc.h(target)
    qc.s(target)


def compute_temporary_logical_and(
    qc: QuantumCircuit, first_control, second_control, target,
) -> None:
    """[PAPER P2] Prepare the target and apply the complete Fig. 1 block."""
    prepare_t_state(qc, target)
    temporary_logical_and(qc, first_control, second_control, target)


def reset_recycled_work(qc: QuantumCircuit, wire) -> None:
    """[PAPER P2] Reset the consumed work wire before exchanging its role."""
    qc.reset(wire)


# 3. Comparator recurrence from the numbered construction steps.

def compute_least_significant_stage(
    qc: QuantumCircuit, b, first_work, second_work, a_value: int,
):
    """[PAPER steps 2-3] Seed the ripple and return (current, spare)."""
    encode_a_into_b(qc, b, a_value)
    a0 = _a_bit(a_value, 0)
    encode_constant_bit(qc, first_work, a0)
    compute_temporary_logical_and(qc, b[0], first_work, second_work)
    encode_constant_bit(qc, first_work, a0)
    return second_work, first_work


def advance_second_bit(
    qc: QuantumCircuit, b_bit, current, spare, a_value: int,
):
    """[PAPER steps 4-5] Process bit one and exchange the work roles."""
    encode_constant_bit(qc, current, _a_bit(a_value, 1))
    qc.cx(current, b_bit)
    compute_temporary_logical_and(qc, current, b_bit, spare)
    qc.cx(current, b_bit)
    reset_recycled_work(qc, current)
    return spare, current


def advance_higher_bit(
    qc: QuantumCircuit,
    b_bit,
    current,
    spare,
    previous_a_bit: int,
    current_a_bit: int,
):
    """[PAPER step 6] Process one bit i>=2 and exchange the work roles."""
    encode_constant_bit(qc, current, previous_a_bit)
    encode_constant_bit(qc, current, current_a_bit)
    qc.cx(current, b_bit)
    compute_temporary_logical_and(qc, current, b_bit, spare)
    qc.cx(current, b_bit)
    reset_recycled_work(qc, current)
    return spare, current


def compute_raw_gt(
    qc: QuantumCircuit, b, work, a_value: int,
):
    """[PAPER steps 2-7] Compute A>B, restore B and return the output wire."""
    n = len(b)
    if n < 2:
        raise ValueError("The published recurrence requires n >= 2")

    current, spare = compute_least_significant_stage(
        qc, b, work[0], work[1], a_value,
    )
    current, spare = advance_second_bit(
        qc, b[1], current, spare, a_value,
    )

    for i in range(2, n):
        current, spare = advance_higher_bit(
            qc,
            b[i],
            current,
            spare,
            _a_bit(a_value, i - 1),
            _a_bit(a_value, i),
        )

    # [IMPLEMENTATION I3] Figure 3 identifies this as a_(N-1).
    encode_constant_bit(qc, current, _a_bit(a_value, n - 1))
    restore_b_register(qc, b, a_value)
    return current, spare


# 4. Qcmp adapter.  This is the only class that needs registering.

class GT_Donaire_Constant_TAND:
    meta = ComparatorMeta(
        key="gt_donaire_constant_tand",
        name="Donaire N+2-qubit gate-encoded temporary-AND comparator (2025)",
        paper=(
            'L. M. Donaire et al., "Halving the Number of Qubits of Quantum '
            'Comparators", PPAM 2024, LNCS 15580, 290-300 (2025). '
            'doi:10.1007/978-3-031-85700-3_21.'
        ),
        kind="gt",
        signed=False,
        outputs=("gt",),
        ancilla_mode="clean",
        notes=(
            "A is compiled into X/omitted-I operations; only B is a quantum "
            "operand. The paper's N+2-qubit network recycles two work wires "
            "with reset, so it is deterministic for basis inputs but is not a "
            "coherent comparison oracle. N>=2; B and the spare wire are clean."
        ),
        constant_operands=("a",),
    )

    def build(
        self,
        n: int,
        a_value: int | None = None,
        measure: bool = True,
        **kwargs,
    ) -> BuildOutput:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be an integer")
        n = int(n)
        if n < 2:
            raise ValueError("The paper's comparator requires n >= 2")
        if a_value is None:
            a_value = (1 << n) - 1
        if isinstance(a_value, bool) or not isinstance(a_value, Integral):
            raise TypeError("a_value must be an integer")
        a_value = int(a_value)
        if a_value < 0 or a_value >= 1 << n:
            raise ValueError(f"a_value {a_value} does not fit in {n} bits")
        if not isinstance(measure, bool):
            raise TypeError("measure must be a bool")
        if kwargs:
            raise TypeError(f"Unexpected build argument(s): {', '.join(sorted(kwargs))}")

        b = QuantumRegister(n, "b")
        work = QuantumRegister(2, "work")
        qc = QuantumCircuit(b, work, name=self.meta.key)

        output, spare = compute_raw_gt(qc, b, work, a_value)

        regs = {
            "b": b,
            "work": work,
            "out_qubit": output,
            "out_qubits": {"gt": output},
            "spare_qubit": spare,
            "constant_values": {"a": a_value},
        }
        decode_counts = None
        if measure:
            c_out = ClassicalRegister(1, "c_out")
            qc.add_register(c_out)
            regs["c_out"] = c_out
            qc.measure(output, c_out[0])
            output_index = qc.find_bit(c_out[0]).index

            def decode_counts(counts):
                if not counts:
                    raise ValueError("Cannot decode empty measurement counts")
                selected = marginal_counts(counts, indices=[output_index])
                return {"gt": int(max(selected, key=selected.get))}

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("b",),
                solution_qubits=(output,),
                restored_zero_qubits=(spare,),
            ),
        )
