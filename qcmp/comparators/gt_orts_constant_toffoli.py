"""Gate-encoded-A half comparator from Orts and Gil-Merino (2026).

Paper
-----
F. Orts and R. Gil-Merino, "A Framework for Quantum Circuit
Optimization: Comparators as a Case Study", IEEE Transactions on Computers,
75(2), 706-719 (2026), doi:10.1109/TC.2025.3642981.  The supplied source is
``7-Orts.pdf``.

PROVENANCE AND ASSUMPTIONS - search for [PAPER] and [IMPLEMENTATION].
-------------------------------------------------------------------
[PAPER P1] Section IV-B and Figs. 4-6 describe a half comparator whose output
is one exactly when A > B.  B is stored in an N-qubit register.  A is a
compile-time basis-state value: every ``a_i gate`` is X when a_i=1 and an
identity gate otherwise.  The paper explicitly permits omitting those identity
gates.  The construction therefore uses 2*N qubits: N for B and N work/output
qubits.

[PAPER P2] Steps 2-5 give the forward CNOT/Toffoli network.  The result is the
last work qubit.  Figure 6 applies Bennett-style uncomputation to every gate
which does not target that result, restoring B and the other work qubits.

[IMPLEMENTATION I1] Identity gates are omitted.  ``encode_constant_bit`` is
kept as a separate function so every place where the paper inserts A is
visible and its X cost is not hidden in classical preprocessing.

[IMPLEMENTATION I2] Figure 4 measures the result before drawing the cleanup,
because the paper assumes classical basis-state inputs.  Here measurement is
postponed until after cleanup.  Nothing after result production targets or
controls that qubit, so this preserves the paper's classical results and also
leaves a useful coherent comparator when ``measure=False``.

[IMPLEMENTATION I3] The written general construction accesses a_2 in step 3,
so it directly specifies N>=3.  Table V nevertheless reports N=2.  For N=2,
``compute_two_bit_gt`` implements the same Boolean predicate directly with
X/CX/CCX gates, keeps the advertised 2*N qubits, and restores B.  This small
case is explicitly our specialization; it is not claimed to reproduce an
undrawn two-bit circuit gate for gate.

[IMPLEMENTATION I4] ``build(n)`` without ``a_value`` compiles A=2**n-1, the
paper's all-ones worst-case convention.  Qcmp's runner supplies its normal
``A`` argument as ``a_value`` because this plugin declares
``constant_operands=("a",)``.  No runner or base-class change is needed beyond
the support already introduced for the preceding constant-operand comparator.
"""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


# 1. Constant encoding used by every red ``a_i gate`` in Fig. 4.

def encode_constant_bit(qc: QuantumCircuit, wire, bit: int) -> None:
    """[PAPER P1] XOR one classical A bit into ``wire`` using X or omitted I."""
    if bit not in (0, 1):
        raise ValueError("A constant bit must be 0 or 1")
    if bit:
        qc.x(wire)


def _a_bit(a_value: int, index: int) -> int:
    return (a_value >> index) & 1


def encode_a_on_b_except_lsb(
    qc: QuantumCircuit, b, a_value: int,
) -> None:
    """[PAPER P1, step 2] Apply a_j to b_j for j=N-1,...,1."""
    for j in range(len(b) - 1, 0, -1):
        encode_constant_bit(qc, b[j], _a_bit(a_value, j))


# 2. Forward network from steps 2-5.

def compute_initial_pair(
    qc: QuantumCircuit, b, work, a_value: int,
) -> None:
    """[PAPER P2, step 2] Seed the comparison in ``work[1]``.

    On exit, ``work[0]`` is zero and ``work[1]`` contains the first comparison
    term.  The higher B wires contain b_j XOR a_j until cleanup.
    """
    a0 = _a_bit(a_value, 0)
    a1 = _a_bit(a_value, 1)

    encode_constant_bit(qc, work[0], a0)
    encode_constant_bit(qc, work[1], a1)
    encode_a_on_b_except_lsb(qc, b, a_value)
    qc.cx(work[0], b[0])
    qc.ccx(b[0], work[0], work[1])
    encode_constant_bit(qc, work[0], a0)


def compute_middle_stage(
    qc: QuantumCircuit,
    b_bit,
    previous,
    target,
    a_i: int,
    a_next: int,
) -> None:
    """[PAPER P2, steps 3-4] Advance one internal comparison stage."""
    encode_constant_bit(qc, target, a_next)
    encode_constant_bit(qc, target, a_i)
    qc.cx(previous, b_bit)
    qc.ccx(b_bit, previous, target)


def compute_output_stage(
    qc: QuantumCircuit, b_msb, previous, output, a_msb: int,
) -> None:
    """[PAPER P2, step 5] Produce the A>B bit in ``output``."""
    qc.cx(previous, b_msb)
    qc.ccx(b_msb, previous, output)
    encode_constant_bit(qc, output, a_msb)


def _middle_wires(work, index: int):
    """Return the incoming and target wires for internal bit ``index``."""
    if index == 1:
        return work[1], work[0]
    if index == 2:
        return work[0], work[2]
    return work[index - 1], work[index]


def compute_paper_gt(
    qc: QuantumCircuit, b, work, a_value: int,
):
    """[PAPER P2] Execute the N>=3 forward network and return its output wire."""
    n = len(b)
    if n < 3:
        raise ValueError("The paper's written steps directly require n >= 3")

    compute_initial_pair(qc, b, work, a_value)
    for i in range(1, n - 1):
        previous, target = _middle_wires(work, i)
        compute_middle_stage(
            qc,
            b[i],
            previous,
            target,
            _a_bit(a_value, i),
            _a_bit(a_value, i + 1),
        )

    # The final middle-stage target is the input to step 5.
    previous = work[0] if n == 3 else work[n - 2]
    output = work[n - 1]
    compute_output_stage(
        qc, b[n - 1], previous, output, _a_bit(a_value, n - 1),
    )
    return output


# 3. Bennett cleanup from Fig. 6, excluding gates which target the result.

def uncompute_middle_stage(
    qc: QuantumCircuit,
    b_bit,
    previous,
    target,
    a_i: int,
    a_next: int,
) -> None:
    """[PAPER P2, Fig. 6] Invert one internal stage exactly."""
    qc.ccx(b_bit, previous, target)
    qc.cx(previous, b_bit)
    encode_constant_bit(qc, target, a_i)
    encode_constant_bit(qc, target, a_next)


def uncompute_initial_pair(
    qc: QuantumCircuit, b, work, a_value: int,
) -> None:
    """[PAPER P2, Fig. 6] Invert step 2 and restore B and both seed wires."""
    a0 = _a_bit(a_value, 0)
    a1 = _a_bit(a_value, 1)

    encode_constant_bit(qc, work[0], a0)
    qc.ccx(b[0], work[0], work[1])
    qc.cx(work[0], b[0])
    encode_a_on_b_except_lsb(qc, b, a_value)
    encode_constant_bit(qc, work[1], a1)
    encode_constant_bit(qc, work[0], a0)


def uncompute_paper_gt_garbage(
    qc: QuantumCircuit, b, work, a_value: int,
) -> None:
    """[PAPER P2, Fig. 6] Restore every wire except the N>=3 GT output."""
    n = len(b)
    if n < 3:
        raise ValueError("The paper's written steps directly require n >= 3")

    previous = work[0] if n == 3 else work[n - 2]
    # Reverse only the output stage's data CNOT.  Its CCX and a_MSB gate target
    # the result and are deliberately retained, exactly as described below Fig. 6.
    qc.cx(previous, b[n - 1])

    for i in range(n - 2, 0, -1):
        previous, target = _middle_wires(work, i)
        uncompute_middle_stage(
            qc,
            b[i],
            previous,
            target,
            _a_bit(a_value, i),
            _a_bit(a_value, i + 1),
        )

    uncompute_initial_pair(qc, b, work, a_value)


# 4. Explicitly identified two-bit specialization (not drawn in the paper).

def compute_two_bit_gt(
    qc: QuantumCircuit, b, work, a_value: int,
):
    """[IMPLEMENTATION I3] Clean N=2 realization of ``A > B``.

    The four possible compile-time values of A reduce to small Boolean
    functions.  ``work[1]`` is the output and ``work[0]`` remains zero.
    """
    output = work[1]
    if a_value == 0:
        return output
    if a_value == 1:
        # A>B iff B=00.
        qc.x(b[0])
        qc.x(b[1])
        qc.ccx(b[0], b[1], output)
        qc.x(b[1])
        qc.x(b[0])
        return output
    if a_value == 2:
        # A>B iff the most-significant bit of B is zero.
        qc.x(output)
        qc.cx(b[1], output)
        return output

    # A=3: A>B except when B=3.
    qc.x(output)
    qc.ccx(b[0], b[1], output)
    return output


# 5. Qcmp adapter.  This is the only class that needs registering.

class GT_Orts_Constant_Toffoli:
    meta = ComparatorMeta(
        key="gt_orts_constant_toffoli",
        name="Orts gate-encoded-A Toffoli comparator (2026)",
        paper=(
            'F. Orts and R. Gil-Merino, "A Framework for Quantum Circuit '
            'Optimization: Comparators as a Case Study", IEEE Transactions '
            'on Computers 75(2), 706-719 (2026). '
            'doi:10.1109/TC.2025.3642981.'
        ),
        kind="gt",
        signed=False,
        outputs=("gt",),
        ancilla_mode="clean",
        notes=(
            "A is compiled into X/omitted-I operations; only B is a quantum "
            "operand. The N>=3 CNOT/Toffoli network follows Section IV-B and "
            "Figs. 4-6. N=2 is an explicitly documented direct specialization. "
            "build(n) defaults to all-one A for worst-case equations."
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
            raise ValueError("This implementation supports n >= 2")
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
        work = QuantumRegister(n, "work")
        qc = QuantumCircuit(b, work, name=self.meta.key)

        if n == 2:
            output = compute_two_bit_gt(qc, b, work, a_value)
        else:
            output = compute_paper_gt(qc, b, work, a_value)
            uncompute_paper_gt_garbage(qc, b, work, a_value)

        regs = {
            "b": b,
            "work": work,
            "out_qubit": output,
            "out_qubits": {"gt": output},
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
                restored_zero_qubits=tuple(q for q in work if q != output),
            ),
        )
