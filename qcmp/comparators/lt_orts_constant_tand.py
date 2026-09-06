"""Half-comparator with a gate-encoded constant from Orts et al. (2023).

Paper: F. Orts, E. Filatovas, E. M. Garzon and G. Ortega, "A quantum
circuit to generate random numbers within a specific interval",
EPJ Quantum Technology 10, 17 (2023), doi:10.1140/epjqt/s40507-023-00174-1.
PDF supplied as 6-Orts.pdf.

PROVENANCE AND ASSUMPTIONS - search for [PAPER] and [IMPLEMENTATION].
-------------------------------------------------------------------
[PAPER P1] Section 3.1 and Fig. 2 specify N>2. Only A is represented by an
N-qubit input register. B is a compile-time natural number: its bit b_i is
introduced on one reusable wire with X if b_i=1 (I/omission otherwise),
then removed with the same gate. This is the source of the 2*N+1 qubit count.

[PAPER P2] A is inverted and the circuit propagates the carry of NOT(A)+B
from the least to the most significant bit. Temporary logical-AND gates are
used for the N carry computations; the first N-1 are measure-uncomputed.

[PAPER ERRATUM / VERIFIED] The prose and Fig. 2 label the result as 0 when
A<B and 1 otherwise. Direct Boolean derivation and exhaustive simulation of
the written recurrence instead give 1 exactly when A<B, including equality
as 0. This plugin exposes the PHYSICAL circuit result as ``lt``. It does not
hide the discrepancy by complementing it in the decoder. See
``paper_claimed_ge_from_raw_lt`` for the explicit optional conversion.

[IMPLEMENTATION I1] The paper draws temporary logical-AND blocks and cites
Gidney. Their explicit Clifford+T synthesis below uses the same four-T compute
sequence as qcmp's 2021 Orts comparator. Its old one-control Z fixup is NOT
reused: erasing a coherently computed AND by X-basis measurement requires a
conditional CZ between both controls. This is verified on superpositions.

[IMPLEMENTATION I2] The measure-and-fixup uncomputation leaves its measured
target in a classical 0/1 state. We explicitly reset that consumed target to
|0> so ResourceLayout can truthfully declare it reusable. Qcmp assigns no
gate cost/delay to reset. This reset is not included in the paper's gate list.

[IMPLEMENTATION I3] ``build(n)`` without b_value compiles B=2**n-1. This is
only a documented worst-case convention for the generic equations command;
the runner always supplies its --B value. Exact X counts depend on the set
bits of B. N=1 and N=2 are rejected rather than silently extending a design
which the paper states for N>2.

REPORTED VERSUS IMPLEMENTED RESOURCES
------------------------------------
The paper treats a temporary logical-AND as one block and reports, for the
worst-case B with all bits one: qubits=2*N+1, X=6*N-2, outer CX=8*N-11,
temporary-AND=N, its uncomputation=N-1, T-count=4*N and T-depth=2*N.

This module emits the internals of each block. Therefore qcmp sees additional
H/S/CX gates, plus N-1 measurements and resets. The paper's T-count remains
directly testable. No hardware routing or SWAP cost is included.

INSTALLATION
------------
Register once in qcmp/comparators/__init__.py:
  from qcmp.comparators.lt_orts_constant_tand import LT_Orts_Constant_TAND
  register(LT_Orts_Constant_TAND())

Then the unchanged CLI meaning is preserved: --A is loaded into the quantum
register and --B is compiled into X/I choices by the runner/library extension.
  python -m qcmp run --cmp lt_orts_constant_tand --n 4 --A 3 --B 7
"""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


# 1. Constant encoding - explicitly separate because B is not an input register.

def encode_constant_bit(qc: QuantumCircuit, wire, bit: int) -> None:
    """[PAPER P1] XOR the classical bit into ``wire`` using X or omitted I."""
    if bit not in (0, 1):
        raise ValueError("A constant bit must be 0 or 1")
    if bit:
        qc.x(wire)


# 2. Temporary logical-AND and its measurement-based inverse.

def prepare_t_state(qc: QuantumCircuit, target) -> None:
    """[PAPER P2] Prepare (|0> + exp(i*pi/4)|1>)/sqrt(2) = T|+>."""
    qc.h(target)
    qc.t(target)


def temporary_logical_and(qc: QuantumCircuit, x, y, target) -> None:
    """[IMPLEMENTATION I1] Exact AND into a prepared target; four T/Tdg total.

    ``prepare_t_state`` must be called first. For computational x/y and that
    prepared target, the result is |x,y,x AND y> with amplitude +1.
    """
    qc.cx(x, target)
    qc.cx(y, target)
    qc.cx(target, x)
    qc.cx(target, y)
    qc.tdg(x)
    qc.tdg(y)
    qc.t(target)
    qc.cx(target, x)
    qc.cx(target, y)
    qc.h(target)
    qc.s(target)


def uncompute_temporary_logical_and(
    qc: QuantumCircuit, x, y, target, result_bit,
) -> None:
    """[PAPER P2 + IMPLEMENTATION I2] Measure/fixup AND inverse.

    If the X-basis measurement is one it contributes (-1)**(x*y), so a CZ
    between BOTH controls removes that phase. The explicit reset makes the
    measured target reusable and is separately visible in the circuit.
    """
    qc.h(target)
    qc.measure(target, result_bit)
    with qc.if_test((result_bit, 1)):
        qc.cz(x, y)
    qc.reset(target)


# 3. Paper recurrence with B inserted gate by gate.

def compute_raw_lt(
    qc: QuantumCircuit, a, b_wire, carries, b_value: int,
) -> None:
    """[PAPER P1/P2] Forward pass; leaves raw ``A < B`` in carries[-1]."""
    n = len(a)
    for qubit in a:
        qc.x(qubit)

    for i in range(n):
        bit = (b_value >> i) & 1
        encode_constant_bit(qc, b_wire, bit)
        if i:
            qc.cx(carries[i - 1], a[i])
            qc.cx(carries[i - 1], b_wire)

        prepare_t_state(qc, carries[i])
        temporary_logical_and(qc, a[i], b_wire, carries[i])

        if i:
            # Restore the encoded B wire, then accumulate the previous carry.
            qc.cx(carries[i - 1], b_wire)
            qc.cx(carries[i - 1], carries[i])
        encode_constant_bit(qc, b_wire, bit)


def uncompute_raw_lt_garbage(
    qc: QuantumCircuit, a, b_wire, carries, fix_bits, b_value: int,
) -> None:
    """[PAPER P2 + IMPLEMENTATION I2] Clean every qubit except raw LT."""
    n = len(a)

    # Reverse the only data CNOT of the final iteration. Its carry is output.
    qc.cx(carries[n - 2], a[n - 1])

    # Reverse complete iterations N-2 ... 1.
    for i in range(n - 2, 0, -1):
        bit = (b_value >> i) & 1
        encode_constant_bit(qc, b_wire, bit)
        qc.cx(carries[i - 1], carries[i])
        qc.cx(carries[i - 1], b_wire)
        uncompute_temporary_logical_and(
            qc, a[i], b_wire, carries[i], fix_bits[i],
        )
        qc.cx(carries[i - 1], b_wire)
        qc.cx(carries[i - 1], a[i])
        encode_constant_bit(qc, b_wire, bit)

    # Reverse the first AND, which has no incoming carry.
    encode_constant_bit(qc, b_wire, b_value & 1)
    uncompute_temporary_logical_and(
        qc, a[0], b_wire, carries[0], fix_bits[0],
    )
    encode_constant_bit(qc, b_wire, b_value & 1)

    for qubit in a:
        qc.x(qubit)


def paper_claimed_ge_from_raw_lt(qc: QuantumCircuit, raw_lt) -> None:
    """[IMPLEMENTATION] Optional X making GE=NOT(LT), not used by the plugin.

    This is the explicit operation required to obtain the polarity printed in
    the prose/Fig. 2. Calling it adds one X and changes the plugin semantics.
    """
    qc.x(raw_lt)


# 4. qcmp adapter.

class LT_Orts_Constant_TAND:
    meta = ComparatorMeta(
        key="lt_orts_constant_tand",
        name="Orts gate-encoded constant comparator with temporary AND (2023)",
        paper=(
            'F. Orts et al., "A quantum circuit to generate random numbers '
            'within a specific interval", EPJ Quantum Technology 10, 17 '
            '(2023). doi:10.1140/epjqt/s40507-023-00174-1.'
        ),
        kind="lt",
        signed=False,
        outputs=("lt",),
        ancilla_mode="clean",
        notes=(
            "B is compiled into X/I operations: only A is a quantum operand. "
            "The written recurrence was exhaustively verified to output A<B, "
            "opposite to the polarity printed in the paper. Measured temporary-"
            "AND targets are explicitly reset; N>2 as specified by the paper. "
            "build(n) defaults to all-one B only for worst-case equations."
        ),
        constant_operands=("b",),
    )

    def build(
        self,
        n: int,
        b_value: int | None = None,
        measure: bool = True,
        **kwargs,
    ) -> BuildOutput:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be an integer")
        n = int(n)
        if n <= 2:
            raise ValueError("The paper specifies this comparator only for n > 2")
        if b_value is None:
            b_value = (1 << n) - 1
        if isinstance(b_value, bool) or not isinstance(b_value, Integral):
            raise TypeError("b_value must be an integer")
        b_value = int(b_value)
        if b_value < 0 or b_value >= 1 << n:
            raise ValueError(f"b_value {b_value} does not fit in {n} bits")
        if not isinstance(measure, bool):
            raise TypeError("measure must be a bool")
        if kwargs:
            raise TypeError(f"Unexpected build argument(s): {', '.join(sorted(kwargs))}")

        a = QuantumRegister(n, "a")
        b_wire = QuantumRegister(1, "b_constant_wire")
        carries = QuantumRegister(n, "carry")
        fix = ClassicalRegister(n - 1, "c_fix")
        qc = QuantumCircuit(a, b_wire, carries, fix, name=self.meta.key)

        compute_raw_lt(qc, a, b_wire[0], carries, b_value)
        uncompute_raw_lt_garbage(qc, a, b_wire[0], carries, fix, b_value)

        out = carries[n - 1]
        regs = {
            "a": a,
            "b_constant_wire": b_wire,
            "carry": carries,
            "out_qubit": out,
            "out_qubits": {"lt": out},
            "constant_values": {"b": b_value},
            "c_fix": fix,
        }
        decode_counts = None
        if measure:
            c_out = ClassicalRegister(1, "c_out")
            qc.add_register(c_out)
            regs["c_out"] = c_out
            qc.measure(out, c_out[0])
            output_index = qc.find_bit(c_out[0]).index

            def decode_counts(counts):
                if not counts:
                    raise ValueError("Cannot decode empty measurement counts")
                selected = marginal_counts(counts, indices=[output_index])
                return {"lt": int(max(selected, key=selected.get))}

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("a",),
                solution_qubits=(out,),
                restored_zero_qubits=(b_wire[0], *carries[:-1]),
            ),
        )
