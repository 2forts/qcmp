"""Comparator extracted from Yuan et al. (2022), Fig. 4 (PDF page 7).

S. Yuan, W. Zhao, S. Gao, S. Xia, B. Hang and H. Qu,
"An adaptive threshold-based quantum image segmentation algorithm and its
simulation", Quantum Information Processing 21, 359 (2022).
https://doi.org/10.1007/s11128-022-03709-0

Provenance and scope
--------------------
The paper draws the two-bit comparator, not a general n-bit gate sequence.
For n=2 this implementation is Fig. 4, up to commuting independent gates
and expanding its negative control into X-CCX-X. For n>2 it is an explicit
in-place ripple extension of that circuit; n=1 is the single-bit reduction.
It is NOT the different Yuan et al. (2023) QFT comparator.

With little-endian, unsigned operands the unmeasured circuit implements

    |A>|B>|z>|0> -> |A>|B>|z XOR [A < B]>|0>.

For sense='gt', interchange the operands inside the circuit. Equality
therefore yields zero in BOTH modes; the output is never complemented to
change the sense. The work qubit is absent for n=1.

The comparator itself has no reset or intermediate measurement. Only the
default final readout is measured; measure=False returns a coherent circuit
with no classical register. The reset operations elsewhere in the paper's
adder/subtractor/segmentation algorithm are not part of Fig. 4.

Gate accounting, before input preparation or transpilation:
* n=1: 2 CX + 1 CCX; 3 qubits (one is the result).
* n>=2: (4n-1) CX + (2n-1) CCX + (4n-6) X; 2n+2 qubits.
  One auxiliary is the result and one is restored to |0>; no garbage remains.
  Of the CCX gates, 2n-3 originate from negative-control Toffolis.
  Counting each such Toffoli as cost 5 reproduces the paper's 14n-6.
  qcmp counts the explicit X gates too: its default quantum_cost is 18n-12,
  and its default CCX profile gives t_count=14n-7.
"""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _negative_ccx(qc: QuantumCircuit, positive, negative, target) -> None:
    """Flip target iff positive=1 and negative=0, without relative phases."""
    qc.x(negative)
    qc.ccx(positive, negative, target)
    qc.x(negative)


def _borrow_step(qc: QuantumCircuit, carry, a_bit, b_bit) -> None:
    """Store the next borrow in b_bit; retain enough information to invert.

    p = carry XOR b_bit, q = a_bit XOR b_bit,
    next_borrow = b_bit XOR (p AND NOT q).
    """
    qc.cx(b_bit, a_bit)
    qc.cx(b_bit, carry)
    _negative_ccx(qc, carry, a_bit, b_bit)


def _undo_borrow_step(qc: QuantumCircuit, carry, a_bit, b_bit) -> None:
    _negative_ccx(qc, carry, a_bit, b_bit)
    qc.cx(b_bit, carry)
    qc.cx(b_bit, a_bit)


def _append_comparator(qc: QuantumCircuit, a, b, h) -> None:
    """Append the unsigned A<B circuit. h[0] is output; h[1] is workspace."""
    n = len(a)
    out = h[0]

    # First borrow: b_0 AND (a_0 XOR b_0) = NOT(a_0) AND b_0.
    qc.cx(b[0], a[0])
    if n == 1:
        qc.ccx(a[0], b[0], out)
        qc.cx(b[0], a[0])
        return

    work = h[1]
    qc.ccx(a[0], b[0], work)

    # Extension beyond Fig. 4: borrow bits are temporarily stored in b[i].
    # This needs no additional work qubits as n grows.
    for i in range(1, n - 1):
        carry = work if i == 1 else b[i - 1]
        _borrow_step(qc, carry, a[i], b[i])

    # Final stage of Fig. 4: produce the comparison, leaving b_MSB intact.
    carry = work if n == 2 else b[n - 2]
    qc.cx(b[n - 1], a[n - 1])
    qc.cx(b[n - 1], carry)
    qc.cx(b[n - 1], out)
    _negative_ccx(qc, carry, a[n - 1], out)
    qc.cx(b[n - 1], carry)
    qc.cx(b[n - 1], a[n - 1])

    # Reverse the intermediate stages, then clear the first borrow.
    for i in range(n - 2, 0, -1):
        carry = work if i == 1 else b[i - 1]
        _undo_borrow_step(qc, carry, a[i], b[i])

    qc.ccx(a[0], b[0], work)
    qc.cx(b[0], a[0])


class LT_Yuan_Adaptive:
    """Fig. 4 comparator, with an explicitly documented n-bit extension."""

    meta = ComparatorMeta(
        key="lt_yuan_adaptive",
        name="Fig. 4 comparator and ripple extension (Yuan et al., 2022)",
        paper=(
            'S. Yuan, W. Zhao, S. Gao, S. Xia, B. Hang and H. Qu, '
            '"An adaptive threshold-based quantum image segmentation '
            'algorithm and its simulation", Quantum Information Processing '
            '21, 359 (2022). doi:10.1007/s11128-022-03709-0.'
        ),
        kind="lt",
        signed=False,
        outputs=("lt",),
        ancilla_mode="clean",
        notes=(
            "Fig. 4 for n=2; explicit ripple extension for n>2 and reduction "
            "for n=1. Preserves both inputs and clears one work qubit for "
            "n>=2. Default output is lt; sense='gt' swaps operands and "
            "returns gt. measure=False leaves a coherent result. No resets. "
            "Negative controls are expanded into X-CCX-X; qcmp default "
            "quantum_cost=18*n-12 for n>=2, versus 14*n-6 when each "
            "negative-control Toffoli is counted as a primitive of cost 5."
        ),
    )

    def build(
        self,
        n: int,
        sense: str = "lt",
        measure: bool = True,
        **kwargs,
    ) -> BuildOutput:
        """Build for n unsigned bits per operand; inputs are loaded by qcmp.

        Workspace and output start in |0> in normal qcmp use. For coherent
        use, the output may start in either state and is XORed with the
        predicate, while the workspace must still start in |0>.
        """
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be an integer")
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        if sense not in ("lt", "gt"):
            raise ValueError("sense must be 'lt' or 'gt'")
        if not isinstance(measure, bool):
            raise TypeError("measure must be a bool")
        if kwargs:
            raise TypeError(f"Unexpected build argument(s): {', '.join(sorted(kwargs))}")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        h = QuantumRegister(2 if n >= 2 else 1, "h")
        qc = QuantumCircuit(a, b, h, name=self.meta.key)

        left, right = (a, b) if sense == "lt" else (b, a)
        _append_comparator(qc, left, right, h)

        regs = {"a": a, "b": b, "h": h, "out_qubit": h[0]}
        decode_counts = None
        if measure:
            c_out = ClassicalRegister(1, "c_out")
            qc.add_register(c_out)
            qc.measure(h[0], c_out[0])
            regs["c_out"] = c_out
            output_index = qc.clbits.index(c_out[0])

            def decode_counts(counts):
                if not counts:
                    raise ValueError("Cannot decode empty measurement counts")
                output_counts = marginal_counts(counts, indices=[output_index])
                bit = max(output_counts, key=output_counts.get)
                return {sense: int(bit)}

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("a", "b"),
                solution_qubits=(h[0],),
                restored_zero_qubits=tuple(h[1:]),
            ),
        )
