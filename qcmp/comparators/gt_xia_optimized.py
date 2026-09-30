from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _open_ccx(qc: QuantumCircuit, closed_control, open_control, target) -> None:
    """Toffoli with the second control active on |0>."""
    qc.x(open_control)
    qc.ccx(closed_control, open_control, target)
    qc.x(open_control)


def _forward_slice(qc: QuantumCircuit, carry, xor_bit, a_bit) -> None:
    """One low-to-high comparison slice from Xia et al.'s recurrence."""
    qc.cx(a_bit, carry)
    _open_ccx(qc, carry, xor_bit, a_bit)


def _inverse_slice(qc: QuantumCircuit, carry, xor_bit, a_bit) -> None:
    """Inverse of _forward_slice, restoring the operand and prior carry."""
    _open_ccx(qc, carry, xor_bit, a_bit)
    qc.cx(a_bit, carry)


def _write_final_result(qc: QuantumCircuit, carry, xor_bit, a_bit, out) -> None:
    """Copy the final A>B recurrence value to `out` without dirtying carry."""
    qc.cx(a_bit, carry)
    qc.cx(a_bit, out)
    _open_ccx(qc, carry, xor_bit, out)
    qc.cx(a_bit, carry)


class GT_Xia_Optimized:
    meta = ComparatorMeta(
        key="gt_xia_optimized",
        name="Optimized ripple comparator (Xia et al., 2019)",
        paper=(
            'H.-Y. Xia et al., "Design and simulation of quantum image '
            'binarization using quantum comparator", Modern Physics Letters A '
            '35, 2050049 (2020).'
        ),
        kind="gt",
        signed=False,
        outputs=("gt",),
        ancilla_mode="clean",
        notes=(
            "Implements the n-bit recurrence in Figures 8-11: the final x1 "
            "bit is 1 iff A>B. A single clean carry auxiliary is propagated "
            "from the least to the most significant bit, then uncomputed to "
            "preserve both operands."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        carry = QuantumRegister(1, "carry")
        out = QuantumRegister(1, "x1")
        c_out = ClassicalRegister(1, "c_out")
        qc = QuantumCircuit(a, b, carry, out, c_out, name=self.meta.key)

        # The paper computes the relation from low to high bits. Store
        # a_i XOR b_i in b_i, leaving a_i available as the recurrence target.
        for bit in range(n):
            qc.cx(a[bit], b[bit])

        # With b_i=a_i XOR b_i, b_i AND a_i is a_i AND NOT b_i: the
        # first (least-significant) A>B relation.
        if n == 1:
            qc.ccx(b[0], a[0], out[0])
        else:
            qc.ccx(b[0], a[0], carry[0])

            for bit in range(1, n - 1):
                previous = carry[0] if bit == 1 else a[bit - 1]
                _forward_slice(qc, previous, b[bit], a[bit])

            previous = carry[0] if n == 2 else a[n - 2]
            _write_final_result(qc, previous, b[n - 1], a[n - 1], out[0])

            for bit in range(n - 2, 0, -1):
                previous = carry[0] if bit == 1 else a[bit - 1]
                _inverse_slice(qc, previous, b[bit], a[bit])

            qc.ccx(b[0], a[0], carry[0])

        for bit in range(n - 1, -1, -1):
            qc.cx(a[bit], b[bit])

        qc.measure(out[0], c_out[0])

        def decode_counts(counts):
            index = qc.clbits.index(c_out[0])
            marginal = marginal_counts(counts, indices=[index])
            return {"gt": int(max(marginal, key=marginal.get))}

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0],),
            restored_zero_qubits=(carry[0],),
        )
        return BuildOutput(
            circuit=qc,
            regs={
                "a": a,
                "b": b,
                "carry": carry,
                "out": out,
                "out_qubit": out[0],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
