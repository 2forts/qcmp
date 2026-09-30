from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _open_ccx(qc: QuantumCircuit, controls, target, open_indexes: set[int]) -> None:
    for index in open_indexes:
        qc.x(controls[index])
    qc.ccx(controls[0], controls[1], target)
    for index in reversed(list(open_indexes)):
        qc.x(controls[index])


class CMP_Maity_QCC:
    meta = ComparatorMeta(
        key="cmp_maity_qcc",
        name="Two-qubit quantum comparator circuit (Maity, 2022)",
        paper=(
            'H. Maity, "Design and implementation of a two-qubit quantum '
            'comparator circuit (Q-CC)", Journal of Computational Electronics '
            "21, 530-534 (2022)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="clean",
        notes=(
            "Implements the 2-bit Q-CC truth table from Maity. Paper outputs "
            "are X=1 iff A<B, Y=1 iff A=B, and Z=1 iff A>B."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n != 2:
            raise ValueError("cmp_maity_qcc implements Maity's 2-bit Q-CC, so n must be 2")

        a = QuantumRegister(2, "a")
        b = QuantumRegister(2, "b")
        xnor = QuantumRegister(2, "xnor")
        bit_lt = QuantumRegister(2, "bit_lt")
        bit_gt = QuantumRegister(2, "bit_gt")
        out = QuantumRegister(3, "out")  # out[0]=lt/X, out[1]=eq/Y, out[2]=gt/Z
        c_out = ClassicalRegister(3, "c_out")

        qc = QuantumCircuit(a, b, xnor, bit_lt, bit_gt, out, c_out, name=self.meta.key)

        # Paper bit order is A1 A0 and B1 B0. qcmp stores integers little-endian.
        a0, a1 = a[0], a[1]
        b0, b1 = b[0], b[1]

        # Per-bit relations used by the simplified functions in the paper.
        _open_ccx(qc, [a1, b1], bit_lt[1], {0})  # not(A1) and B1
        _open_ccx(qc, [a0, b0], bit_lt[0], {0})  # not(A0) and B0
        _open_ccx(qc, [a1, b1], bit_gt[1], {1})  # A1 and not(B1)
        _open_ccx(qc, [a0, b0], bit_gt[0], {1})  # A0 and not(B0)

        for source, target in ((a1, xnor[1]), (b1, xnor[1]), (a0, xnor[0]), (b0, xnor[0])):
            qc.cx(source, target)
        qc.x(xnor[1])
        qc.x(xnor[0])

        # X = A<B = not(A1)B1 OR ((A1 xnor B1) not(A0)B0).
        qc.cx(bit_lt[1], out[0])
        qc.ccx(xnor[1], bit_lt[0], out[0])

        # Y = A=B = (A1 xnor B1)(A0 xnor B0).
        qc.ccx(xnor[1], xnor[0], out[1])

        # Z = A>B = A1 not(B1) OR ((A1 xnor B1) A0 not(B0)).
        qc.cx(bit_gt[1], out[2])
        qc.ccx(xnor[1], bit_gt[0], out[2])

        qc.measure(out[0], c_out[0])
        qc.measure(out[1], c_out[1])
        qc.measure(out[2], c_out[2])

        def decode_counts(counts):
            values = {}
            for label, cbit in (("lt", c_out[0]), ("eq", c_out[1]), ("gt", c_out[2])):
                idx = qc.clbits.index(cbit)
                mc = marginal_counts(counts, indices=[idx])
                bit = max(mc, key=mc.get)
                values[label] = int(bit)
            return values

        regs = {
            "a": a,
            "b": b,
            "xnor": xnor,
            "bit_lt": bit_lt,
            "bit_gt": bit_gt,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }
        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0], out[1], out[2]),
        )
        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts, resources=resources)
