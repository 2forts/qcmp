from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta


def _mcx(qc: QuantumCircuit, controls, target) -> None:
    qc.mcx(list(controls), target)


def _toggle_borrow_out(qc: QuantumCircuit, a, b, borrow_in, borrow_out) -> None:
    """Toggle borrow_out with the full-subtractor borrow for one bit."""
    # Term 1: not(a_i) and b_i.
    qc.x(a)
    qc.ccx(a, b, borrow_out)
    qc.x(a)

    # Term 2: borrow_in and a_i == b_i. The two equality cases are disjoint
    # from term 1, so XOR toggles implement the boolean OR reversibly.
    qc.x(a)
    qc.x(b)
    _mcx(qc, [borrow_in, a, b], borrow_out)
    qc.x(a)
    qc.x(b)
    _mcx(qc, [borrow_in, a, b], borrow_out)


class LT_Xia_QIP_Half:
    meta = ComparatorMeta(
        key="lt_xia_qip",
        name="Half multi-bit comparator (Xia et al., 2019)",
        paper=(
            'H. Xia, H. Li, H. Zhang, Y. Liang, and J. Xin, '
            '"Novel multi-bit quantum comparators and their application in '
            'image binarization", Quantum Information Processing 18, 229 (2019).'
        ),
        kind="lt",
        signed=False,
        outputs=("lt",),
        ancilla_mode="clean",
        notes=(
            "Scalable implementation of the half multi-bit comparator borrow "
            "recurrence: the output is 1 iff A<B."
        ),
    )

    def build(self, n: int, sense: str = "lt", **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")
        if sense != "lt":
            raise ValueError("This comparator currently supports only sense='lt'")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        borrow = QuantumRegister(n + 1, "borrow")
        out = QuantumRegister(1, "out")
        c_out = ClassicalRegister(1, "c_out")

        qc = QuantumCircuit(a, b, borrow, out, c_out, name=self.meta.key)

        for i in range(n):
            if i == 0:
                qc.x(a[i])
                qc.ccx(a[i], b[i], borrow[i + 1])
                qc.x(a[i])
            else:
                _toggle_borrow_out(qc, a[i], b[i], borrow[i], borrow[i + 1])

        qc.cx(borrow[n], out[0])

        for i in range(n - 1, -1, -1):
            if i == 0:
                qc.x(a[i])
                qc.ccx(a[i], b[i], borrow[i + 1])
                qc.x(a[i])
            else:
                _toggle_borrow_out(qc, a[i], b[i], borrow[i], borrow[i + 1])

        qc.measure(out[0], c_out[0])

        def decode_counts(counts):
            idx = qc.clbits.index(c_out[0])
            mc = marginal_counts(counts, indices=[idx])
            bit = max(mc, key=mc.get)
            return {"lt": int(bit)}

        regs = {
            "a": a,
            "b": b,
            "borrow": borrow,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }

        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts)
