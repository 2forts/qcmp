from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import ComparatorMeta, BuildOutput


def cgc(qc: QuantumCircuit, c, a, b) -> None:
    """
    CGC gate from Xia et al. (2018), Fig. 9(a).

    Inputs:
        c = c_i
        a = a_i
        b = b_i

    Outputs in-place:
        c -> p_i = b_i XOR c_i
        a -> q_i = b_i XOR a_i
        b -> c_{i+1} = b_i XOR (p_i AND NOT q_i)

    The middle control in the final Toffoli is an OPEN control.
    """
    qc.cx(b, a)   # a <- a XOR b   (= q_i)
    qc.cx(b, c)   # c <- c XOR b   (= p_i)

    # Open control on a:
    qc.x(a)
    qc.ccx(c, a, b)
    qc.x(a)


def icgc(qc: QuantumCircuit, p, q, c_next) -> None:
    """
    ICGC gate from Xia et al. (2018), Fig. 9(b).

    Inputs:
        p      = p_i
        q      = q_i
        c_next = c_{i+1}

    Restores in-place:
        p      -> C_i = c_i
        q      -> A_i = a_i
        c_next -> B_i = b_i

    The middle control in the first Toffoli is an OPEN control.
    """
    # Open control on q:
    qc.x(q)
    qc.ccx(p, q, c_next)
    qc.x(q)

    qc.cx(c_next, p)
    qc.cx(c_next, q)


class LT_XIA_CGC:
    meta = ComparatorMeta(
        key="lt_xia_cgc",
        name="Ripple-carry comparator using CGC/ICGC (Xia et al., 2018)",
        paper=(
            'H. Xia et al., "An Efficient Design of Reversible Multi-Bit '
            'Quantum Comparator Via Only a Single Ancillary Bit", '
            'International Journal of Theoretical Physics 57, 3727–3744 (2018).'
        ),
        kind="lt",
        signed=False,
        outputs=("lt",),
        ancilla_mode="clean",
        notes=(
            "Implements the n-bit comparator of Section 3.2 / Fig. 11. "
            "The circuit computes the subtraction carry chain with CGC gates, "
            "copies c_n to the output qubit, and then uncomputes with ICGC gates. "
            "According to the paper, x1 = c_n and x1=1 iff a<b."
        ),
    )

    def build(self, n: int, sense: str = "lt", **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")
        if sense != "lt":
            raise ValueError("This comparator currently supports only sense='lt'")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        c0 = QuantumRegister(1, "c0")     # single ancilla, initialized to |0>
        x1 = QuantumRegister(1, "x1")     # output qubit, initialized to |0>
        c_out = ClassicalRegister(1, "c_out")

        qc = QuantumCircuit(a, b, c0, x1, c_out, name=self.meta.key)

        # ------------------------------------------------------------
        # Forward pass: compute c_1, c_2, ..., c_n
        #
        # First stage uses (c0, a0, b0)
        # For i>=1, the previous carry c_i is stored in b[i-1],
        # exactly as in Fig. 11 of the paper.
        # ------------------------------------------------------------
        cgc(qc, c0[0], a[0], b[0])

        for i in range(1, n):
            cgc(qc, b[i - 1], a[i], b[i])

        # x1 = c_n
        qc.cx(b[n - 1], x1[0])

        # ------------------------------------------------------------
        # Backward pass: restore all inputs
        # ------------------------------------------------------------
        for i in range(n - 1, 0, -1):
            icgc(qc, b[i - 1], a[i], b[i])

        icgc(qc, c0[0], a[0], b[0])

        qc.measure(x1[0], c_out[0])

        def decode_counts(counts):
            idx = qc.clbits.index(c_out[0])
            mc = marginal_counts(counts, indices=[idx])
            bit = max(mc, key=mc.get)  # "0" or "1"
            return {"lt": int(bit)}

        regs = {
            "a": a,
            "b": b,
            "c0": c0,
            "x1": x1,
            "out_qubit": x1[0],
            "c_out": c_out,
        }

        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts)