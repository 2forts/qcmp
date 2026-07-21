from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _open_ccx(qc: QuantumCircuit, c_closed, c_open, target) -> None:
    """Toffoli with one closed control and one open control."""
    qc.x(c_open)
    qc.ccx(c_closed, c_open, target)
    qc.x(c_open)


def _forward_carry_stage(qc: QuantumCircuit, c_in, q_i, b_i) -> None:
    """
    Assumes q_i already stores a_i XOR b_i.
    Input:
        c_in = current borrow/carry information
        q_i  = a_i XOR b_i
        b_i  = original b_i

    Output:
        c_in -> b_i XOR c_in      temporary p_i
        b_i  -> next borrow bit
    """
    qc.cx(b_i, c_in)
    _open_ccx(qc, c_in, q_i, b_i)


def _inverse_carry_stage(qc: QuantumCircuit, c_in, q_i, b_i) -> None:
    """Inverse of _forward_carry_stage."""
    _open_ccx(qc, c_in, q_i, b_i)
    qc.cx(b_i, c_in)


def _final_output_stage(qc: QuantumCircuit, c_in, q_i, b_i, out) -> None:
    """
    Compute the final comparison bit into out without leaving c_in dirty.

    out becomes 1 iff A < B.
    """
    qc.cx(b_i, c_in)          # c_in = b_i XOR c_in
    qc.cx(b_i, out)           # out starts as b_i
    _open_ccx(qc, c_in, q_i, out)
    qc.cx(b_i, c_in)          # restore c_in


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
            "Paper-faithful half multi-bit comparator following Xia et al. "
            "(QIP 2019): one clean auxiliary qubit plus one output qubit. "
            "The b-register is used temporarily during the ripple comparison "
            "and is restored by uncomputation. Output is 1 iff A < B."
        ),
    )

    def build(self, n: int, sense: str = "lt", **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")
        if sense != "lt":
            raise ValueError("This comparator currently supports only sense='lt'")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        aux = QuantumRegister(1, "aux")   # paper's clean auxiliary |0>
        out = QuantumRegister(1, "out")   # comparison result
        c_out = ClassicalRegister(1, "c_out")

        qc = QuantumCircuit(a, b, aux, out, c_out, name=self.meta.key)

        # ------------------------------------------------------------
        # Step 1: q_i = a_i XOR b_i, stored in a_i.
        #
        # This matches the paper diagrams where each a_i line is first
        # XORed with b_i and later restored.
        # ------------------------------------------------------------
        for i in range(n):
            qc.cx(b[i], a[i])

        # ------------------------------------------------------------
        # n = 1: paper's half 1-bit comparator.
        #
        # Since a[0] now stores a0 XOR b0:
        #     (a0 XOR b0) AND b0 = NOT(original a0) AND b0
        # so out = 1 iff a0 < b0.
        # ------------------------------------------------------------
        if n == 1:
            qc.ccx(a[0], b[0], out[0])

        else:
            # --------------------------------------------------------
            # Step 2: first borrow into aux.
            #
            # aux = (a0 XOR b0) AND b0
            #     = NOT(original a0) AND b0
            # --------------------------------------------------------
            qc.ccx(a[0], b[0], aux[0])

            # --------------------------------------------------------
            # Step 3: propagate intermediate borrows through b[1..n-2].
            #
            # After stage i, b[i] temporarily stores the borrow after
            # comparing the lower i+1 bits.
            # --------------------------------------------------------
            for i in range(1, n - 1):
                c_in = aux[0] if i == 1 else b[i - 1]
                _forward_carry_stage(qc, c_in, a[i], b[i])

            # --------------------------------------------------------
            # Step 4: compute final borrow into out.
            #
            # For n=2, the incoming borrow is aux.
            # For n>2, it is stored temporarily in b[n-2].
            # --------------------------------------------------------
            c_final_in = aux[0] if n == 2 else b[n - 2]
            _final_output_stage(qc, c_final_in, a[n - 1], b[n - 1], out[0])

            # --------------------------------------------------------
            # Step 5: uncompute intermediate borrow chain.
            # --------------------------------------------------------
            for i in range(n - 2, 0, -1):
                c_in = aux[0] if i == 1 else b[i - 1]
                _inverse_carry_stage(qc, c_in, a[i], b[i])

            # --------------------------------------------------------
            # Step 6: clean aux.
            # --------------------------------------------------------
            qc.ccx(a[0], b[0], aux[0])

        # ------------------------------------------------------------
        # Step 7: restore a_i from q_i = a_i XOR b_i.
        # b is also restored by the inverse chain above.
        # ------------------------------------------------------------
        for i in range(n - 1, -1, -1):
            qc.cx(b[i], a[i])

        qc.measure(out[0], c_out[0])

        def decode_counts(counts):
            idx = qc.clbits.index(c_out[0])
            mc = marginal_counts(counts, indices=[idx])
            bit = max(mc, key=mc.get)
            return {"lt": int(bit)}

        regs = {
            "a": a,
            "b": b,
            "aux": aux,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0],),
            restored_zero_qubits=(aux[0],),
        )

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=resources,
        )