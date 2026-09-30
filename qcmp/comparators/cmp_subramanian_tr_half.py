from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta


def _open_mcx(qc: QuantumCircuit, controls, target, open_indexes: set[int]) -> None:
    """Apply a multi-controlled X with selected open controls."""
    for index in open_indexes:
        qc.x(controls[index])
    qc.mcx(controls, target)
    for index in reversed(list(open_indexes)):
        qc.x(controls[index])


def _tr_half_subtractor(qc: QuantumCircuit, a_bit, b_bit, diff_bit, borrow_bit) -> None:
    """
    TR-gate half-subtractor behavior used by Subramanian et al.

    Each bit stage produces:
        diff   = A xor B
        borrow = not(A) and B
    """
    qc.cx(a_bit, diff_bit)
    qc.cx(b_bit, diff_bit)

    qc.x(a_bit)
    qc.ccx(a_bit, b_bit, borrow_bit)
    qc.x(a_bit)


class CMP_Subramanian_TR_Half:
    meta = ComparatorMeta(
        key="cmp_subramanian_tr_half",
        name="TR-gate half-subtraction comparator (Subramanian et al., 2016)",
        paper=(
            'S. Saravanan, I. Vennila, and S. Mohanram, "Design and '
            'Implementation of an Efficient Reversible Comparator Using TR Gate", '
            "Circuits and Systems 7, 2578-2592 (2016)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="clean",
        notes=(
            "Implements the half-subtraction method from the TR-gate comparator. "
            "Each bit computes D=A xor B and Borrow=not(A) and B, then the "
            "decision state is propagated from MSB to LSB."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        diff = QuantumRegister(n, "diff")
        borrow = QuantumRegister(n, "borrow")
        lt_state = QuantumRegister(n + 1, "lt_state")
        gt_state = QuantumRegister(n + 1, "gt_state")
        out = QuantumRegister(3, "out")  # out[0]=lt, out[1]=eq, out[2]=gt
        c_out = ClassicalRegister(3, "c_out")

        qc = QuantumCircuit(
            a,
            b,
            diff,
            borrow,
            lt_state,
            gt_state,
            out,
            c_out,
            name=self.meta.key,
        )

        # The paper scans from MSB to LSB. qcmp loads integers little-endian,
        # so stage 0 must use a[n-1]/b[n-1].
        stage_a = [a[n - 1 - i] for i in range(n)]
        stage_b = [b[n - 1 - i] for i in range(n)]

        for i, (a_i, b_i) in enumerate(zip(stage_a, stage_b)):
            _tr_half_subtractor(qc, a_i, b_i, diff[i], borrow[i])

        for i in range(n):
            qc.cx(lt_state[i], lt_state[i + 1])
            qc.cx(gt_state[i], gt_state[i + 1])

            # lt_next = lt_i OR ((NOT lt_i) AND (NOT gt_i) AND borrow_i)
            _open_mcx(qc, [lt_state[i], gt_state[i], borrow[i]], lt_state[i + 1], {0, 1})

            # gt_next = gt_i OR ((NOT lt_i) AND (NOT gt_i) AND diff_i AND NOT borrow_i)
            _open_mcx(
                qc,
                [gt_state[i], lt_state[i], diff[i], borrow[i]],
                gt_state[i + 1],
                {0, 1, 3},
            )

        qc.cx(lt_state[n], out[0])
        qc.x(out[1])
        qc.cx(lt_state[n], out[1])
        qc.cx(gt_state[n], out[1])
        qc.cx(gt_state[n], out[2])

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
            "diff": diff,
            "borrow": borrow,
            "lt_state": lt_state,
            "gt_state": gt_state,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }

        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts)
