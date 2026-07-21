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


class CMP_Al_Rabadi:
    meta = ComparatorMeta(
        key="cmp_al_rabadi",
        name="Iterative equality-based comparator (Al-Rabadi, 2009)",
        paper=(
            'A. N. Al-Rabadi, "Closed-System Quantum Logic Network '
            'Implementation of the Viterbi Algorithm", Facta Universitatis, '
            "Series: Electronics and Energetics 22(1), 1-33 (2009)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="clean",
        notes=(
            "Implements the iterative comparator network from Figures 12-14. "
            "The circuit compares A and B from most significant bit to least "
            "significant bit, propagating two state chains: A<B and A>B."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        lt_state = QuantumRegister(n + 1, "lt_state")
        gt_state = QuantumRegister(n + 1, "gt_state")
        out = QuantumRegister(3, "out")  # out[0]=lt, out[1]=eq, out[2]=gt
        c_out = ClassicalRegister(3, "c_out")

        qc = QuantumCircuit(a, b, lt_state, gt_state, out, c_out, name=self.meta.key)

        # The paper labels digits from left to right: x1...xn and y1...yn.
        # qcmp loads integer registers little-endian, so stage 0 uses a[n-1]/b[n-1].
        stage_a = [a[n - 1 - i] for i in range(n)]
        stage_b = [b[n - 1 - i] for i in range(n)]

        for i, (a_i, b_i) in enumerate(zip(stage_a, stage_b)):
            # Start each next-state qubit as a copy of the previous state.
            qc.cx(lt_state[i], lt_state[i + 1])
            qc.cx(gt_state[i], gt_state[i + 1])

            # lt_next = lt_i OR ((NOT gt_i) AND (NOT a_i) AND b_i)
            # Reversible form:
            # lt_next ^= (NOT lt_i) AND (NOT gt_i) AND (NOT a_i) AND b_i
            _open_mcx(
                qc,
                [lt_state[i], gt_state[i], a_i, b_i],
                lt_state[i + 1],
                {0, 1, 2},
            )

            # gt_next = gt_i OR ((NOT lt_i) AND a_i AND (NOT b_i))
            # Reversible form:
            # gt_next ^= (NOT gt_i) AND (NOT lt_i) AND a_i AND (NOT b_i)
            _open_mcx(
                qc,
                [gt_state[i], lt_state[i], a_i, b_i],
                gt_state[i + 1],
                {0, 1, 3},
            )

        # Output circuit equivalent to Figure 13(b)/14(b).
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
            "lt_state": lt_state,
            "gt_state": gt_state,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }

        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts)
