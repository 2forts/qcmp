from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _zrqc1(qc: QuantumCircuit, a_bit, b_bit, equal, greater) -> None:
    """ZRQC1 relation block from Zhou et al., Figure 13.

    With its zero-valued target lines, the block produces the one-bit
    relations equal = NOT(a XOR b) and greater = a AND NOT(b).
    """
    qc.x(equal)
    qc.cx(a_bit, equal)
    qc.cx(b_bit, equal)

    qc.x(b_bit)
    qc.ccx(a_bit, b_bit, greater)
    qc.x(b_bit)


class CMP_Zhou_ZRQC:
    meta = ComparatorMeta(
        key="cmp_zhou_zrqc",
        name="ZRQC1 cascade reversible comparator (Zhou et al., 2013)",
        paper=(
            'R.-G. Zhou, M.-Q. Zhang, Q. Wu, and Y.-C. Li, "Optimization '
            'Approaches for Designing a Novel 4-Bit Reversible Comparator", '
            'International Journal of Theoretical Physics 52, 559-575 (2013).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the scalable cascade of ZRQC1 relation blocks and "
            "prefix products in Figures 13, 14, and 18. The Qiskit CCX/CX "
            "realization preserves the paper's Boolean structure; its gate "
            "counts are not a direct replacement for the paper's V/Vdg cost."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        bit_eq = QuantumRegister(n, "bit_eq")
        bit_gt = QuantumRegister(n, "bit_gt")
        bit_lt = QuantumRegister(n, "bit_lt")
        eq = QuantumRegister(n, "eq")
        gt = QuantumRegister(n, "gt")
        lt = QuantumRegister(n, "lt")
        c_out = ClassicalRegister(3, "c_out")
        qc = QuantumCircuit(a, b, bit_eq, bit_gt, bit_lt, eq, gt, lt, c_out, name=self.meta.key)

        # Each ZRQC1 instance provides the equality and greater-than terms
        # shown on the left side of Fig. 18. Stages run from MSB to LSB.
        for stage, bit in enumerate(range(n - 1, -1, -1)):
            _zrqc1(qc, a[bit], b[bit], bit_eq[stage], bit_gt[stage])
            qc.x(a[bit])
            qc.ccx(a[bit], b[bit], bit_lt[stage])
            qc.x(a[bit])

            if stage == 0:
                qc.cx(bit_eq[stage], eq[stage])
                qc.cx(bit_gt[stage], gt[stage])
                qc.cx(bit_lt[stage], lt[stage])
                continue

            qc.ccx(eq[stage - 1], bit_eq[stage], eq[stage])

            qc.cx(gt[stage - 1], gt[stage])
            qc.ccx(eq[stage - 1], bit_gt[stage], gt[stage])

            qc.cx(lt[stage - 1], lt[stage])
            qc.ccx(eq[stage - 1], bit_lt[stage], lt[stage])

        last = n - 1
        qc.measure(gt[last], c_out[0])
        qc.measure(lt[last], c_out[1])
        qc.measure(eq[last], c_out[2])

        def decode_counts(counts):
            bitstring = max(counts, key=counts.get).replace(" ", "")
            return {
                "gt": int(bitstring[-1]),
                "lt": int(bitstring[-2]),
                "eq": int(bitstring[-3]),
            }

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(gt[last], lt[last], eq[last]),
        )
        return BuildOutput(
            circuit=qc,
            regs={
                "a": a,
                "b": b,
                "bit_eq": bit_eq,
                "bit_gt": bit_gt,
                "bit_lt": bit_lt,
                "eq": eq,
                "gt": gt,
                "lt": lt,
                "out_qubit": lt[last],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
