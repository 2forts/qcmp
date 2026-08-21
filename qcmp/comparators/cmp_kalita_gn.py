from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _gn_gate(qc: QuantumCircuit, a_line, b_line, c_line) -> None:
    """Apply the GN gate from Kalita and Saikia, Figure 2.

    It maps (A, B, C) to (A XOR C, NOT(A) C XOR B, C). The
    implementation below is the Boolean CCX/CX realization of that mapping.
    """
    qc.cx(c_line, a_line)
    qc.ccx(c_line, a_line, b_line)


def _inverse_gn_gate(qc: QuantumCircuit, a_line, b_line, c_line) -> None:
    qc.ccx(c_line, a_line, b_line)
    qc.cx(c_line, a_line)


class CMP_Kalita_GN:
    meta = ComparatorMeta(
        key="cmp_kalita_gn",
        name="GN-gate cascaded reversible comparator (Kalita and Saikia, 2015)",
        paper=(
            'G. Kalita and N. Saikia, "Reversible Comparator Circuit Using a '
            'New Reversible Gate", International Conference on Computing, '
            'Communication and Technologies (2015).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the cascading multi-bit comparison principle from "
            "Figures 5 and 6 using the paper's GN mapping. The paper gives "
            "a general construction for N>=2; this Boolean realization also "
            "handles N=1. Its CCX/CX cost is not the paper's controlled-V cost."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        gn_gt = QuantumRegister(1, "gn_gt")
        gn_lt = QuantumRegister(1, "gn_lt")
        bit_gt = QuantumRegister(1, "bit_gt")
        bit_lt = QuantumRegister(1, "bit_lt")
        eq = QuantumRegister(n, "eq")
        gt = QuantumRegister(n, "gt")
        lt = QuantumRegister(n, "lt")
        c_out = ClassicalRegister(3, "c_out")
        qc = QuantumCircuit(
            a,
            b,
            gn_gt,
            gn_lt,
            bit_gt,
            bit_lt,
            eq,
            gt,
            lt,
            c_out,
            name=self.meta.key,
        )

        # The GN gate yields A AND NOT(B) when its first input is B, its
        # second input is a clean target, and its third input is A. Swapping
        # A and B yields the less-than term. The temporary GN cells are
        # uncomputed after their terms have been propagated.
        for stage, bit in enumerate(range(n - 1, -1, -1)):
            qc.cx(b[bit], gn_gt[0])
            _gn_gate(qc, gn_gt[0], bit_gt[0], a[bit])

            qc.cx(a[bit], gn_lt[0])
            _gn_gate(qc, gn_lt[0], bit_lt[0], b[bit])

            if stage == 0:
                qc.cx(bit_gt[0], gt[stage])
                qc.cx(bit_lt[0], lt[stage])
            else:
                qc.cx(gt[stage - 1], gt[stage])
                qc.ccx(eq[stage - 1], bit_gt[0], gt[stage])

                qc.cx(lt[stage - 1], lt[stage])
                qc.ccx(eq[stage - 1], bit_lt[0], lt[stage])

            # Exactly one of gt, lt, and eq is true for a binary comparison.
            qc.x(eq[stage])
            qc.cx(gt[stage], eq[stage])
            qc.cx(lt[stage], eq[stage])

            _inverse_gn_gate(qc, gn_lt[0], bit_lt[0], b[bit])
            qc.cx(a[bit], gn_lt[0])
            _inverse_gn_gate(qc, gn_gt[0], bit_gt[0], a[bit])
            qc.cx(b[bit], gn_gt[0])

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
                "gn_gt": gn_gt,
                "gn_lt": gn_lt,
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
