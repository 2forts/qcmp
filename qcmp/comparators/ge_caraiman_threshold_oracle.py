from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout
from qcmp.comparators.cmp_oliveira_qbsc import open_ccx, uc_instruction


class GE_Caraiman_Threshold_Oracle:
    """Threshold oracle from Caraiman and Manta, built on the QBSC block."""

    meta = ComparatorMeta(
        key="ge_caraiman_threshold_oracle",
        name="Thresholding oracle using a QBSC (Caraiman and Manta, 2015)",
        paper=(
            'S. Caraiman and V. I. Manta, "Image segmentation on a quantum '
            'computer", Quantum Information Processing 14, 1693-1715 (2015).'
        ),
        kind="ge",
        signed=False,
        outputs=("ge",),
        ancilla_mode="dirty",
        notes=(
            "Implements the threshold oracle of Figure 5(b), using the QBSC "
            "construction cited by the paper. In qcmp notation A is the "
            "threshold T and B is the gray-level c; ge=1 iff B >= A."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 2:
            raise ValueError("ge_caraiman_threshold_oracle requires n >= 2")

        threshold = QuantumRegister(n, "a")
        color = QuantumRegister(n, "b")
        gt = QuantumRegister(n, "gt")
        lt = QuantumRegister(n, "lt")
        equal_prefix = QuantumRegister(n - 1, "eq_prefix")
        out = QuantumRegister(1, "out")
        c_out = ClassicalRegister(1, "c_out")
        qc = QuantumCircuit(
            threshold, color, gt, lt, equal_prefix, out, c_out,
            name=self.meta.key,
        )

        # This is the U_CMP block used by the oracle in Figure 5(b). The
        # high-order comparison bit is propagated down to gt[0]/lt[0].
        uc = uc_instruction()
        stage_a = [threshold[n - 1 - i] for i in range(n)]
        stage_b = [color[n - 1 - i] for i in range(n)]
        for i in range(n):
            qc.append(uc, [stage_a[i], stage_b[i], gt[i], lt[i]])
        for i in range(n - 1):
            open_ccx(qc, gt[i], lt[i], equal_prefix[i])
        for i in range(n - 1, 0, -1):
            qc.ccx(gt[i], equal_prefix[i - 1], gt[i - 1])
            qc.ccx(lt[i], equal_prefix[i - 1], lt[i - 1])

        # The paper flips its oracle qubit for c >= T. gt[0] means T > c,
        # hence out = NOT gt[0].
        qc.x(out[0])
        qc.cx(gt[0], out[0])
        qc.measure(out[0], c_out[0])

        def decode_counts(counts):
            index = qc.clbits.index(c_out[0])
            bit = max(marginal_counts(counts, indices=[index]), key=lambda k: marginal_counts(counts, indices=[index])[k])
            return {"ge": int(bit)}

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0],),
        )
        return BuildOutput(
            circuit=qc,
            regs={
                "a": threshold,
                "b": color,
                "gt": gt,
                "lt": lt,
                "eq_prefix": equal_prefix,
                "out": out,
                "out_qubit": out[0],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
