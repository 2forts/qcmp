from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _open_mcx(qc: QuantumCircuit, controls, target, open_indexes: set[int]) -> None:
    for index in open_indexes:
        qc.x(controls[index])
    qc.mcx(controls, target)
    for index in reversed(list(open_indexes)):
        qc.x(controls[index])


class GT_Barui_Threshold:
    meta = ComparatorMeta(
        key="gt_barui_threshold",
        name="Threshold comparator for quantum image binarization (Barui et al., 2024)",
        paper=(
            'A. Barui, M. Pal, and P. K. Panigrahi, "A novel approach to '
            'threshold quantum images by using unsharp measurements", '
            "Quantum Information Processing 23, 76 (2024)."
        ),
        kind="gt",
        signed=False,
        outputs=("gt",),
        ancilla_mode="dirty",
        notes=(
            "Implements the comparator role from Fig. 8: output X changes iff "
            "the encoded intensity I is greater than the threshold T. In qcmp, "
            "register a is I and register b is T."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        intensity = QuantumRegister(n, "a")
        threshold = QuantumRegister(n, "b")
        bit_gt = QuantumRegister(n, "bit_gt")
        bit_lt = QuantumRegister(n, "bit_lt")
        eq_prefix = QuantumRegister(n + 1, "eq_prefix")
        out = QuantumRegister(1, "out")
        c_out = ClassicalRegister(1, "c_out")

        qc = QuantumCircuit(
            intensity,
            threshold,
            bit_gt,
            bit_lt,
            eq_prefix,
            out,
            c_out,
            name=self.meta.key,
        )

        qc.x(eq_prefix[0])
        for stage, bit in enumerate(range(n - 1, -1, -1)):
            _open_mcx(qc, [intensity[bit], threshold[bit]], bit_gt[stage], {1})
            _open_mcx(qc, [intensity[bit], threshold[bit]], bit_lt[stage], {0})

            qc.ccx(eq_prefix[stage], bit_gt[stage], out[0])
            _open_mcx(
                qc,
                [eq_prefix[stage], bit_gt[stage], bit_lt[stage]],
                eq_prefix[stage + 1],
                {1, 2},
            )

        qc.measure(out[0], c_out[0])

        def decode_counts(counts):
            idx = qc.clbits.index(c_out[0])
            mc = marginal_counts(counts, indices=[idx])
            bit = max(mc, key=mc.get)
            return {"gt": int(bit)}

        regs = {
            "a": intensity,
            "b": threshold,
            "bit_gt": bit_gt,
            "bit_lt": bit_lt,
            "eq_prefix": eq_prefix,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }
        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0],),
        )
        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts, resources=resources)
