from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


class GT_Du_Nonzero:
    meta = ComparatorMeta(
        key="gt_du_nonzero",
        name="Nonzero pixel-state checker (Du et al., 2022)",
        paper=(
            'S. Du et al., "Binarization of grayscale quantum image denoted '
            'with novel enhanced quantum representations", Results in Physics '
            '39, 105710 (2022).'
        ),
        kind="gt",
        signed=False,
        outputs=("gt",),
        ancilla_mode="clean",
        notes=(
            "Implements U1/Qf from Figures 2-4: gt=1 iff the pixel register A "
            "contains at least one 1 bit, i.e. A>0. The paper uses this fixed "
            "zero-threshold predicate for binarization; B is not part of it."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        pixel = QuantumRegister(n, "a")
        unused_reference = QuantumRegister(n, "b")
        out = QuantumRegister(1, "out")
        c_out = ClassicalRegister(1, "c_out")
        qc = QuantumCircuit(pixel, unused_reference, out, c_out, name=self.meta.key)

        # Q_f is a Boolean OR over the color qubits. An all-zero controlled
        # operation first detects A=0, then the final X gives A!=0. The input
        # register is restored before the result is used by the next stage.
        for qubit in pixel:
            qc.x(qubit)
        qc.mcx(list(pixel), out[0])
        for qubit in reversed(pixel):
            qc.x(qubit)
        qc.x(out[0])
        qc.measure(out[0], c_out[0])

        def decode_counts(counts):
            index = qc.clbits.index(c_out[0])
            marginal = marginal_counts(counts, indices=[index])
            return {"gt": int(max(marginal, key=marginal.get))}

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0],),
        )
        return BuildOutput(
            circuit=qc,
            regs={
                "a": pixel,
                "b": unused_reference,
                "out": out,
                "out_qubit": out[0],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
