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


class CMP_Lopez_Proposal2:
    meta = ComparatorMeta(
        key="cmp_lopez_prop2",
        name="Fault-tolerant full comparator, proposal 2 (Lopez et al., 2023)",
        paper=(
            'L. O. Lopez, F. Orts, G. Ortega, V. Gonzalez-Ruiz, and '
            'E. M. Garzon, "Fault-tolerant quantum algorithm for '
            'dual-threshold image segmentation", The Journal of Supercomputing '
            "79, 12549-12562 (2023)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="dirty",
        notes=(
            "Implements the scalable full-comparator logic of proposed "
            "comparator circuit 2. The paper presents the N=3 instance and "
            "general metrics T-count=16N, T-depth=10N-6, ancillas=3N+3."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        bit_lt = QuantumRegister(n, "bit_lt")
        bit_gt = QuantumRegister(n, "bit_gt")
        eq_prefix = QuantumRegister(n + 1, "eq_prefix")
        out = QuantumRegister(3, "out")  # out[0]=lt, out[1]=eq, out[2]=gt
        c_out = ClassicalRegister(3, "c_out")

        qc = QuantumCircuit(a, b, bit_lt, bit_gt, eq_prefix, out, c_out, name=self.meta.key)

        # eq_prefix[i] means all more-significant processed bits are equal.
        qc.x(eq_prefix[0])

        stage_bits = list(range(n - 1, -1, -1))
        for stage, bit in enumerate(stage_bits):
            _open_mcx(qc, [a[bit], b[bit]], bit_lt[stage], {0})  # not(a_i) and b_i
            _open_mcx(qc, [a[bit], b[bit]], bit_gt[stage], {1})  # a_i and not(b_i)

            qc.ccx(eq_prefix[stage], bit_lt[stage], out[0])
            qc.ccx(eq_prefix[stage], bit_gt[stage], out[2])

            # Continue only while previous bits are equal and this bit is equal.
            _open_mcx(
                qc,
                [eq_prefix[stage], bit_lt[stage], bit_gt[stage]],
                eq_prefix[stage + 1],
                {1, 2},
            )

        qc.cx(eq_prefix[n], out[1])

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
            "bit_lt": bit_lt,
            "bit_gt": bit_gt,
            "eq_prefix": eq_prefix,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }
        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(out[0], out[1], out[2]),
        )
        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts, resources=resources)
