from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _one_bit_relation(
    qc: QuantumCircuit,
    a_bit,
    b_bit,
    equal,
    greater,
    lesser,
) -> None:
    """Write the three relations for a single pair of bits into clean targets."""
    # equal = NOT(a XOR b)
    qc.cx(a_bit, equal)
    qc.cx(b_bit, equal)
    qc.x(equal)

    # greater = a AND NOT b; lesser = NOT a AND b.
    qc.x(b_bit)
    qc.ccx(a_bit, b_bit, greater)
    qc.x(b_bit)
    qc.x(a_bit)
    qc.ccx(a_bit, b_bit, lesser)
    qc.x(a_bit)


class CMP_Rangaraju_Serial:
    meta = ComparatorMeta(
        key="cmp_rangaraju_serial",
        name="Serial reversible binary comparator (Rangaraju et al., 2012)",
        paper=(
            'R. H. G. Rangaraju, V. Hegde, R. K. B., and M. K. N., '
            '"Design of Efficient Reversible Binary Comparator", Procedia '
            'Engineering 30, 897-904 (2012).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the serial input-circuit/one-bit-cell arrangement of "
            "Figures 4-11. Every stage retains its relation signals, matching "
            "the paper's garbage-output-oriented reversible construction."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        eq = QuantumRegister(n, "eq")
        gt = QuantumRegister(n, "gt")
        lt = QuantumRegister(n, "lt")
        c_out = ClassicalRegister(3, "c_out")
        qc = QuantumCircuit(a, b, eq, gt, lt, c_out, name=self.meta.key)

        # The first stage is the paper's input circuit applied to the MSB.
        msb = n - 1
        _one_bit_relation(qc, a[msb], b[msb], eq[0], gt[0], lt[0])

        # Each following one-bit cell only changes the relation when all more
        # significant bits were equal. This is the serial connection in Figs. 7-11.
        for stage, bit in enumerate(range(n - 2, -1, -1), start=1):
            qc.cx(gt[stage - 1], gt[stage])
            qc.cx(lt[stage - 1], lt[stage])

            qc.x(b[bit])
            qc.mcx([eq[stage - 1], a[bit], b[bit]], gt[stage])
            qc.x(b[bit])

            qc.x(a[bit])
            qc.mcx([eq[stage - 1], a[bit], b[bit]], lt[stage])
            qc.x(a[bit])

            # eq[stage] = eq[stage - 1] AND (a[bit] == b[bit]).
            qc.x(a[bit])
            qc.x(b[bit])
            qc.mcx([eq[stage - 1], a[bit], b[bit]], eq[stage])
            qc.x(b[bit])
            qc.x(a[bit])
            qc.mcx([eq[stage - 1], a[bit], b[bit]], eq[stage])

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
                "eq": eq,
                "gt": gt,
                "lt": lt,
                "out_qubit": lt[last],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
