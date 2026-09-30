from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _npg(qc: QuantumCircuit, a_bit, b_line, greater, lesser) -> None:
    """Apply Sarker et al.'s 4x4 Newly Proposed Gate (NPG).

    The inputs are (A, B, C, D) and the outputs are
    (A, A XOR NOT(B), A NOT(B) XOR C,
    (A XOR NOT(B)) XOR (A NOT(B)) XOR C XOR D).
    For C=0 and D=1, the last three values encode equality, greater-than,
    and less-than for one bit.
    """
    qc.x(b_line)
    qc.ccx(a_bit, b_line, greater)
    qc.cx(a_bit, b_line)
    qc.cx(b_line, lesser)
    qc.cx(greater, lesser)


class CMP_Sarker_NPG:
    meta = ComparatorMeta(
        key="cmp_sarker_npg",
        name="NPG comparator-cell and propagator-cell comparator (Sarker et al., 2014)",
        paper=(
            'A. Sarker, M. S. Amin, A. Bose, and N. Islam, "An Optimized '
            'Design of Binary Comparator Circuit in Quantum Computing", 3rd '
            'International Conference on Informatics, Electronics & Vision (2014).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the NPG comparator cell of Figure 7 followed by the "
            "MSB-to-LSB propagator recurrence of Figures 8-10. The paper's "
            "reported formulas are garbage=4N-3, constants=2N, delay=4N, "
            "and gate count=1+9(N-1)."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        npg_b = QuantumRegister(n, "npg_b")
        bit_gt = QuantumRegister(n, "bit_gt")
        bit_lt = QuantumRegister(n, "bit_lt")
        eq = QuantumRegister(n, "eq")
        gt = QuantumRegister(n, "gt")
        lt = QuantumRegister(n, "lt")
        c_out = ClassicalRegister(3, "c_out")
        qc = QuantumCircuit(a, b, npg_b, bit_gt, bit_lt, eq, gt, lt, c_out, name=self.meta.key)

        # The NPG cell creates Q, R, and S for each bit. It operates on a
        # copy of B because its Q output overwrites the B line.
        for stage, bit in enumerate(range(n - 1, -1, -1)):
            qc.cx(b[bit], npg_b[stage])
            qc.x(bit_lt[stage])
            _npg(qc, a[bit], npg_b[stage], bit_gt[stage], bit_lt[stage])

            if stage == 0:
                qc.cx(npg_b[stage], eq[stage])
                qc.cx(bit_gt[stage], gt[stage])
                qc.cx(bit_lt[stage], lt[stage])
                continue

            # Equations in the propagator cell: R_i = (A_i>B_i)Q_{i+1} XOR
            # R_{i+1}, S_i = (A_i<B_i)Q_{i+1} XOR S_{i+1}, and
            # Q_i = NOT(R_i XOR S_i).
            qc.cx(gt[stage - 1], gt[stage])
            qc.ccx(bit_gt[stage], eq[stage - 1], gt[stage])

            qc.cx(lt[stage - 1], lt[stage])
            qc.ccx(bit_lt[stage], eq[stage - 1], lt[stage])

            qc.x(eq[stage])
            qc.cx(gt[stage], eq[stage])
            qc.cx(lt[stage], eq[stage])

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
                "npg_b": npg_b,
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
