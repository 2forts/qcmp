from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _two_bit_relation(qc: QuantumCircuit, a_hi, a_lo, b_hi, b_lo, high_eq, equal, greater, lesser) -> None:
    """Write the GT/LT/EQ relation for a two-bit block into clean targets."""
    qc.x(high_eq)
    qc.cx(a_hi, high_eq)
    qc.cx(b_hi, high_eq)

    qc.x(b_hi)
    qc.ccx(a_hi, b_hi, greater)
    qc.x(b_hi)

    qc.x(a_hi)
    qc.ccx(a_hi, b_hi, lesser)
    qc.x(a_hi)

    qc.x(b_lo)
    qc.mcx([high_eq, a_lo, b_lo], greater)
    qc.x(b_lo)

    qc.x(a_lo)
    qc.mcx([high_eq, a_lo, b_lo], lesser)
    qc.x(a_lo)

    # equal = high_eq AND NOT(a_lo XOR b_lo).
    qc.x(a_lo)
    qc.x(b_lo)
    qc.mcx([high_eq, a_lo, b_lo], equal)
    qc.x(b_lo)
    qc.x(a_lo)
    qc.mcx([high_eq, a_lo, b_lo], equal)


class CMP_Nagamani_Multistage:
    meta = ComparatorMeta(
        key="cmp_nagamani_multistage",
        name="Multi-stage magnitude comparator (Nagamani et al., 2016)",
        paper=(
            'A. N. Nagamani, D. Rengarajan, and V. K. Agrawal, "An Optimized '
            'Design of Reversible Magnitude and Signed Comparators", 2016 IEEE '
            'International Conference on Electron Devices and Solid-State Circuits.'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the unsigned multi-stage structure: two-bit Stage 1 "
            "blocks followed by the priority-encoding Stage 2. The paper "
            "generalizes for N=2^k and reports quantum cost 16N-8 and delay "
            "10log2(N)+7. This Qiskit realization preserves the structure but "
            "does not duplicate the paper's exact Feynman/Peres/Fredkin synthesis."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 4 or n & (n - 1):
            raise ValueError("cmp_nagamani_multistage requires n to be a power of two and n >= 4")

        groups = n // 2
        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        high_eq = QuantumRegister(groups, "high_eq")
        pair_eq = QuantumRegister(groups, "pair_eq")
        pair_gt = QuantumRegister(groups, "pair_gt")
        pair_lt = QuantumRegister(groups, "pair_lt")
        eq = QuantumRegister(groups, "eq")
        gt = QuantumRegister(groups, "gt")
        lt = QuantumRegister(groups, "lt")
        c_out = ClassicalRegister(3, "c_out")
        qc = QuantumCircuit(a, b, high_eq, pair_eq, pair_gt, pair_lt, eq, gt, lt, c_out, name=self.meta.key)

        # Stage 1 compares two-bit blocks in decreasing significance order.
        for group in range(groups):
            hi = n - 1 - 2 * group
            lo = hi - 1
            _two_bit_relation(
                qc,
                a[hi],
                a[lo],
                b[hi],
                b[lo],
                high_eq[group],
                pair_eq[group],
                pair_gt[group],
                pair_lt[group],
            )

            if group == 0:
                qc.cx(pair_eq[group], eq[group])
                qc.cx(pair_gt[group], gt[group])
                qc.cx(pair_lt[group], lt[group])
                continue

            # Stage 2 priority encoding: a lower block contributes only if all
            # preceding blocks were equal.
            qc.ccx(eq[group - 1], pair_eq[group], eq[group])

            qc.cx(gt[group - 1], gt[group])
            qc.ccx(eq[group - 1], pair_gt[group], gt[group])

            qc.cx(lt[group - 1], lt[group])
            qc.ccx(eq[group - 1], pair_lt[group], lt[group])

        last = groups - 1
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
                "high_eq": high_eq,
                "pair_eq": pair_eq,
                "pair_gt": pair_gt,
                "pair_lt": pair_lt,
                "eq": eq,
                "gt": gt,
                "lt": lt,
                "out_qubit": lt[last],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
