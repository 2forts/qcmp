from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


class CMP_Phaneendra_Prefix:
    meta = ComparatorMeta(
        key="cmp_phaneendra_prefix",
        name="Optimized prefix reversible quantum comparator (Phaneendra et al., 2014)",
        paper=(
            'S. P. Phaneendra, C. Vudadha, V. Sreehari, and M. B. Srinivas, '
            '"An Optimized Design of Reversible Quantum Comparator", 2014 '
            '27th International Conference on VLSI Design and 2014 13th '
            'International Conference on Embedded Systems, 557-562 (2014).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the paper's three-stage prefix construction: local "
            "G/E generation, prefix grouping with E = E_high E_low and "
            "G = G_high XOR (E_high G_low), and the final lesser output."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        leaf_eq = QuantumRegister(n, "leaf_eq")
        leaf_gt = QuantumRegister(n, "leaf_gt")
        lt = QuantumRegister(1, "lt")
        c_out = ClassicalRegister(3, "c_out")

        registers = [a, b, leaf_eq, leaf_gt]
        if n > 1:
            group_eq = QuantumRegister(n - 1, "group_eq")
            group_gt = QuantumRegister(n - 1, "group_gt")
            registers.extend((group_eq, group_gt))
        else:
            group_eq = None
            group_gt = None
        registers.extend((lt, c_out))
        qc = QuantumCircuit(*registers, name=self.meta.key)

        # Stage 1 of the paper: G_i = A_i AND NOT B_i and E_i = NOT(A_i XOR B_i).
        for bit in range(n):
            qc.x(leaf_eq[bit])
            qc.cx(a[bit], leaf_eq[bit])
            qc.cx(b[bit], leaf_eq[bit])

            qc.x(b[bit])
            qc.ccx(a[bit], b[bit], leaf_gt[bit])
            qc.x(b[bit])

        # Stage 2: combine adjacent high/low groups into a prefix tree. The
        # nodes remain ordered from most to least significant bit.
        nodes = [(leaf_eq[bit], leaf_gt[bit]) for bit in range(n - 1, -1, -1)]
        merge = 0
        while len(nodes) > 1:
            next_nodes = []
            for index in range(0, len(nodes) - 1, 2):
                high_eq, high_gt = nodes[index]
                low_eq, low_gt = nodes[index + 1]

                qc.ccx(high_eq, low_eq, group_eq[merge])
                qc.cx(high_gt, group_gt[merge])
                qc.ccx(high_eq, low_gt, group_gt[merge])
                next_nodes.append((group_eq[merge], group_gt[merge]))
                merge += 1

            if len(nodes) % 2:
                next_nodes.append(nodes[-1])
            nodes = next_nodes

        final_eq, final_gt = nodes[0]

        # Stage 3: G and E cannot be simultaneously one, so LT = NOT(GT OR EQ).
        qc.x(lt[0])
        qc.cx(final_gt, lt[0])
        qc.cx(final_eq, lt[0])

        qc.measure(final_gt, c_out[0])
        qc.measure(lt[0], c_out[1])
        qc.measure(final_eq, c_out[2])

        def decode_counts(counts):
            bitstring = max(counts, key=counts.get).replace(" ", "")
            return {
                "gt": int(bitstring[-1]),
                "lt": int(bitstring[-2]),
                "eq": int(bitstring[-3]),
            }

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(final_gt, lt[0], final_eq),
        )
        regs = {
            "a": a,
            "b": b,
            "leaf_eq": leaf_eq,
            "leaf_gt": leaf_gt,
            "lt": lt,
            "out_qubit": lt[0],
            "c_out": c_out,
        }
        if group_eq is not None and group_gt is not None:
            regs["group_eq"] = group_eq
            regs["group_gt"] = group_gt
        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=resources,
        )
