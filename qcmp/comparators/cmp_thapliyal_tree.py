from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _open_ccx(qc: QuantumCircuit, controls, target, open_indexes: set[int]) -> None:
    for index in open_indexes:
        qc.x(controls[index])
    qc.ccx(controls[0], controls[1], target)
    for index in reversed(list(open_indexes)):
        qc.x(controls[index])


class CMP_Thapliyal_Tree:
    meta = ComparatorMeta(
        key="cmp_thapliyal_tree",
        name="Comparator tree based on reversible logic (Thapliyal et al., 2009)",
        paper=(
            'H. Thapliyal, N. Ranganathan, and R. Ferreira, "Design of a '
            'Comparator Tree Based on Reversible Logic", IEEE International '
            "Conference on Nanotechnology (2009)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="dirty",
        notes=(
            "Implements the binary-tree comparator structure. Leaves are 2-bit "
            "reversible comparator nodes; internal nodes combine high/low "
            "partial comparison results. Supports n as a power of two."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 2 or not _is_power_of_two(n):
            raise ValueError("cmp_thapliyal_tree requires n to be a power of two and n >= 2")

        leaf_count = n // 2
        node_count = 2 * leaf_count - 1
        combine_count = leaf_count - 1

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        node_lt = QuantumRegister(node_count, "node_lt")
        node_gt = QuantumRegister(node_count, "node_gt")
        leaf_work = QuantumRegister(5 * leaf_count, "leaf_work")
        combine_eq = QuantumRegister(max(combine_count, 1), "combine_eq")
        c_out = ClassicalRegister(2, "c_out")  # c_out[0]=lt, c_out[1]=gt

        qc = QuantumCircuit(a, b, node_lt, node_gt, leaf_work, combine_eq, c_out, name=self.meta.key)

        # Leaves compare adjacent 2-bit groups, from most significant group to least.
        current_level = []
        for leaf in range(leaf_count):
            hi = n - 1 - 2 * leaf
            lo = hi - 1
            gt_hi = leaf_work[5 * leaf]
            lt_hi = leaf_work[5 * leaf + 1]
            gt_lo = leaf_work[5 * leaf + 2]
            lt_lo = leaf_work[5 * leaf + 3]
            eq_hi = leaf_work[5 * leaf + 4]

            _open_ccx(qc, [a[hi], b[hi]], gt_hi, {1})
            _open_ccx(qc, [a[hi], b[hi]], lt_hi, {0})
            _open_ccx(qc, [a[lo], b[lo]], gt_lo, {1})
            _open_ccx(qc, [a[lo], b[lo]], lt_lo, {0})

            qc.x(eq_hi)
            qc.cx(a[hi], eq_hi)
            qc.cx(b[hi], eq_hi)

            qc.cx(lt_hi, node_lt[leaf])
            qc.ccx(eq_hi, lt_lo, node_lt[leaf])
            qc.cx(gt_hi, node_gt[leaf])
            qc.ccx(eq_hi, gt_lo, node_gt[leaf])
            current_level.append(leaf)

        next_node = leaf_count
        next_eq = 0
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                high = current_level[i]
                low = current_level[i + 1]
                parent = next_node
                eq_high = combine_eq[next_eq]
                next_node += 1
                next_eq += 1

                qc.x(eq_high)
                qc.cx(node_lt[high], eq_high)
                qc.cx(node_gt[high], eq_high)

                qc.cx(node_lt[high], node_lt[parent])
                qc.ccx(eq_high, node_lt[low], node_lt[parent])
                qc.cx(node_gt[high], node_gt[parent])
                qc.ccx(eq_high, node_gt[low], node_gt[parent])
                next_level.append(parent)
            current_level = next_level

        root = current_level[0]
        qc.measure(node_lt[root], c_out[0])
        qc.measure(node_gt[root], c_out[1])

        def decode_counts(counts):
            values = {}
            for label, cbit in (("lt", c_out[0]), ("gt", c_out[1])):
                idx = qc.clbits.index(cbit)
                mc = marginal_counts(counts, indices=[idx])
                bit = max(mc, key=mc.get)
                values[label] = int(bit)
            values["eq"] = int(values["lt"] == 0 and values["gt"] == 0)
            return {"lt": values["lt"], "eq": values["eq"], "gt": values["gt"]}

        regs = {
            "a": a,
            "b": b,
            "node_lt": node_lt,
            "node_gt": node_gt,
            "leaf_work": leaf_work,
            "combine_eq": combine_eq,
            "out_qubit": node_lt[root],
            "c_out": c_out,
        }
        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(node_lt[root], node_gt[root]),
        )
        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts, resources=resources)
