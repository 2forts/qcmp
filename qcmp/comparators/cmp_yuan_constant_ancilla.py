from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _one_bit_relation(qc: QuantumCircuit, a_bit, b_bit, gt, lt) -> None:
    """Write (a_bit>b_bit, a_bit<b_bit) into two clean target qubits."""
    qc.x(b_bit)
    qc.ccx(a_bit, b_bit, gt)
    qc.x(b_bit)

    qc.x(a_bit)
    qc.ccx(a_bit, b_bit, lt)
    qc.x(a_bit)


def _equality_from_relation(qc: QuantumCircuit, gt, lt, equal) -> None:
    """The relation bits are both zero exactly when their source bits match."""
    qc.x(gt)
    qc.x(lt)
    qc.ccx(gt, lt, equal)
    qc.x(lt)
    qc.x(gt)


class CMP_Yuan_Constant_Ancilla:
    meta = ComparatorMeta(
        key="cmp_yuan_constant_ancilla",
        name="Constant-ancilla three-way comparator (Yuan et al., 2020)",
        paper=(
            'S. Yuan, C. Wen, B. Hang, and Y. Gong, "The dual-threshold '
            'quantum image segmentation algorithm and its simulation", '
            'Quantum Information Processing 19, 425 (2020).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="reset",
        notes=(
            "Implements the five-auxiliary-qubit comparison pattern of Figure 9. "
            "It combines one-bit relations from least to most significant bit and "
            "resets no-longer-needed work qubits, as in the paper."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        s = QuantumRegister(5, "s")
        c_out = ClassicalRegister(2, "c_out")
        qc = QuantumCircuit(a, b, s, c_out, name=self.meta.key)

        # Figure 9 starts with the least-significant one-bit relation in s1/s2.
        # qcmp's integer encoding is little-endian, so bit 0 is that LSB.
        current_gt, current_lt = s[0], s[1]
        work_gt, work_lt, equal = s[2], s[3], s[4]
        _one_bit_relation(qc, a[0], b[0], current_gt, current_lt)

        for bit in range(1, n):
            # Compute the relation of the new, more-significant bit. If it is
            # equal, fold the previous relation into it. The terms are mutually
            # exclusive, so the controlled-X operations implement the OR.
            _one_bit_relation(qc, a[bit], b[bit], work_gt, work_lt)
            _equality_from_relation(qc, work_gt, work_lt, equal)
            qc.ccx(current_gt, equal, work_gt)
            qc.ccx(current_lt, equal, work_lt)

            # Yuan et al. reset the old pair and equality workspace so the same
            # five ancillary qubits can be reused at every next bit position.
            qc.reset(current_gt)
            qc.reset(current_lt)
            qc.reset(equal)
            current_gt, work_gt = work_gt, current_gt
            current_lt, work_lt = work_lt, current_lt

        qc.measure(current_gt, c_out[0])
        qc.measure(current_lt, c_out[1])

        def decode_counts(counts):
            bitstring = max(counts, key=counts.get).replace(" ", "")
            gt = int(bitstring[-1])
            lt = int(bitstring[-2])
            return {"gt": gt, "lt": lt, "eq": int(gt == 0 and lt == 0)}

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(current_gt, current_lt),
            restored_zero_qubits=(work_gt, work_lt, equal),
        )
        return BuildOutput(
            circuit=qc,
            regs={
                "a": a,
                "b": b,
                "s": s,
                "out_qubit": current_lt,
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
