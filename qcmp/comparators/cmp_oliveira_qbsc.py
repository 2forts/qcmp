from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def uc_instruction():
    """Reusable U_c block from the QBSC construction."""
    a_bit = QuantumRegister(1, "a")
    b_bit = QuantumRegister(1, "b")
    z1 = QuantumRegister(1, "z1")
    z2 = QuantumRegister(1, "z2")

    qc = QuantumCircuit(a_bit, b_bit, z1, z2, name="U_c")
    qc.x(b_bit[0])
    qc.ccx(a_bit[0], b_bit[0], z1[0])
    qc.x(a_bit[0])
    qc.x(b_bit[0])
    qc.ccx(a_bit[0], b_bit[0], z2[0])
    qc.x(a_bit[0])

    return qc.to_instruction(label="U_c")


def open_ccx(qc: QuantumCircuit, control_a, control_b, target) -> None:
    """Toggle target iff both controls are 0."""
    qc.x(control_a)
    qc.x(control_b)
    qc.ccx(control_a, control_b, target)
    qc.x(control_a)
    qc.x(control_b)


class CMP_Oliveira_QBSC:
    meta = ComparatorMeta(
        key="cmp_oliveira_qbsc",
        name="Quantum bit string comparator (Oliveira and Ramos, 2007)",
        paper=(
            'D. S. Oliveira and R. V. Ramos, "Quantum bit string comparator: '
            'circuits and applications", Quantum Computers and Computing 7(1), '
            "17-26 (2007)."
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the QBSC U_c-block construction. The decoded outputs are "
            "gt=1 iff A>B, lt=1 iff A<B, and eq=1 iff A=B."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 2:
            raise ValueError("cmp_oliveira_qbsc requires n >= 2")

        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        z1 = QuantumRegister(n, "z1")
        z2 = QuantumRegister(n, "z2")
        z3 = QuantumRegister(n - 1, "z3")
        c_out = ClassicalRegister(2, "c_out")

        qc = QuantumCircuit(a, b, z1, z2, z3, c_out, name=self.meta.key)
        uc = uc_instruction()

        # The paper labels bits from left to right (A1...An). qcmp loads integer
        # bits little-endian, so stage 0 uses a[n-1]/b[n-1] (the MSB).
        stage_a = [a[n - 1 - i] for i in range(n)]
        stage_b = [b[n - 1 - i] for i in range(n)]

        for i in range(n):
            qc.append(uc, [stage_a[i], stage_b[i], z1[i], z2[i]])

        for i in range(n - 1):
            open_ccx(qc, z1[i], z2[i], z3[i])

        for i in range(n - 1, 0, -1):
            qc.ccx(z1[i], z3[i - 1], z1[i - 1])
            qc.ccx(z2[i], z3[i - 1], z2[i - 1])

        # O1=z1[0], O2=z2[0]. Qiskit count strings print c_out[1] before c_out[0],
        # so decode_counts reads the rightmost bit as O1 and the previous bit as O2.
        qc.measure(z1[0], c_out[0])
        qc.measure(z2[0], c_out[1])

        def decode_counts(counts):
            bitstring = max(counts, key=counts.get).replace(" ", "")
            o1 = int(bitstring[-1])
            o2 = int(bitstring[-2])
            return {
                "gt": int(o1 == 1 and o2 == 0),
                "lt": int(o1 == 0 and o2 == 1),
                "eq": int(o1 == 0 and o2 == 0),
            }

        regs = {
            "a": a,
            "b": b,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "out_qubit": z2[0],
            "c_out": c_out,
        }

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(z1[0], z2[0]),
        )
        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=resources,
        )
