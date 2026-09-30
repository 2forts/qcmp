from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _tr_gate(qc: QuantumCircuit, a_bit, b_line, target) -> None:
    """TR mapping: (A, B, C) -> (A, A XOR B, A NOT(B) XOR C)."""
    qc.x(b_line)
    qc.ccx(a_bit, b_line, target)
    qc.x(b_line)
    qc.cx(a_bit, b_line)


def _inverse_tr_gate(qc: QuantumCircuit, a_bit, b_line, target) -> None:
    qc.cx(a_bit, b_line)
    qc.x(b_line)
    qc.ccx(a_bit, b_line, target)
    qc.x(b_line)


def _bjn_gate(qc: QuantumCircuit, p_bit, q_bit, target) -> None:
    """BJN mapping for a clean target: target ^= NOT(P) AND NOT(Q)."""
    qc.x(p_bit)
    qc.x(q_bit)
    qc.ccx(p_bit, q_bit, target)
    qc.x(q_bit)
    qc.x(p_bit)


class CMP_Biswal_FaultTolerant:
    meta = ComparatorMeta(
        key="cmp_biswal_fault_tolerant",
        name="Fault-tolerant 4-bit comparator structure (Biswal et al., 2019)",
        paper=(
            'L. Biswal, C. Bandyopadhyay, and H. Rahaman, "Fault-tolerant '
            'Quantum Implementation of 1-bit and 4-bit Comparator Circuit using '
            'Clifford+T-group", IEEE (2019).'
        ),
        kind="cmp",
        signed=False,
        outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",
        notes=(
            "Implements the corrected 4-bit reversible comparison structure of "
            "Figure 5: one TR relation block per bit and a BJN equality output. "
            "The Boolean reversible layer is validated here; the paper's fully "
            "optimized Clifford+T realization in Figure 7 is not reproduced gate "
            "for gate, so its reported T-depth must not be inferred from this circuit."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n != 4:
            raise ValueError("cmp_biswal_fault_tolerant implements the paper's fixed 4-bit circuit, so n must be 4")

        a = QuantumRegister(4, "a")
        b = QuantumRegister(4, "b")
        tr_b = QuantumRegister(1, "tr_b")
        bit_gt = QuantumRegister(1, "bit_gt")
        bit_eq = QuantumRegister(1, "bit_eq")
        bit_lt = QuantumRegister(1, "bit_lt")
        eq = QuantumRegister(4, "eq")
        gt = QuantumRegister(4, "gt")
        lt = QuantumRegister(4, "lt")
        out_eq = QuantumRegister(1, "out_eq")
        c_out = ClassicalRegister(3, "c_out")
        qc = QuantumCircuit(
            a,
            b,
            tr_b,
            bit_gt,
            bit_eq,
            bit_lt,
            eq,
            gt,
            lt,
            out_eq,
            c_out,
            name=self.meta.key,
        )

        # The Figure 5 TR blocks produce A_i AND NOT(B_i). A clean work
        # register is reused after each stage is propagated and uncomputed.
        for stage, bit in enumerate(range(3, -1, -1)):
            qc.cx(b[bit], tr_b[0])
            _tr_gate(qc, a[bit], tr_b[0], bit_gt[0])

            # tr_b now contains A_i XOR B_i, so its inversion is XNOR.
            qc.x(bit_eq[0])
            qc.cx(tr_b[0], bit_eq[0])

            # The three one-bit relations are mutually exclusive.
            qc.x(bit_lt[0])
            qc.cx(bit_gt[0], bit_lt[0])
            qc.cx(bit_eq[0], bit_lt[0])

            if stage == 0:
                qc.cx(bit_gt[0], gt[stage])
                qc.cx(bit_lt[0], lt[stage])
                qc.cx(bit_eq[0], eq[stage])
            else:
                qc.cx(gt[stage - 1], gt[stage])
                qc.ccx(eq[stage - 1], bit_gt[0], gt[stage])

                qc.cx(lt[stage - 1], lt[stage])
                qc.ccx(eq[stage - 1], bit_lt[0], lt[stage])

                qc.ccx(eq[stage - 1], bit_eq[0], eq[stage])

            qc.cx(bit_eq[0], bit_lt[0])
            qc.cx(bit_gt[0], bit_lt[0])
            qc.x(bit_lt[0])
            qc.cx(tr_b[0], bit_eq[0])
            qc.x(bit_eq[0])
            _inverse_tr_gate(qc, a[bit], tr_b[0], bit_gt[0])
            qc.cx(b[bit], tr_b[0])

        # The BJN gate in Figure 5 combines the final non-greater and
        # non-lesser signals into equality.
        _bjn_gate(qc, gt[3], lt[3], out_eq[0])

        qc.measure(gt[3], c_out[0])
        qc.measure(lt[3], c_out[1])
        qc.measure(out_eq[0], c_out[2])

        def decode_counts(counts):
            bitstring = max(counts, key=counts.get).replace(" ", "")
            return {
                "gt": int(bitstring[-1]),
                "lt": int(bitstring[-2]),
                "eq": int(bitstring[-3]),
            }

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(gt[3], lt[3], out_eq[0]),
            restored_zero_qubits=(tr_b[0], bit_gt[0], bit_eq[0], bit_lt[0]),
        )
        return BuildOutput(
            circuit=qc,
            regs={
                "a": a,
                "b": b,
                "tr_b": tr_b,
                "bit_gt": bit_gt,
                "bit_eq": bit_eq,
                "bit_lt": bit_lt,
                "eq": eq,
                "gt": gt,
                "lt": lt,
                "out_eq": out_eq,
                "out_qubit": lt[3],
                "c_out": c_out,
            },
            decode_counts=decode_counts,
            resources=resources,
        )
