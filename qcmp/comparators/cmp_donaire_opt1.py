from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta


def prepare_t_state(qc: QuantumCircuit, qubit) -> None:
    """Prepare |T> = T|+>, the auxiliary state used by Temporary logical-AND."""
    qc.h(qubit)
    qc.t(qubit)


def temporary_logical_and(qc: QuantumCircuit, x, y, target) -> None:
    """
    Temporary logical-AND block used in Donaire et al.'s Fig. 3.

    The block follows the same Clifford+T pattern used in the paper: four
    T/Tdg gates after the target has been prepared in the |T> state.
    """
    qc.cx(x, target)
    qc.cx(y, target)
    qc.cx(target, x)
    qc.cx(target, y)
    qc.tdg(x)
    qc.tdg(y)
    qc.t(target)
    qc.cx(target, x)
    qc.cx(target, y)
    qc.h(target)
    qc.s(target)


def optimised_tr(qc: QuantumCircuit, a_bit, b_bit, c_bit, t_bit) -> None:
    """
    Donaire et al. optimised TR gate from Fig. 3.

    Input/output action:
        A -> A
        B -> A xor B
        C -> A and not(B_original) xor C

    The T auxiliary qubit is prepared inside this block, matching the paper's
    requirement that it enters the Temporary logical-AND in the |T> state.
    """
    qc.x(b_bit)
    prepare_t_state(qc, t_bit)
    temporary_logical_and(qc, a_bit, b_bit, t_bit)
    qc.cx(t_bit, c_bit)
    qc.x(b_bit)
    qc.cx(a_bit, b_bit)


class CMP_Donaire_Optimised1:
    meta = ComparatorMeta(
        key="cmp_donaire_opt1",
        name="Optimised proposed circuit 1 (Donaire et al., 2024)",
        paper=(
            'L. M. Donaire, G. Ortega, E. M. Garzon, and F. Orts, '
            '"Lowering the cost of quantum comparator circuits", '
            "The Journal of Supercomputing 80, 13900-13917 (2024)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="clean",
        notes=(
            "Implements the 2-bit optimised proposed circuit 1 from Fig. 7. "
            "The relevant paper outputs are Z=1 iff x<y and Y=1 iff x>y; "
            "qcmp decodes these as lt and gt, with eq inferred when both are 0. "
            "The circuit uses six optimised TR blocks from Fig. 3."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n != 2:
            raise ValueError("cmp_donaire_opt1 implements the 2-bit circuit from Fig. 7, so n must be 2")

        x = QuantumRegister(2, "a")
        y = QuantumRegister(2, "b")

        # Working B-lines for the six optimised TR gates. The TR gate mutates
        # its B line to A xor B, so copies keep the original inputs readable.
        b_work = QuantumRegister(6, "b_work")
        t = QuantumRegister(6, "T")

        eq_msb = QuantumRegister(1, "eq_msb")
        terms = QuantumRegister(4, "terms")  # gt_msb, lt_msb, gt_lsb, lt_lsb
        out = QuantumRegister(2, "out")      # out[0]=Z/lt, out[1]=Y/gt
        c_out = ClassicalRegister(2, "c_out")

        qc = QuantumCircuit(x, y, b_work, t, eq_msb, terms, out, c_out, name=self.meta.key)

        # Copy operands used as TR B-lines.
        qc.cx(y[1], b_work[0])  # gt_msb: A=x1, B=y1
        qc.cx(x[1], b_work[1])  # lt_msb: A=y1, B=x1
        qc.cx(y[0], b_work[2])  # gt_lsb: A=x0, B=y0
        qc.cx(x[0], b_work[3])  # lt_lsb: A=y0, B=x0

        # Four bit-level comparisons: A and not(B).
        optimised_tr(qc, x[1], b_work[0], terms[0], t[0])
        optimised_tr(qc, y[1], b_work[1], terms[1], t[1])
        optimised_tr(qc, x[0], b_work[2], terms[2], t[2])
        optimised_tr(qc, y[0], b_work[3], terms[3], t[3])

        # eq_msb = not(x1 xor y1)
        qc.x(eq_msb[0])
        qc.cx(x[1], eq_msb[0])
        qc.cx(y[1], eq_msb[0])

        # Fig. 7 outputs: Y for x>y and Z for x<y.
        qc.cx(terms[0], out[1])
        qc.cx(terms[1], out[0])

        # Use two more optimised TR blocks to add the lower-bit decision only
        # when the most significant bits are equal.
        qc.x(b_work[4])
        qc.cx(terms[2], b_work[4])  # b_work[4] = not(gt_lsb)
        optimised_tr(qc, eq_msb[0], b_work[4], out[1], t[4])

        qc.x(b_work[5])
        qc.cx(terms[3], b_work[5])  # b_work[5] = not(lt_lsb)
        optimised_tr(qc, eq_msb[0], b_work[5], out[0], t[5])

        qc.measure(out[0], c_out[0])
        qc.measure(out[1], c_out[1])

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
            "a": x,
            "b": y,
            "b_work": b_work,
            "T": t,
            "eq_msb": eq_msb,
            "terms": terms,
            "out": out,
            "out_qubit": out[0],
            "c_out": c_out,
        }

        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts)
