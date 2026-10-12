from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def _al_rabadi_cell_fig14a(
    qc: QuantumCircuit,
    x_i,
    y_i,
    a_i,
    b_i,
    z0,
    z1,
    z2,
    z3,
    a_next,
    b_next,
) -> None:
    """
    Gate-by-gate transcription of Al-Rabadi 2009, Fig. 14(a).

    Logical role:
        x_i, y_i : current compared digits
        a_i      : previous state, X_prefix < Y_prefix
        b_i      : previous state, X_prefix > Y_prefix

    Outputs:
        a_next   : next state, X_prefix_new < Y_prefix_new
        b_next   : next state, X_prefix_new > Y_prefix_new

    z0..z3 are the four internal |0> lines of Fig. 14(a).

    This cell is intentionally NOT cleaned/uncomputed, because Fig. 14(a)
    is a reversible synthesis with garbage lines, not a clean oracle.
    """

    # Column 1: NOT on x_i and a_i.
    qc.x(x_i)
    qc.x(a_i)

    # Column 2: Toffoli with controls x_i, y_i and target z0.
    # Since x_i has been inverted, z0 = NOT(x_i_original) AND y_i.
    qc.ccx(x_i, y_i, z0)

    # Column 3: NOT on x_i, y_i and b_i.
    qc.x(x_i)
    qc.x(y_i)
    qc.x(b_i)

    # Column 4: Toffoli with controls z0, b_i and target z1.
    # Since b_i has been inverted:
    # z1 = (NOT x_i_original AND y_i_original) AND NOT(b_i_original).
    qc.ccx(z0, b_i, z1)

    # Column 5: NOT on z1.
    qc.x(z1)

    # Column 6: Toffoli with controls x_i, y_i and target z2.
    # At this point x_i is restored and y_i is inverted:
    # z2 = x_i_original AND NOT(y_i_original).
    qc.ccx(x_i, y_i, z2)

    # Column 7: Toffoli with controls a_i, z2 and target z3.
    # Since a_i has been inverted:
    # z3 = NOT(a_i_original) AND x_i_original AND NOT(y_i_original).
    qc.ccx(a_i, z2, z3)

    # Column 8: NOT on z3.
    qc.x(z3)

    # Column 9: Toffoli with controls a_i, z1 and target a_next.
    qc.ccx(a_i, z1, a_next)

    # Column 10: Toffoli with controls b_i, z3 and target b_next.
    qc.ccx(b_i, z3, b_next)

    # Column 11: NOT on a_next and b_next.
    qc.x(a_next)
    qc.x(b_next)


def _al_rabadi_output_fig14b(
    qc: QuantumCircuit,
    a_final,
    b_final,
    eq_target,
) -> None:
    """
    Gate-by-gate transcription of Al-Rabadi 2009, Fig. 14(b).

    Given:
        a_final = X < Y
        b_final = X > Y
        eq_target initialized to |0>

    Computes:
        eq_target = X == Y = NOT(a_final) AND NOT(b_final)

    a_final and b_final are restored.
    """

    qc.x(a_final)
    qc.x(b_final)
    qc.ccx(a_final, b_final, eq_target)
    qc.x(a_final)
    qc.x(b_final)


class CMP_Al_Rabadi:
    meta = ComparatorMeta(
        key="cmp_al_rabadi",
        name="Gate-faithful iterative comparator (Al-Rabadi, 2009)",
        paper=(
            'A. N. Al-Rabadi, "Closed-System Quantum Logic Network '
            'Implementation of the Viterbi Algorithm", Facta Universitatis, '
            "Series: Electronics and Energetics 22(1), 1-33 (2009)."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="garbage",
        notes=(
            "Gate-by-gate transcription of the iterative comparator from "
            "Figures 14(a) and 14(b) of Al-Rabadi (2009). The circuit uses "
            "one Fig. 14(a) cell per compared bit and the Fig. 14(b) output "
            "circuit. It intentionally leaves reversible garbage and does not "
            "uncompute internal states."
        ),
    )

    def build(self, n: int, **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")

        # User-facing compared registers:
        # X = a[n-1] ... a[0]
        # Y = b[n-1] ... b[0]
        #
        # The paper labels digits left-to-right as x1...xn and y1...yn.
        # Qiskit integer registers are normally little-endian, so we feed
        # the cells from MSB to LSB.
        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")

        # Paper state chains:
        # a_state[0] = a1 = 0
        # b_state[0] = b1 = 0
        # a_state[i+1], b_state[i+1] are produced by cell i.
        #
        # Here:
        #   a_state means "X prefix < Y prefix"
        #   b_state means "X prefix > Y prefix"
        a_state = QuantumRegister(n + 1, "a_state")
        b_state = QuantumRegister(n + 1, "b_state")

        # Fig. 14(a) uses four additional internal |0> lines per cell.
        garbage = QuantumRegister(4 * n, "garbage")

        # Fig. 14(b) uses one |0> target to produce equality.
        eq = QuantumRegister(1, "eq")

        c_out = ClassicalRegister(3, "c_out")

        qc = QuantumCircuit(
            a,
            b,
            a_state,
            b_state,
            garbage,
            eq,
            c_out,
            name=self.meta.key,
        )

        stage_x = [a[n - 1 - i] for i in range(n)]
        stage_y = [b[n - 1 - i] for i in range(n)]

        for i, (x_i, y_i) in enumerate(zip(stage_x, stage_y)):
            g_base = 4 * i

            _al_rabadi_cell_fig14a(
                qc=qc,
                x_i=x_i,
                y_i=y_i,
                a_i=a_state[i],
                b_i=b_state[i],
                z0=garbage[g_base + 0],
                z1=garbage[g_base + 1],
                z2=garbage[g_base + 2],
                z3=garbage[g_base + 3],
                a_next=a_state[i + 1],
                b_next=b_state[i + 1],
            )

        # Fig. 14(b): compute equality from final less-than / greater-than flags.
        _al_rabadi_output_fig14b(
            qc,
            a_final=a_state[n],
            b_final=b_state[n],
            eq_target=eq[0],
        )

        # Output convention:
        #   c_out[0] = lt = X < Y
        #   c_out[1] = eq = X = Y
        #   c_out[2] = gt = X > Y
        qc.measure(a_state[n], c_out[0])
        qc.measure(eq[0], c_out[1])
        qc.measure(b_state[n], c_out[2])

        def decode_counts(counts):
            values = {}
            for label, cbit in (
                ("lt", c_out[0]),
                ("eq", c_out[1]),
                ("gt", c_out[2]),
            ):
                idx = qc.clbits.index(cbit)
                mc = marginal_counts(counts, indices=[idx])
                bit = max(mc, key=mc.get)
                values[label] = int(bit)
            return values

        regs = {
            "a": a,
            "b": b,
            "a_state": a_state,
            "b_state": b_state,
            "garbage": garbage,
            "eq": eq,
            "lt_qubit": a_state[n],
            "eq_qubit": eq[0],
            "gt_qubit": b_state[n],
            "out_qubits": (a_state[n], eq[0], b_state[n]),
            "out_qubit": a_state[n],
            "c_out": c_out,
        }

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("a", "b"),
                solution_qubits=(a_state[n], eq[0], b_state[n]),
                restored_zero_qubits=(),
            ),
        )
