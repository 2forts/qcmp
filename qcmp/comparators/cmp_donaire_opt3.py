"""Gate-level implementation of Donaire et al.'s optimised comparator 3."""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


def prepare_t_state(qc: QuantumCircuit, target) -> None:
    """Prepare T|+>, the target state required by temporary logical-AND."""
    qc.h(target)
    qc.t(target)


def temporary_logical_and(
    qc: QuantumCircuit, first_control, second_control, target,
) -> None:
    """Apply the four-T temporary logical-AND decomposition from Fig. 11."""
    qc.cx(first_control, target)
    qc.cx(second_control, target)
    qc.cx(target, first_control)
    qc.cx(target, second_control)
    qc.tdg(first_control)
    qc.tdg(second_control)
    qc.t(target)
    qc.cx(target, first_control)
    qc.cx(target, second_control)
    qc.h(target)
    qc.s(target)


def optimised_gn(qc: QuantumCircuit, x, y, t_target, less_output) -> None:
    """Implement the optimised GN network shown in Fig. 11.

    The final values are x=(x_original == y), t_target=(x_original > y),
    and less_output=(x_original < y).  The y wire is preserved.
    """
    qc.x(x)
    prepare_t_state(qc, t_target)
    temporary_logical_and(qc, x, y, t_target)
    qc.x(x)

    qc.cx(t_target, less_output)
    qc.cx(y, x)
    qc.cx(x, t_target)
    qc.x(x)


class CMP_Donaire_Optimised3:
    meta = ComparatorMeta(
        key="cmp_donaire_opt3",
        name="Optimised proposed circuit 3 (Donaire et al., 2024)",
        paper=(
            'L. M. Donaire, G. Ortega, E. M. Garzon, and F. Orts, '
            '"Lowering the cost of quantum comparator circuits", '
            "The Journal of Supercomputing 80, 13900-13917 (2024), "
            "doi:10.1007/s11227-024-05959-4."
        ),
        kind="cmp",
        signed=False,
        outputs=("lt", "eq", "gt"),
        ancilla_mode="clean",
        notes=(
            "Gate-level implementation of the 1-bit full comparator in "
            "Fig. 11. The input a line is overwritten by eq, the T line ends "
            "as gt, and a zero-initialized line ends as lt."
        ),
    )

    def build(
        self, n: int, measure: bool = True, **kwargs,
    ) -> BuildOutput:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be an integer")
        n = int(n)
        if n != 1:
            raise ValueError(
                "cmp_donaire_opt3 implements the 1-bit circuit from Fig. 11, "
                "so n must be 1"
            )
        if not isinstance(measure, bool):
            raise TypeError("measure must be a bool")
        if kwargs:
            raise TypeError(f"Unexpected build argument(s): {', '.join(sorted(kwargs))}")

        a = QuantumRegister(1, "a")
        b = QuantumRegister(1, "b")
        t = QuantumRegister(1, "T")
        less = QuantumRegister(1, "less")
        qc = QuantumCircuit(a, b, t, less, name=self.meta.key)

        optimised_gn(qc, a[0], b[0], t[0], less[0])

        output_qubits = {"lt": less[0], "eq": a[0], "gt": t[0]}
        regs = {
            "a": a,
            "b": b,
            "T": t,
            "less": less,
            "out_qubit": less[0],
            "out_qubits": output_qubits,
        }

        decode_counts = None
        if measure:
            c_out = ClassicalRegister(3, "c_out")
            qc.add_register(c_out)
            regs["c_out"] = c_out
            for index, label in enumerate(("lt", "eq", "gt")):
                qc.measure(output_qubits[label], c_out[index])
            output_indices = {
                label: qc.find_bit(c_out[index]).index
                for index, label in enumerate(("lt", "eq", "gt"))
            }

            def decode_counts(counts):
                if not counts:
                    raise ValueError("Cannot decode empty measurement counts")
                result = {}
                for label, index in output_indices.items():
                    selected = marginal_counts(counts, indices=[index])
                    result[label] = int(max(selected, key=selected.get))
                return result

        # The equality result is written in place on a[0]. ResourceLayout
        # therefore retains that wire as an operand and lists the two
        # auxiliary output wires as solutions.
        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("a", "b"),
                solution_qubits=(less[0], t[0]),
            ),
        )