"""Functional adaptation of Donaire et al.'s optimised comparator 2.

The paper proposes a full 2-bit comparator in Fig. 9 and reports four
temporary logical-AND blocks, 16 T gates, T-depth 4, and eight ancillary
qubits.  The published drawing, however, uses a line as a temporary-AND target
after that line has already been modified.  Such a target is no longer in the
required |T> state.  This module keeps the reported resources and truth table
but uses four independent |T> targets, so it is intentionally identified as an
adaptation rather than a gate-for-gate transcription.
"""

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
    """Apply the four-T temporary logical-AND decomposition from the paper."""
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


def compute_xnor(qc: QuantumCircuit, first_bit, second_bit, target) -> None:
    """Compute NOT(first_bit XOR second_bit) into a zero-initialized target."""
    qc.x(target)
    qc.cx(first_bit, target)
    qc.cx(second_bit, target)


def compute_open_controlled_and(
    qc: QuantumCircuit, first_control, second_control, target,
) -> None:
    """Compute NOT(first_control) AND second_control into a prepared target."""
    qc.x(first_control)
    temporary_logical_and(qc, first_control, second_control, target)
    qc.x(first_control)


def compute_comparison_terms(qc: QuantumCircuit, a, b, t, work) -> None:
    """Compute the four terms used by the adapted 2-bit comparator.

    On return, t[1] is the lower-bit less-than term gated by equality of the
    most significant bits, t[2] is the most-significant less-than term, and
    t[3] is the equality output.
    """
    compute_xnor(qc, a[1], b[1], work[0])
    compute_xnor(qc, a[0], b[0], work[1])

    for target in t:
        prepare_t_state(qc, target)

    # t[0] = not(a[0]) and b[0]
    compute_open_controlled_and(qc, a[0], b[0], t[0])

    # t[1] = (a[1] == b[1]) and t[0]
    temporary_logical_and(qc, work[0], t[0], t[1])

    # t[3] = (a[1] == b[1]) and (a[0] == b[0])
    temporary_logical_and(qc, work[0], work[1], t[3])

    # t[2] = not(a[1]) and b[1]
    compute_open_controlled_and(qc, a[1], b[1], t[2])


def assemble_one_hot_outputs(qc: QuantumCircuit, t, out) -> None:
    """Assemble lt and gt; equality is already stored in t[3]."""
    # The two less-than terms are mutually exclusive.
    qc.cx(t[2], out[0])
    qc.cx(t[1], out[0])

    # Exactly one of lt, eq, and gt is true.
    qc.x(out[1])
    qc.cx(out[0], out[1])
    qc.cx(t[3], out[1])


class CMP_Donaire_Optimised2_Adapted:
    meta = ComparatorMeta(
        key="cmp_donaire_opt2_adapted",
        name="Adapted optimised proposed circuit 2 (Donaire et al., 2024)",
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
            "Functional 2-bit adaptation of the full comparator intended by "
            "Fig. 9. It uses four temporary logical-AND blocks and reproduces "
            "Table 5's T-count, T-depth, and ancillary-qubit count. The "
            "published diagram reuses a modified |T> line as a later "
            "temporary-AND target; this implementation avoids that state-"
            "preparation conflict and is not a gate-for-gate transcription."
        ),
    )

    def build(
        self, n: int, measure: bool = True, **kwargs,
    ) -> BuildOutput:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be an integer")
        n = int(n)
        if n != 2:
            raise ValueError(
                "cmp_donaire_opt2_adapted implements the 2-bit circuit "
                "derived from Fig. 9, so n must be 2"
            )
        if not isinstance(measure, bool):
            raise TypeError("measure must be a bool")
        if kwargs:
            raise TypeError(f"Unexpected build argument(s): {', '.join(sorted(kwargs))}")

        a = QuantumRegister(2, "a")
        b = QuantumRegister(2, "b")
        t = QuantumRegister(4, "T")
        work = QuantumRegister(2, "work")
        out = QuantumRegister(2, "out")
        qc = QuantumCircuit(a, b, t, work, out, name=self.meta.key)

        compute_comparison_terms(qc, a, b, t, work)
        assemble_one_hot_outputs(qc, t, out)

        output_qubits = {"lt": out[0], "eq": t[3], "gt": out[1]}
        regs = {
            "a": a,
            "b": b,
            "T": t,
            "work": work,
            "out": out,
            "out_qubit": out[0],
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

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("a", "b"),
                solution_qubits=(out[0], t[3], out[1]),
            ),
        )