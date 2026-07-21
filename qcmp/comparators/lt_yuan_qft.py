from __future__ import annotations

from math import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.result import marginal_counts

from qcmp.comparators.base import ComparatorMeta, BuildOutput, ResourceLayout


def apply_qft(qc: QuantumCircuit, qubits) -> None:
    qc.append(QFT(len(qubits), do_swaps=False).to_instruction(), qubits)


def apply_iqft(qc: QuantumCircuit, qubits) -> None:
    qc.append(QFT(len(qubits), do_swaps=False, inverse=True).to_instruction(), qubits)


def controlled_phase_add(
    qc: QuantumCircuit,
    source,
    target_fourier,
    sign: int = +1,
) -> None:
    """
    Add (sign=+1) or subtract (sign=-1) the value in `source`
    onto `target_fourier`, assuming `target_fourier` is already
    in the Fourier basis and registers are little-endian.

    source[j] contributes 2^j.
    target_fourier[k] gets angle 2*pi / 2^(k-j+1) for k >= j.
    """
    m = len(target_fourier)
    n = len(source)

    for j in range(n):
        for k in range(j, m):
            angle = sign * (2 * pi) / (2 ** (k - j + 1))
            qc.cp(angle, source[j], target_fourier[k])


class LT_Yuan_QFT:
    meta = ComparatorMeta(
        key="lt_yuan_qft",
        name="QFT-based quantum states comparator (Yuan et al., 2023)",
        paper=(
            'Y. Yuan et al., "An improved QFT-based quantum comparator and '
            'extended modular arithmetic using one ancilla qubit", '
            'New Journal of Physics 25 (2023) 103011.'
        ),
        kind="lt",
        signed=False,
        outputs=("lt",),
        ancilla_mode="clean",
        notes=(
            "Implements the quantum-states comparator from Appendix A "
            "(Figure 10). The framework loads A into register a and B into "
            "register b, and the circuit preserves both inputs."
        ),
    )

    def build(self, n: int, sense: str = "lt", **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError("n must be >= 1")
        if sense not in ("lt", "gt"):
            raise ValueError("sense must be 'lt' or 'gt'")

        # x1 = a, x2 = b in the current framework
        a = QuantumRegister(n, "a")
        b = QuantumRegister(n, "b")
        anc = QuantumRegister(1, "anc")
        c_out = ClassicalRegister(1, "c_out")

        qc = QuantumCircuit(a, b, anc, c_out, name=self.meta.key)

        # Extended target register for the subtraction stage:
        # |x2>|0>  -> use anc as the most significant overflow/sign qubit.
        target_ext = list(b[:]) + [anc[0]]

        # ------------------------------------------------------------------
        # Stage 1: compute comparison information by subtracting x1 from x2
        # on the Fourier basis of the extended register |x2>|0>.
        #
        # Result anc = 1 iff x2 < x1.
        # In framework terms:
        #   anc = 1 iff B < A
        # ------------------------------------------------------------------
        apply_qft(qc, target_ext)
        controlled_phase_add(qc, a, target_ext, sign=-1)
        apply_iqft(qc, target_ext)

        # ------------------------------------------------------------------
        # Stage 2: restore x2 by adding x1 back only on the n-bit b register.
        # This preserves both inputs while keeping anc as the comparison bit.
        # ------------------------------------------------------------------
        apply_qft(qc, list(b[:]))
        controlled_phase_add(qc, a, list(b[:]), sign=+1)
        apply_iqft(qc, list(b[:]))

        qc.measure(anc[0], c_out[0])

        def decode_counts(counts):
            idx = qc.clbits.index(c_out[0])
            mc = marginal_counts(counts, indices=[idx])
            bit = max(mc, key=mc.get)  # "0" or "1"

            raw_b_lt_a = int(bit)      # 1 iff B < A

            if sense == "lt":
                # We want A < B
                return {"lt": 1 - raw_b_lt_a}
            else:
                # A > B
                return {"gt": raw_b_lt_a}

        regs = {
            "a": a,
            "b": b,
            "anc": anc,
            "out_qubit": anc[0],
            "c_out": c_out,
        }

        resources = ResourceLayout(
            operand_registers=("a", "b"),
            solution_qubits=(anc[0],),
        )

        return BuildOutput(
            circuit=qc,
            regs=regs,
            decode_counts=decode_counts,
            resources=resources,
        )