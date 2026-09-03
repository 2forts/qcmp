"""Toffoli Comparator ADAPTED from Ganapathy et al. (preprint, 2025).

Source: Vieeralingaam Ganapathy, Sasikumar S, Mariselvam A K, Mohamed Nizar S,
Senthil Kumar V M, "Design and Analysis of Quantum Comparator Circuits
for Constrained Optimization Problems." DOI: 10.21203/rs.3.rs-7733902/v1.
Provided PDF: 28-Ganapathy.pdf. Does not implement the QFT alternative described in the text.

SOURCE: search for [PAPER] and [ADAPTATION] in the functions.
---------------------------------------------------------
[PAPER P1] Sec. 2.1, pp. 3–4: EQ = NOT(A XOR B), GT = A AND NOT(B),
LT = NOT(A) AND B. The sequence of X, CX, and CCX is specified in the text.
For n = 1, that sequence is implemented verbatim (without preprocessing the data).

[PAPER P2] Our interpretation of Fig. 3, p. 11 (page 12 of the PDF): separate
copies of XOR and XNOR are computed for each pair of bits. An AND chain
propagates the equality of the upper bits. The GT/LT outputs are alternated
using X gates with THREE inputs: A/B bit, local XOR, and upper equality.
In Fig. 3: A=(q0,q1), B=(q2,q3), XOR=(q9,q10), XNOR=(q7,q8),
prefixes=(q5,q6), EQ=q4, GT=q11, and LT=q12; q6 is set to one.
The tuplets above are in little-endian order, not the order shown in the figure.

[ADAPTATION A1] Fig. 3 shows only examples of TWO bits, and Fig. 4
deals with backpack constraints, not a general comparator for two
four-bit words. We explicitly extend the pattern from Fig. 3:
  D_i = A_i XOR B_i; E_i = NOT(D_i).
  P_(n-1) = 1; P_(i-1) = P_i AND E_i.
  GT = XOR_i(P_i AND D_i AND A_i).
  LT = XOR_i(P_i AND D_i AND B_i).
  EQ = P_0 AND E_0.
Only the MOST SIGNIFICANT differing bit can contribute: the terms
are mutually exclusive, so XOR is equivalent to OR here.

[ADAPTATION A2] Each X of the three controls in Fig. 3 is synthesized using THREE
CCXs and ONE reusable clean auxiliary gate. This decomposition is not specified
in the article. Neither relative-phase Toffoli gates nor opaque MCX gates are used.

[ADAPTATION A3] The redundant X-X pairs of operands in
Fig. 3 are omitted; the preparation of A/B is separated, and only the three results are measured.
The Xs that prepare constants or XNORs ARE included in the resources.

[ADAPTATION A4] After writing GT/LT/EQ, the prefixes and
the XOR/XNOR operations are undone. This additional cleanup is NOT shown in Fig. 3. The
inputs remain intact, and ALL work is reset to zero. With measure=False,
a consistent oracle is obtained with XOR write operations on the three outputs;
there are no intermediate measurements, resets, discards, or classical A/B evaluation.

[ADAPTATION A5] qcmp API: unsigned integers, little-endian registers,
initially zero working registers, results (gt, lt, eq). For other output states,
these bits are swapped; they are not overwritten. Does not support working registers
in unknown states.

RESOURCES FROM THIS CODE, NOT FIGURES FROM THE PAPER
-----------------------------------------
Without counting measurements, with qcmp profiles X=CX=1, CCX=5, CCX T-count=7:
  n=1: X=5, CX=2, CCX=2, cost=17, T-count=14, qubits=5, auxiliary qubits=3.
  n>=2: X=2*n+2, CX=8*n, CCX=8*n-1.
         cost=50*n-3, T-count=56*n-7.
         qubits=5*n+4, auxiliary qubits=3*n+4, clean work=3*n+1.
  Both cases: three results, zero garbage outputs, three optional measurements.
The T-count is a per-profile estimate; the module outputs X/CX/CCX, not a
Clifford+T physical synthesis. The equations include the additional cleanup.
"""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


# 1. Funciones de un bit: secuencias de puertas especificadas en el paper.

def xor_into(qc: QuantumCircuit, a, b, target) -> None:
    """[PAPER Fig. 3] target ^= a XOR b; exactly two CX operations.

    It does not modify a or b. It is its own inverse, even if target was not zero.
    """
    qc.cx(a, target)
    qc.cx(b, target)


def xnor_into(qc: QuantumCircuit, a, b, target) -> None:
    """[PAPER Sec. 2.1] target ^= NOT(a XOR b): dos CX y una X."""
    xor_into(qc, a, b, target)
    qc.x(target)


def gt_bit_into(qc: QuantumCircuit, a, b, target) -> None:
    """[PAPER Sec. 2.1] target ^= NOT(a XOR b): two CXs and one X."""
    qc.x(b)
    qc.ccx(a, b, target)
    qc.x(b)


def lt_bit_into(qc: QuantumCircuit, a, b, target) -> None:
    """[PAPER Sec. 2.1] target ^= NOT(a) AND b; restores a."""
    qc.x(a)
    qc.ccx(a, b, target)
    qc.x(a)


def paper_one_bit_comparator(qc: QuantumCircuit, a, b, gt, lt, eq) -> None:
    """[PAPER P1] Text order: EQ first, then GT and LT.

    All qubits must be distinct. With outputs initially set to zero,
    it obtains all three indicators. It neither prepares nor measures the operands.
    """
    xnor_into(qc, a, b, eq)
    gt_bit_into(qc, a, b, gt)
    lt_bit_into(qc, a, b, lt)


# 2. Sintesis elegida por nosotros para las puertas de tres controles.

def c3x_via_clean_ancilla(qc: QuantumCircuit, c0, c1, c2, target, scratch) -> None:
    """[ADAPTATION A2] target ^= c0*c1*c2, with scratch initially set to zero.

    scratch <- c0*c1; target ^= scratch*c2; scratch <- 0.
    These are three exact CCX operations: they do not introduce relative phases. scratch is returned
    in a clean state and can be reused; the five qubits must be distinct.
    """
    qc.ccx(c0, c1, scratch)
    qc.ccx(scratch, c2, target)
    qc.ccx(c0, c1, scratch)


# 3. Generalizacion explicita del patron de la Fig. 3.

def compute_bit_indicators(qc: QuantumCircuit, a, b, different, equal) -> None:
    """[PAPER P2 + ADAPTATION A1] D_i = A_i XOR B_i; E_i = NOT(D_i).

    “different” and “equal” start at zero. Separate copies are maintained as shown in
    Fig. 3. Repeating this function reverses its effects, since the writes
    to each destination commute and a/b remain unchanged.
    """
    for i in range(len(a)):
        xor_into(qc, a[i], b[i], different[i])
        xnor_into(qc, a[i], b[i], equal[i])


def compute_equal_prefixes(qc: QuantumCircuit, equal, prefix) -> None:
    """[PAPER P2 + ADAPTATION A1] P_i = AND_{j>i}(E_j).

    “equal” and ‘prefix’ have length n. “prefix” starts at zero; the
    empty prefix of the most significant bit is one, explicitly set to X.
    """
    n = len(equal)
    qc.x(prefix[n - 1])
    for i in range(n - 1, 0, -1):
        qc.ccx(prefix[i], equal[i], prefix[i - 1])


def uncompute_equal_prefixes(qc: QuantumCircuit, equal, prefix) -> None:
    """[ADAPTATION A4] The inverse of the previous sequence, in reverse order."""
    for i in range(1, len(equal)):
        qc.ccx(prefix[i], equal[i], prefix[i - 1])
    qc.x(prefix[len(equal) - 1])


def write_relation_flags(qc: QuantumCircuit, a, b, different, equal, prefix,
                         gt, lt, eq, scratch) -> None:
    """[PAPER P2 + ADAPTATION A1/A2] Write EQ, GT, and LT using XOR.

    With the correct prefixes and flags, only the highest-order bit
    can trigger GT or LT. It does not modify data or flag registers.
    scratch starts and ends at zero.
    """
    qc.ccx(prefix[0], equal[0], eq)
    for i in range(len(a) - 1, -1, -1):
        c3x_via_clean_ancilla(qc, a[i], different[i], prefix[i], gt, scratch)
        c3x_via_clean_ancilla(qc, b[i], different[i], prefix[i], lt, scratch)


def multibit_comparator_adapted(qc: QuantumCircuit, a, b, gt, lt, eq,
                                different, equal, prefix, scratch) -> None:
    """[ADAPTATION A1-A4] Clean comparator n≥2 based on Fig. 3.

    The registers must be disjoint and of equal length. different,
    equal, prefix, and scratch must initially be set to zero. The elementary functions
    are separated so that the synthesis is verifiable.
    """
    n = len(a)
    if n < 2 or any(len(reg) != n for reg in (b, different, equal, prefix)):
        raise ValueError("Requires equal-width n>=2 operand and work registers")
    compute_bit_indicators(qc, a, b, different, equal)
    compute_equal_prefixes(qc, equal, prefix)
    write_relation_flags(qc, a, b, different, equal, prefix, gt, lt, eq, scratch)
    uncompute_equal_prefixes(qc, equal, prefix)
    compute_bit_indicators(qc, a, b, different, equal)  # Inversa, limpia D/E.


# 4. Integracion con qcmp: solo datos de la interfaz, no logica clasica A>B.

class CMP_Ganapathy_Toffoli_Adapted:
    meta = ComparatorMeta(
        key="cmp_ganapathy_toffoli_adapted",
        name="Ganapathy Toffoli: one-bit text and adapted clean prefix comparator (2025)",
        paper=(
            'V. Ganapathy et al., "Design and Analysis of Quantum Comparator '
            'Circuits for Constrained Optimization Problem", preprint (2025). '
            'doi:10.21203/rs.3.rs-7733902/v1.'
        ),
        kind="cmp", signed=False, outputs=("gt", "lt", "eq"),
        ancilla_mode="clean",
        notes=(
            "ADAPTED: n=1 follows Section 2.1; n>=2 generalizes the XOR/XNOR "
            "and higher-equality prefix structure read from Fig. 3. Own exact "
            "three-CCX synthesis of each C3X and extra work uncomputation. "
            "Zero-initialized work; inputs preserved; no garbage outputs. "
            "Default cost 50*n-3 for n>=2 includes our cleanup, NOT a paper "
            "resource claim. No QFT comparator or knapsack oracle included."
        ),
    )

    def build(self, n: int, measure: bool = True, **kwargs) -> BuildOutput:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be an integer")
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        if not isinstance(measure, bool):
            raise TypeError("measure must be a bool")
        if kwargs:
            raise TypeError(f"Unexpected build argument(s): {', '.join(sorted(kwargs))}")

        a, b = QuantumRegister(n, "a"), QuantumRegister(n, "b")
        out = QuantumRegister(3, "out")  # gt, lt, eq, en este orden.
        qc = QuantumCircuit(a, b, out, name=self.meta.key)
        gt, lt, eq = out
        regs = {"a": a, "b": b, "out": out,
                "out_qubits": dict(zip(self.meta.outputs, out)), "out_qubit": lt}
        restored_zero = ()

        if n == 1:
            paper_one_bit_comparator(qc, a[0], b[0], gt, lt, eq)
        else:
            different = QuantumRegister(n, "different")
            equal = QuantumRegister(n, "equal")
            prefix = QuantumRegister(n, "prefix")
            scratch = QuantumRegister(1, "scratch")
            qc.add_register(different, equal, prefix, scratch)
            regs.update(different=different, equal=equal, prefix=prefix, scratch=scratch)
            multibit_comparator_adapted(
                qc, a, b, gt, lt, eq, different, equal, prefix, scratch[0],
            )
            restored_zero = tuple([*different, *equal, *prefix, *scratch])

        decode_counts = None
        if measure:
            c_out = ClassicalRegister(3, "c_out")
            qc.add_register(c_out)
            regs["c_out"] = c_out
            for qubit, clbit in zip(out, c_out):
                qc.measure(qubit, clbit)
            indices = [qc.find_bit(bit).index for bit in c_out]

            def decode_counts(counts):
                if not counts:
                    raise ValueError("Cannot decode empty measurement counts")
                mc = marginal_counts(counts, indices=indices)
                bits = max(mc, key=mc.get)  # Qiskit imprime EQ LT GT.
                return {name: int(bits[-1-i]) for i, name in enumerate(self.meta.outputs)}

        return BuildOutput(
            circuit=qc, regs=regs, decode_counts=decode_counts,
            resources=ResourceLayout(
                operand_registers=("a", "b"), solution_qubits=tuple(out),
                restored_zero_qubits=restored_zero,
            ),
        )
