"""Comparator ADAPTED from Kumar et al. (2024); not a verbatim reproduction.

Source: A. S. Kumar et al., "A Comparator Tree Design with Optimized Quantum
Cost Using Reversible Logic Techniques," IACIS 2024.
DOI: 10.1109/IACIS61494.2024.10721638. Provided PDF: 22-Kumar.pdf.

SOURCE AND ASSUMPTIONS (search for [PAPER] and [ADAPTATION] in the functions)
----------------------------------------------------------------------
[PAPER] Fig. 1: Feynman(A,B) = (A, A XOR B).
[PAPER] Fig. 2: TR(A,B,C) = (A, A XOR B, (A AND NOT B) XOR C).
[PAPER] Fig. 3: UPG(A,B,C) = (A, (A OR B) XOR C, (A AND B) XOR C).
[PAPER] Fig. 4: three TRs, two Feynmans, one UPG, one NOT, and four constants.
[PAPER] Figs. 6–7: tree structure for comparing longer words.

[ADAPTATION A1] The X/CX/CCX decompositions are our own. We do not claim
that they are optimal or that they are the ones used for the costs in the paper.
[ADAPTATION A2] Our literal circuit in Figs. 4–5 fails in 11 out of 16 cases
involving two bits. For example, X=1, Y=0 yields GT=LT=EQ=1. It is preserved separately
in paper_literal_2bit_circuit(), ONLY for auditing purposes, never as a plugin.
[ADAPTATION A3] TR3 is rewired to give priority to the most significant bit:
TR3(L_low, D_high, L_high). D_high is copied, not D_low, for the UPG.
[ADAPTATION A4] The node outputs NOT_LT and NOT_GT, not direct GT/LT flags.
These two signals encode a relationship and are used as X/Y inputs for the
next level. The output circuit in Fig. 5 is replaced by an
explicit conversion to GT, LT, EQ. This is not hidden in the decoder.
[ADAPTATION A5] Balanced binary tree, unsigned little-endian inputs.
The case n=1 and widths other than 2, 8, and 64 are our own extensions.
[ADAPTATION A6] To preserve the framework’s a/b registers, we work
with computational copies prepared using 2*n CX. This preserves
global coherence, but does NOT clone arbitrary quantum states.
[ADAPTATION A7] Garbage outputs are retained. With `measure=False`, the circuit
is unitary, without reset, but it is NOT a comparison oracle with
clean operation. Its correlated auxiliary states must not be discarded or reset if
coherence needs to be preserved. All auxiliary inputs start
from zero; the one constants are explicitly prepared with X.

RESOURCES FOR THIS IMPLEMENTATION (not the published numbers)
---------------------------------------------------------
With the qcmp profile CX=X=1, CCX=5, and 7 T per CCX:
  TR: 1 CX + 1 CCX -> cost 6; UPG: 2 CX + 1 CCX -> cost 7.
  n>=2: CX=9*n-5, CCX=4*(n-1), X=2*n+1; final measurements=3.
  quantum_cost = 31*n - 24, t_count = 28*(n - 1).
  qubits = 8*n - 3, auxiliary qubits = 6*n - 3, waste = 6*n - 6, results = 3.
  n=1: 4 CX, 2 CCX, 1 X; cost 15, T-count 14, 5 qubits, waste 0.
The T-count comes from the CCX decomposition profile, not from explicit T operations.
The costs include input copies and constant preparation.
The costs of 20/128/1136 reported in the paper are not reproduced.

VALIDATION OF THIS SUBMISSION
-------------------------
Python 3.12.13, Qiskit 2.5.2, Aer 0.17.2: 34 new tests and 53 existing tests
passed. Includes the complete gate matrices, the literal diagnosis
11/16, 5,460 pairs of 1–6 bits via evaluation of the
X/CX/CCX sequence, 1,364 1-5-bit pairs in Aer MPS, complex
1-2-bit overlaps including noise, and six boundary cases per width for
8 and 64 bits in Aer MPS. Registers, the runner, metrics,
and equations were also verified. Not validated on quantum hardware.
The test environment does not expose /proc/meminfo: ONLY on the local launcher was
the psutil RAM query replaced with an estimated value of 512 MiB. Neither
the simulator nor its results were modified; this is not part of the module.
"""

from __future__ import annotations

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from qcmp.comparators.base import BuildOutput, ComparatorMeta, ResourceLayout


# 1. PUERTAS: ecuaciones del paper; descomposiciones deducidas por nosotros.

def feynman(qc: QuantumCircuit, a, b) -> None:
    """[PAPER Fig. 1] (A,B) -> (A,A XOR B): exactamente una CX."""
    qc.cx(a, b)


def tr(qc: QuantumCircuit, a, b, c) -> None:
    """[PAPER Fig. 2] (A, B, C) -> (A, A XOR B, A*NOT(B) XOR C).

    [ADAPTATION A1] First, b = A XOR B. Then, a*b = A*NOT(B),
    so a single CCX with that modified b is sufficient. X is not needed.
    This sequence, not shown in the paper, does not introduce relative phases.
    """
    qc.cx(a, b)
    qc.ccx(a, b, c)


def upg(qc: QuantumCircuit, a, b, c) -> None:
    """[PAPER Fig. 3] (A, B, C) -> (A, (A OR B) XOR C, A*B XOR C).

    [ADAPTATION A1] First, c = C XOR A*B; then, b = A XOR B XOR c.
    We use A OR B = A XOR B XOR A*B. The ‘+’ in Fig. 3 is OR.
    """
    qc.ccx(a, b, c)
    qc.cx(a, b)
    qc.cx(c, b)


def not_gate(qc: QuantumCircuit, bit) -> None:
    """[PAPER Fig. 4] Reversible Classic NOT: an X-shaped gate."""
    qc.x(bit)


# 2. AUDITORIA: lectura literal conservada, sin 'arreglar' sus resultados.

def paper_literal_2bit_circuit():
    """Returns (circuit, registers) from our literal reading of Fig. 4-5.

    This is a reversible nine-qubit circuit, but its output labels
    DO NOT satisfy the truth table of a comparator. It is not recorded in qcmp.
    Initialize x/y as two-bit words; set the rest to zero initially.
    It does not include measurements. regs[‘outputs’] is ordered as (gt, lt, eq).
    """
    x, y = QuantumRegister(2, "x"), QuantumRegister(2, "y")
    aux = QuantumRegister(4, "aux")
    extra = QuantumRegister(1, "extra")
    qc = QuantumCircuit(x, y, aux, extra, name="kumar_paper_literal_diagnostic")
    r_hi, r_lo, d_copy, one = aux

    tr(qc, y[1], x[1], r_hi)
    tr(qc, y[0], x[0], r_lo)
    feynman(qc, x[0], d_copy)  # Fig. 4 copia D_low.
    tr(qc, r_hi, x[0], r_lo)  # Fig. 4: TR3(L_high,D_low,L_low).
    not_gate(qc, one)
    upg(qc, x[1], d_copy, one)
    feynman(qc, r_lo, d_copy)
    not_gate(qc, r_lo)
    fig4_gt, fig4_lt = r_lo, d_copy

    # Fig. 5: primera Feynman(fig4_lt,1), segunda(fig4_gt,fig4_lt).
    not_gate(qc, extra[0])
    feynman(qc, fig4_lt, extra[0])
    feynman(qc, fig4_gt, fig4_lt)
    return qc, {"x": x, "y": y, "outputs": (fig4_lt, extra[0], fig4_gt)}


def audit_paper_2bit() -> list[dict]:
    """Independent Boolean evaluation: returns the 11 counterexamples.

    It is not a substitute for the quantum circuit, nor is it used to obtain the
    results from the plugin. It serves to make the discrepancy reproducible.
    """
    failures = []
    for x in range(4):
        for y in range(4):
            xh, xl, yh, yl = x >> 1, x & 1, y >> 1, y & 1
            dh, dl = xh ^ yh, xl ^ yl
            lh, ll = yh & (1 ^ xh), yl & (1 ^ xl)
            t = ll ^ (lh & (1 ^ dl))
            equality = 1 ^ (dh | dl)
            u, v = 1 ^ t, t ^ equality
            actual = {"gt": u ^ v, "lt": 1 ^ v, "eq": u}
            expected = {"gt": int(x > y), "lt": int(x < y), "eq": int(x == y)}
            if actual != expected:
                failures.append({"x": x, "y": y, "actual": actual, "expected": expected})
    return failures


# 3. NODO ADAPTADO: mismo repertorio de puertas, cableado documentado.

def comparator_node_adapted(qc: QuantumCircuit, x_hi, y_hi, x_lo, y_lo, aux):
    """Compare (X_hi X_lo) with (Y_hi Y_lo), giving priority to hi over lo.

    aux consists of four DIFFERENT qubits, initially set to zero. Modifies the
    four data qubits and leaves six dummy outputs. Returns
    (NOT_LT, NOT_GT), not (GT, LT). The caller retains the data if needed.
    """
    if len(aux) != 4:
        raise ValueError("Each comparator node requires four zero ancillas")
    r_hi, r_lo, d_hi_copy, one = aux

    # [PAPER] Dos comparadores de un bit: D_i=Y_i XOR X_i, L_i=Y_i*NOT(X_i).
    tr(qc, y_hi, x_hi, r_hi)
    tr(qc, y_lo, x_lo, r_lo)

    # [ADAPTACION A3] Copiar D_high ANTES de que TR3 lo modifique.
    feynman(qc, x_hi, d_hi_copy)
    # L = L_high XOR (NOT(D_high) AND L_low): gana el bit alto si difiere.
    tr(qc, r_lo, x_hi, r_hi)

    # [PAPER + A3] UPG con tercera entrada 1 obtiene E=NOT(D_low OR D_high).
    not_gate(qc, one)  # Preparacion contada; no es un '1' gratuito.
    upg(qc, x_lo, d_hi_copy, one)

    # [ADAPTACION A4] L XOR E = NOT_GT; NOT(L) = NOT_LT.
    feynman(qc, r_hi, d_hi_copy)
    not_gate(qc, r_hi)
    return r_hi, d_hi_copy


def comparator_tree_adapted(qc: QuantumCircuit, x, y, scratch):
    """[ADAPTATION A5] Balanced tree, little-endian records, n ≥ 2.

    The leaves are pairs (X_i, Y_i). Each node converts the relationship between its
    children into a pair: smaller->(0,1), larger->(1,0), equal->(1,1).
    Comparing these pairs as bits preserves lexicographic order.
    Splitting into halves also maintains the order of odd widths..
    """
    n = len(x)
    if n < 2 or len(y) != n or len(scratch) != 4 * (n - 1):
        raise ValueError("Tree requires equal n>=2 operands and 4*(n-1) scratch qubits")
    pairs = [(x[i], y[i]) for i in range(n - 1, -1, -1)]  # MSB primero.
    cursor = 0

    def reduce_tree(items):
        nonlocal cursor
        if len(items) == 1:
            return items[0]
        middle = len(items) // 2
        high = reduce_tree(items[:middle])
        low = reduce_tree(items[middle:])
        node_aux = list(scratch[cursor:cursor + 4])
        cursor += 4
        return comparator_node_adapted(qc, *high, *low, node_aux)

    result = reduce_tree(pairs)
    assert cursor == len(scratch)
    return result


def output_flags_adapted(qc: QuantumCircuit, not_lt, not_gt, eq_zero):
    """[ADAPTATION A4] EXPLICIT replacement of the wiring in Fig. 5.

    Swap the two gates; EQ = 1 XOR GT XOR LT (they are mutually exclusive).
    eq_zero must start at zero. Return the qubits (GT, LT, EQ).
    """
    not_gate(qc, not_lt)
    not_gate(qc, not_gt)
    not_gate(qc, eq_zero)
    feynman(qc, not_lt, eq_zero)
    feynman(qc, not_gt, eq_zero)
    return not_gt, not_lt, eq_zero


def single_bit_extension(qc: QuantumCircuit, a, b, a_copy, b_copy, eq_zero):
    """[ADAPTATION A5] n=1, not described in the paper.

    a_copy=A and b_copy=B have been prepared using CX. A XOR AB = A*NOT(B)
    and B XOR AB = NOT(A)*B. The original inputs remain intact.
    """
    qc.ccx(a, b, a_copy)
    qc.ccx(a, b, b_copy)
    not_gate(qc, eq_zero)
    feynman(qc, a_copy, eq_zero)
    feynman(qc, b_copy, eq_zero)
    return a_copy, b_copy, eq_zero


# 4. ADAPTADOR qcmp: registros, medidas, decodificacion y recursos reales.

class CMP_Kumar_Tree_Adapted:
    meta = ComparatorMeta(
        key="cmp_kumar_tree_adapted",
        name="Kumar tree: explicitly rewired adaptation (2024)",
        paper=(
            'A. S. Kumar et al., "A Comparator Tree Design with Optimized '
            'Quantum Cost Using Reversible Logic Techniques", IACIS (2024). '
            'doi:10.1109/IACIS61494.2024.10721638.'
        ),
        kind="cmp", signed=False, outputs=("gt", "lt", "eq"),
        ancilla_mode="dirty",  # Convencion qcmp: deja basura, NO acepta ancillas desconocidas.
        notes=(
            "ADAPTED, not a literal reproduction. Gate equations from Figs. 1-3; "
            "own X/CX/CCX synthesis, rewired TR3 and output stage, explicit "
            "NOT_LT/NOT_GT tree encoding, and copied operands. Zero-initialized "
            "ancillas; garbage retained. Literal Figs. 4-5 reading fails 11/16 "
            "two-bit inputs. Our default cost is 31*n-24 for n>=2, NOT the "
            "published costs. n=1 is an extension. See module docstring."
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
        x, y = QuantumRegister(n, "work_x"), QuantumRegister(n, "work_y")
        eq = QuantumRegister(1, "eq")
        qc = QuantumCircuit(a, b, x, y, eq, name=self.meta.key)
        regs = {"a": a, "b": b, "work_x": x, "work_y": y, "eq": eq}
        for i in range(n):
            feynman(qc, a[i], x[i])
            feynman(qc, b[i], y[i])

        if n == 1:
            outputs = single_bit_extension(qc, a[0], b[0], x[0], y[0], eq[0])
        else:
            scratch = QuantumRegister(4 * (n - 1), "scratch")
            qc.add_register(scratch)
            regs["scratch"] = scratch
            not_lt, not_gt = comparator_tree_adapted(qc, x, y, scratch)
            outputs = output_flags_adapted(qc, not_lt, not_gt, eq[0])

        regs["out_qubits"] = dict(zip(self.meta.outputs, outputs))
        regs["out_qubit"] = outputs[1]  # Convencion: resultado lt.
        decode_counts = None
        if measure:
            c_out = ClassicalRegister(3, "c_out")
            qc.add_register(c_out)
            regs["c_out"] = c_out
            for qubit, clbit in zip(outputs, c_out):
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
                operand_registers=("a", "b"), solution_qubits=tuple(outputs),
                # Ningun auxiliar de trabajo se declara limpio indebidamente.
                restored_zero_qubits=(),
            ),
        )


def simulate_mps(n: int, A: int, B: int, shots: int = 32, seed: int = 1) -> dict[str, int]:
    """Usage: Ideal MPS simulation for integer inputs, including 64-bit.

    It does not modify qcmp.runner or use a classical A<B calculation as the result.
    The outputs come from simulating and measuring the circuit. It does not require
    plugin registration. The advantage of MPS depends on state interleaving.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qcmp.encoding import load_int

    built = CMP_Kumar_Tree_Adapted().build(n)
    prefix = QuantumCircuit(*built.circuit.qregs, *built.circuit.cregs)
    load_int(prefix, built.regs["a"], A)
    load_int(prefix, built.regs["b"], B)
    prepared = prefix.compose(built.circuit)
    backend = AerSimulator(method="matrix_product_state", max_parallel_threads=1)
    # Aer anuncia por defecto 63 qubits para MPS. Ajustar el ancho del target
    # es necesario para transpilar 509 qubits; no reserva un vector denso.
    if prepared.num_qubits > backend.num_qubits:
        backend.set_max_qubits(prepared.num_qubits)
    compiled = transpile(prepared, backend, optimization_level=0, seed_transpiler=seed)
    counts = backend.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
    return built.decode_counts(counts)
