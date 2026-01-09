from __future__ import annotations
from typing import Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.result import marginal_counts
from qcmp.comparators.base import ComparatorMeta, BuildOutput

# Temporary logical-AND gate as drawn in Fig. 3 of Orts et al. (2021).
# Compute uses 4 T/Tdg gates; uncompute uses measurement + classically-controlled Z.

def prepare_t_state(qc: QuantumCircuit, q) -> None:
    # |Ti> = (|0> + e^{i pi/4}|1>)/sqrt(2) = T|+>
    qc.h(q)
    qc.t(q)

def temp_and_compute(qc: QuantumCircuit, x, y, a) -> None:
    qc.cx(x, a)
    qc.cx(y, a)
    qc.cx(a, x)
    qc.cx(a, y)
    qc.tdg(x)
    qc.tdg(y)
    qc.t(a)
    qc.cx(a, x)
    qc.cx(a, y)
    qc.h(a)
    qc.s(a)

def temp_and_uncompute(qc: QuantumCircuit, y, a, cbit) -> None:
    qc.h(a)
    qc.measure(a, cbit)
    with qc.if_test((cbit, 1)):
        qc.z(y)

class LT_Orts_TemporaryAND:
    meta = ComparatorMeta(
        key='lt_orts_tand',
        name='Half comparator using temporary logical-AND (Orts et al., 2021)',
        paper='F. Orts et al., "Optimal fault-tolerant quantum comparators for image binarization", The Journal of Supercomputing (2021).',
        kind='lt',
        signed=False,
        outputs=('lt',),
        ancilla_mode='clean',
        notes='Implements the first proposed comparator (Fig. 4) with temporary logical-AND (Fig. 3).'
    )

    def build(self, n: int, sense: str='lt', **kwargs) -> BuildOutput:
        if n < 1:
            raise ValueError('n must be >=1')
        if sense not in ('lt','gt'):
            raise ValueError("sense must be 'lt' or 'gt'")

        a = QuantumRegister(n,'a')
        b = QuantumRegister(n,'b')
        A = QuantumRegister(n,'A')  # A[n-1] will hold the raw result bit
        c_fix = ClassicalRegister(max(n-1,1),'c_fix')
        c_out = ClassicalRegister(1,'c_out')
        qc = QuantumCircuit(a,b,A,c_fix,c_out,name=self.meta.key)

        # Step 1: X on all a_i
        for i in range(n):
            qc.x(a[i])

        # Step 2: temp-AND for a0 b0
        prepare_t_state(qc, A[0])
        temp_and_compute(qc, a[0], b[0], A[0])
        
        # Step 3: forward loop
        for i in range(1,n):
            qc.cx(A[i-1], a[i])
            qc.cx(A[i-1], b[i])
            prepare_t_state(qc, A[i])
            temp_and_compute(qc, a[i], b[i], A[i])
            qc.cx(A[i-1], A[i])

        # Fig. 4 label: output is |0> if a<=b, |1> otherwise (raw_gt).
        if sense=='lt':
            qc.x(A[n-1])  # lt = NOT(raw_gt)

        # Step 4: uncompute garbage
        if n>=2:
            qc.cx(A[n-2], a[n-1])
            qc.cx(A[n-2], b[n-1])
            for i in range(n-2,0,-1):
                qc.cx(A[i-1], A[i])
                temp_and_uncompute(qc, b[i], A[i], c_fix[i-1])
                qc.cx(A[i-1], a[i])
                qc.cx(A[i-1], b[i])
            temp_and_uncompute(qc, b[0], A[0], c_fix[0])

        # Step 5: undo X on a_i
        for i in range(n):
            qc.x(a[i])

        qc.measure(A[n-1], c_out[0])

        def decode_counts(counts):
            # c_out debe existir como ClassicalRegister(1, "c_out")
            # y qc es el QuantumCircuit que estás construyendo.
            idx = qc.clbits.index(c_out[0])          # índice real del bit clásico de salida
            mc = marginal_counts(counts, indices=[idx])  # nos quedamos SOLO con ese bit

            bit = max(mc, key=mc.get)               # "0" o "1"
            return {"lt": 1-int(bit)}

        regs = {'a':a,'b':b,'A':A,'out_qubit':A[n-1],'c_out':c_out}
        return BuildOutput(circuit=qc, regs=regs, decode_counts=decode_counts)

        print("CLASSICAL BITS ORDER:")
        for i, cb in enumerate(qc.clbits):
            print(i, cb)