from __future__ import annotations
from typing import Any, Optional
from qiskit import QuantumCircuit

def available() -> list[str]:
    return ['num_qubits','depth','size','op_counts','weighted_cost']

def compute(c: QuantumCircuit, keys: Optional[list[str]]=None) -> dict[str,Any]:
    ks = available() if not keys else keys
    out: dict[str,Any] = {}
    for k in ks:
        if k=='num_qubits': out[k]=c.num_qubits
        elif k=='depth': out[k]=c.depth()
        elif k=='size': out[k]=c.size()
        elif k=='op_counts': out[k]=dict(c.count_ops())
        elif k=='weighted_cost':
            weights={'x':1,'h':1,'s':1,'sdg':1,'t':1,'tdg':1,'cx':2,'cz':2,'ccx':5,'measure':0,'reset':0}
            total=0
            for op,cnt in c.count_ops().items():
                total += weights.get(op,1)*int(cnt)
            out[k]=total
        else:
            raise KeyError(k)
    return out
