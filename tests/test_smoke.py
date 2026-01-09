import qcmp.comparators  # registers
from qcmp.runner import run

def test_smoke():
    rr = run('lt_orts_tand', n=2, A=0, B=1, shots=128, seed=1)
    assert 'lt' in rr.decoded
