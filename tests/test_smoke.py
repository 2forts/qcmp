import qcmp.comparators  # registers
from qcmp.registry import get
from qcmp.runner import run

def test_smoke():
    rr = run('lt_orts_tand', n=2, A=0, B=1, shots=128, seed=1)
    assert 'lt' in rr.decoded


def test_xia_qip_truth_tables():
    for n in range(1, 5):
        for a in range(2 ** n):
            for b in range(2 ** n):
                rr = run('lt_xia_qip', n=n, A=a, B=b, shots=128, seed=1)
                assert rr.decoded == {'lt': int(a < b)}


def test_oliveira_qbsc_truth_tables():
    for n in range(2, 5):
        for a in range(2 ** n):
            for b in range(2 ** n):
                rr = run('cmp_oliveira_qbsc', n=n, A=a, B=b, shots=128, seed=1)
                assert rr.decoded == {
                    'gt': int(a > b),
                    'lt': int(a < b),
                    'eq': int(a == b),
                }


def test_al_rabadi_truth_tables():
    for n in range(1, 4):
        for a in range(2 ** n):
            for b in range(2 ** n):
                rr = run('cmp_al_rabadi', n=n, A=a, B=b, shots=128, seed=1)
def test_thapliyal_tree_truth_tables():
    for n in (2, 4):
        for a in range(2 ** n):
            for b in range(2 ** n):
                rr = run('cmp_thapliyal_tree', n=n, A=a, B=b, shots=128, seed=1)
                assert rr.decoded == {
                    'lt': int(a < b),
                    'eq': int(a == b),
                    'gt': int(a > b),
                }


def test_thapliyal_tree_builds_8_bit_tree():
    built = get('cmp_thapliyal_tree').build(8)
    assert built.circuit.num_qubits == 53
