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


def test_donaire_opt1_truth_table():
    for a in range(4):
        for b in range(4):
            rr = run('cmp_donaire_opt1', n=2, A=a, B=b, shots=128, seed=1)
            assert rr.decoded == {
                'lt': int(a < b),
                'eq': int(a == b),
                'gt': int(a > b),
            }


def test_donaire_opt1_t_count_matches_paper():
    qc = get('cmp_donaire_opt1').build(2).circuit
    ops = qc.count_ops()
    assert ops.get('t', 0) + ops.get('tdg', 0) == 24
