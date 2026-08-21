import qcmp.comparators  # registers
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


def test_barui_threshold_truth_tables():
    for n in range(1, 5):
        for intensity in range(2 ** n):
            for threshold in range(2 ** n):
                rr = run(
                    'gt_barui_threshold',
                    n=n,
                    A=intensity,
                    B=threshold,
                    shots=128,
                    seed=1,
                )
                assert rr.decoded == {'gt': int(intensity > threshold)}
