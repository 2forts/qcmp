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
def test_morrison_rc_tree_truth_tables():
    for n in range(1, 5):
        for a in range(2 ** n):
            for b in range(2 ** n):
                rr = run('cmp_morrison_rc_tree', n=n, A=a, B=b, shots=128, seed=1)
                assert rr.decoded == {
                    'lt': int(a < b),
                    'eq': int(a == b),
                    'gt': int(a > b),
                }


def test_thapliyal_tree_builds_8_bit_tree():
    built = get('cmp_thapliyal_tree').build(8)
    assert built.circuit.num_qubits == 53
def test_maity_qcc_truth_table():
    for a in range(4):
        for b in range(4):
            rr = run('cmp_maity_qcc', n=2, A=a, B=b, shots=128, seed=1)
            assert rr.decoded == {
                'lt': int(a < b),
                'eq': int(a == b),
                'gt': int(a > b),
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
def test_caraiman_threshold_oracle_truth_tables():
    for n in range(2, 5):
        for threshold in range(2 ** n):
            for color in range(2 ** n):
                rr = run(
                    'ge_caraiman_threshold_oracle',
                    n=n,
                    A=threshold,
                    B=color,
                    shots=128,
                    seed=1,
                )
                assert rr.decoded == {'ge': int(color >= threshold)}
def test_du_nonzero_checker_truth_tables():
    for n in range(1, 5):
        for pixel in range(2 ** n):
            rr = run('gt_du_nonzero', n=n, A=pixel, B=0, shots=128, seed=1)
            assert rr.decoded == {'gt': int(pixel > 0)}
