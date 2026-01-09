from __future__ import annotations
import argparse, json
import qcmp.comparators  # registers
from qcmp.registry import list_all
from qcmp.runner import run
from qcmp.metrics import available as metrics_available

def main():
    p = argparse.ArgumentParser(prog='qcmp')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    sub.add_parser('metrics')
    pr = sub.add_parser('run')
    pr.add_argument('--cmp', required=True)
    pr.add_argument('--n', type=int, required=True)
    pr.add_argument('--A', type=int, required=True)
    pr.add_argument('--B', type=int, required=True)
    pr.add_argument('--shots', type=int, default=1024)
    pr.add_argument('--opt', type=int, default=1, choices=[0,1,2,3])
    pr.add_argument('--basis', default='')
    pr.add_argument('--metrics', default='')
    pr.add_argument('--metrics-all', action='store_true')
    args = p.parse_args()

    if args.cmd=='list':
        for k in list_all(): print(k)
        return
    if args.cmd=='metrics':
        for k in metrics_available(): print(k)
        return
    if args.cmd=='run':
        basis = [x.strip() for x in args.basis.split(',') if x.strip()] or None
        compute_metrics = args.metrics_all or bool(args.metrics.strip())
        metric_keys = None if args.metrics_all else ([x.strip() for x in args.metrics.split(',') if x.strip()] or None)
        rr = run(args.cmp, args.n, args.A, args.B, shots=args.shots, optimization_level=args.opt,
                 basis_gates=basis, compute_metrics=compute_metrics, metric_keys=metric_keys)
        print(json.dumps({'cmp':rr.key,'n':rr.n,'A':rr.A,'B':rr.B,'decoded':rr.decoded,'counts':rr.counts,'metrics':rr.metrics}, indent=2, sort_keys=True))

if __name__=='__main__':
    main()
