from __future__ import annotations

import argparse
import json

import qcmp.comparators  # registers comparator plugins
from qcmp.metrics import (
    DEFAULT_GATE_PROFILES,
    available as metrics_available,
    generate_equations,
    load_gate_profiles,
)
from qcmp.registry import get, list_all
from qcmp.runner import run


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="qcmp")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("metrics")
    sub.add_parser("gate-profiles")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cmp", required=True)
    run_parser.add_argument("--n", type=int, required=True)
    run_parser.add_argument("--A", type=int, required=True)
    run_parser.add_argument("--B", type=int, required=True)
    run_parser.add_argument("--shots", type=int, default=1024)
    run_parser.add_argument("--opt", type=int, default=1, choices=[0, 1, 2, 3])
    run_parser.add_argument("--basis", default="")
    run_parser.add_argument("--metrics", default="")
    run_parser.add_argument("--metrics-all", action="store_true")
    run_parser.add_argument("--gate-profiles", default="")

    equations_parser = sub.add_parser("equations")
    equations_parser.add_argument("--cmp", required=True)
    equations_parser.add_argument("--metrics", default="")
    equations_parser.add_argument("--sample-max", type=int, default=12)
    equations_parser.add_argument("--max-degree", type=int, default=4)
    equations_parser.add_argument("--gate-profiles", default="")

    args = parser.parse_args()

    if args.cmd == "list":
        for key in list_all():
            print(key)
        return

    if args.cmd == "metrics":
        for key in metrics_available():
            print(key)
        return

    if args.cmd == "gate-profiles":
        print(json.dumps(DEFAULT_GATE_PROFILES.to_dict(), indent=2, sort_keys=True))
        return

    if args.cmd == "run":
        basis = _csv(args.basis) or None
        compute_metrics = args.metrics_all or bool(args.metrics.strip())
        metric_keys = None if args.metrics_all else (_csv(args.metrics) or None)
        profiles = load_gate_profiles(args.gate_profiles or None)
        result = run(
            args.cmp,
            args.n,
            args.A,
            args.B,
            shots=args.shots,
            optimization_level=args.opt,
            basis_gates=basis,
            compute_metrics=compute_metrics,
            metric_keys=metric_keys,
            gate_profiles=profiles,
        )
        print(
            json.dumps(
                {
                    "cmp": result.key,
                    "n": result.n,
                    "A": result.A,
                    "B": result.B,
                    "decoded": result.decoded,
                    "counts": result.counts,
                    "metrics": result.metrics,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.cmd == "equations":
        profiles = load_gate_profiles(args.gate_profiles or None)
        report = generate_equations(
            get(args.cmp),
            metrics=_csv(args.metrics) or None,
            sample_max=args.sample_max,
            max_degree=args.max_degree,
            gate_profiles=profiles,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()