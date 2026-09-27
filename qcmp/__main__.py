from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def _json_object(value: str) -> dict:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


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
    equations_parser.add_argument(
        "--build-kwargs", type=_json_object, default=None,
        help='comparator build options as JSON, e.g. {"b_value": 7}',
    )

    benchmark_parser = sub.add_parser(
        "benchmark",
        help="benchmark registered comparators over a range of input widths",
    )
    benchmark_parser.add_argument(
        "--cmp",
        default="",
        help="comma-separated comparator keys; all registered comparators by default",
    )
    benchmark_parser.add_argument("--n-min", type=_positive_int, required=True)
    benchmark_parser.add_argument("--n-max", type=_positive_int, required=True)
    benchmark_parser.add_argument("--A", type=int, default=0)
    benchmark_parser.add_argument("--B", type=int, default=1)
    benchmark_parser.add_argument("--shots", type=_positive_int, default=1024)
    benchmark_parser.add_argument("--repeats", type=_positive_int, default=5)
    benchmark_parser.add_argument("--warmups", type=_non_negative_int, default=1)
    benchmark_parser.add_argument("--seed", type=int, default=1)
    benchmark_parser.add_argument(
        "--opt", type=int, default=1, choices=[0, 1, 2, 3]
    )
    benchmark_parser.add_argument("--basis", default="")
    benchmark_parser.add_argument(
        "--simulation-method",
        default="automatic",
        help="Aer simulation method",
    )
    benchmark_parser.add_argument(
        "--statevector-qubit-limit",
        type=_positive_int,
        default=24,
        help="dense-state width limit used by automatic simulation",
    )
    benchmark_parser.add_argument(
        "--max-qubits",
        type=_positive_int,
        default=None,
        help="skip benchmark points whose circuits exceed this width",
    )
    benchmark_parser.add_argument(
        "--stage",
        choices=["build", "transpile", "simulate", "total"],
        default="total",
        help="timing stage used for ranking and display",
    )
    benchmark_parser.add_argument(
        "--top",
        type=_positive_int,
        default=None,
        help="show only the fastest N eligible comparators",
    )
    benchmark_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="rank comparators that complete only part of the selected range",
    )
    benchmark_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json", "csv"],
        default="table",
    )
    benchmark_parser.add_argument(
        "--output",
        default="",
        help="write the table, JSON, or CSV report to this path",
    )
    benchmark_parser.add_argument(
        "--plot",
        nargs="?",
        const="qcmp-benchmark.png",
        default="",
        metavar="PATH",
        help="write a timing chart; defaults to qcmp-benchmark.png",
    )
    benchmark_parser.add_argument(
        "--log-y",
        action="store_true",
        help="use a logarithmic time axis in the chart",
    )
    benchmark_parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress messages",
    )

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
            build_kwargs=args.build_kwargs,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    if args.cmd == "benchmark":
        from qcmp.benchmark import (
            format_report,
            make_config,
            plot_report,
            run_benchmark,
        )

        config = make_config(
            n_min=args.n_min,
            n_max=args.n_max,
            comparators=_csv(args.cmp) or None,
            repeats=args.repeats,
            warmups=args.warmups,
            shots=args.shots,
            A=args.A,
            B=args.B,
            seed=args.seed,
            optimization_level=args.opt,
            basis_gates=_csv(args.basis) or None,
            simulation_method=args.simulation_method,
            statevector_qubit_limit=args.statevector_qubit_limit,
            max_qubits=args.max_qubits,
        )
        progress = None if args.quiet else lambda message: print(message, file=sys.stderr)
        report = run_benchmark(config, progress=progress)
        rendered = format_report(
            report,
            args.output_format,
            stage=args.stage,
            top=args.top,
            allow_partial=args.allow_partial,
        )

        if args.output and args.plot:
            if Path(args.output).resolve() == Path(args.plot).resolve():
                parser.error("--output and --plot must use different paths")

        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            print(f"Report written to {destination}", file=sys.stderr)
        else:
            print(rendered, end="" if rendered.endswith("\n") else "\n")

        if args.plot:
            chart = plot_report(
                report,
                args.plot,
                stage=args.stage,
                top=args.top,
                allow_partial=args.allow_partial,
                log_y=args.log_y,
            )
            print(f"Chart written to {chart}", file=sys.stderr)


if __name__ == "__main__":
    main()
