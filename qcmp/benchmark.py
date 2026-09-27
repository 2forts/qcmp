"""Runtime benchmarking for registered qcmp comparators."""

from __future__ import annotations

import csv
import io
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from qiskit import QuantumCircuit, transpile

import qcmp.comparators  # register bundled comparator plugins
from qcmp.comparators.base import BuildOutput
from qcmp.encoding import load_int
from qcmp.registry import get, list_all
from qcmp.runner import _simulation_backend


TIMING_STAGES = ("build", "transpile", "simulate", "total")
_OPERANDS = ("a", "b")
_NS_PER_MS = 1_000_000


class UnsupportedBenchmarkPoint(Exception):
    """Raised when a comparator does not support one requested configuration."""


@dataclass(frozen=True)
class TimingStats:
    """Summary of repeated wall-clock measurements in nanoseconds."""

    samples_ns: tuple[int, ...]
    minimum_ns: int
    first_quartile_ns: float
    median_ns: float
    third_quartile_ns: float
    maximum_ns: int

    @classmethod
    def from_samples(cls, samples_ns: Iterable[int]) -> "TimingStats":
        samples = tuple(int(value) for value in samples_ns)
        if not samples:
            raise ValueError("at least one timing sample is required")
        if any(value < 0 for value in samples):
            raise ValueError("timing samples must be non-negative")

        ordered = tuple(sorted(samples))
        if len(ordered) == 1:
            first_quartile = third_quartile = float(ordered[0])
        else:
            first_quartile, _, third_quartile = statistics.quantiles(
                ordered,
                n=4,
                method="inclusive",
            )
        return cls(
            samples_ns=samples,
            minimum_ns=ordered[0],
            first_quartile_ns=first_quartile,
            median_ns=statistics.median(ordered),
            third_quartile_ns=third_quartile,
            maximum_ns=ordered[-1],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "samples_ms": [_milliseconds(value) for value in self.samples_ns],
            "minimum_ms": _milliseconds(self.minimum_ns),
            "first_quartile_ms": _milliseconds(self.first_quartile_ns),
            "median_ms": _milliseconds(self.median_ns),
            "third_quartile_ms": _milliseconds(self.third_quartile_ns),
            "maximum_ms": _milliseconds(self.maximum_ns),
        }


@dataclass(frozen=True)
class BenchmarkPoint:
    comparator: str
    n: int
    status: str
    timings: dict[str, TimingStats] | None = None
    backend_method: str | None = None
    num_qubits: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "skipped", "error"}:
            raise ValueError(f"invalid benchmark status: {self.status!r}")
        if self.status == "ok" and self.timings is None:
            raise ValueError("successful benchmark points require timings")
        if self.status != "ok" and self.timings is not None:
            raise ValueError("failed benchmark points cannot contain timings")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "comparator": self.comparator,
            "n": self.n,
            "status": self.status,
        }
        if self.backend_method is not None:
            result["backend_method"] = self.backend_method
        if self.num_qubits is not None:
            result["num_qubits"] = self.num_qubits
        if self.timings is not None:
            result["timings"] = {
                stage: self.timings[stage].to_dict() for stage in TIMING_STAGES
            }
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class BenchmarkConfig:
    comparators: tuple[str, ...]
    n_min: int
    n_max: int
    repeats: int = 5
    warmups: int = 1
    shots: int = 1024
    a_value: int = 0
    b_value: int = 1
    seed: int = 1
    optimization_level: int = 1
    basis_gates: tuple[str, ...] | None = None
    simulation_method: str = "automatic"
    statevector_qubit_limit: int = 24
    max_qubits: int | None = None

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(range(self.n_min, self.n_max + 1))

    def to_dict(self) -> dict[str, object]:
        return {
            "comparators": list(self.comparators),
            "n_min": self.n_min,
            "n_max": self.n_max,
            "repeats": self.repeats,
            "warmups": self.warmups,
            "shots": self.shots,
            "A": self.a_value,
            "B": self.b_value,
            "seed": self.seed,
            "optimization_level": self.optimization_level,
            "basis_gates": (
                list(self.basis_gates) if self.basis_gates is not None else None
            ),
            "simulation_method": self.simulation_method,
            "statevector_qubit_limit": self.statevector_qubit_limit,
            "max_qubits": self.max_qubits,
        }


@dataclass(frozen=True)
class ComparatorRanking:
    rank: int
    comparator: str
    completed_sizes: int
    requested_sizes: int
    mean_median_ns: float
    minimum_ns: int
    maximum_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "comparator": self.comparator,
            "coverage": {
                "completed": self.completed_sizes,
                "requested": self.requested_sizes,
            },
            "mean_median_ms": _milliseconds(self.mean_median_ns),
            "minimum_ms": _milliseconds(self.minimum_ns),
            "maximum_ms": _milliseconds(self.maximum_ns),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    config: BenchmarkConfig
    points: tuple[BenchmarkPoint, ...]

    def point(self, comparator: str, n: int) -> BenchmarkPoint:
        for point in self.points:
            if point.comparator == comparator and point.n == n:
                return point
        raise KeyError(f"no benchmark point for {comparator!r} at n={n}")

    def rankings(
        self,
        stage: str = "total",
        *,
        allow_partial: bool = False,
    ) -> list[ComparatorRanking]:
        _validate_stage(stage)
        requested_sizes = len(self.config.widths)
        candidates: list[tuple[str, list[BenchmarkPoint]]] = []
        for comparator in self.config.comparators:
            successful = [
                point
                for point in self.points
                if point.comparator == comparator and point.status == "ok"
            ]
            if not successful:
                continue
            if not allow_partial and len(successful) != requested_sizes:
                continue
            candidates.append((comparator, successful))

        scored: list[tuple[float, str, list[BenchmarkPoint]]] = []
        for comparator, successful in candidates:
            medians = [point.timings[stage].median_ns for point in successful]
            scored.append((statistics.fmean(medians), comparator, successful))
        scored.sort(key=lambda item: (item[0], item[1]))

        rankings: list[ComparatorRanking] = []
        for rank, (score, comparator, successful) in enumerate(scored, start=1):
            rankings.append(
                ComparatorRanking(
                    rank=rank,
                    comparator=comparator,
                    completed_sizes=len(successful),
                    requested_sizes=requested_sizes,
                    mean_median_ns=score,
                    minimum_ns=min(
                        point.timings[stage].minimum_ns for point in successful
                    ),
                    maximum_ns=max(
                        point.timings[stage].maximum_ns for point in successful
                    ),
                )
            )
        return rankings

    def to_dict(
        self,
        *,
        stage: str = "total",
        top: int | None = None,
        allow_partial: bool = False,
    ) -> dict[str, object]:
        rankings = _limited_rankings(
            self,
            stage=stage,
            top=top,
            allow_partial=allow_partial,
        )
        selected = _selected_comparators(self, rankings, top)
        return {
            "schema_version": 1,
            "stage": stage,
            "top": top,
            "allow_partial": allow_partial,
            "config": self.config.to_dict(),
            "ranking": [ranking.to_dict() for ranking in rankings],
            "results": [
                point.to_dict()
                for point in self.points
                if point.comparator in selected
            ],
        }


@dataclass(frozen=True)
class _IterationResult:
    timings_ns: dict[str, int]
    backend_method: str
    num_qubits: int


def _milliseconds(value_ns: int | float) -> float:
    return round(float(value_ns) / _NS_PER_MS, 6)


def _validate_stage(stage: str) -> None:
    if stage not in TIMING_STAGES:
        accepted = ", ".join(TIMING_STAGES)
        raise ValueError(f"stage must be one of: {accepted}")


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.comparators:
        raise ValueError("at least one comparator must be selected")
    if config.n_min < 1:
        raise ValueError("n_min must be >= 1")
    if config.n_max < config.n_min:
        raise ValueError("n_max must be >= n_min")
    if config.repeats < 1:
        raise ValueError("repeats must be >= 1")
    if config.warmups < 0:
        raise ValueError("warmups must be >= 0")
    if config.shots < 1:
        raise ValueError("shots must be >= 1")
    if config.optimization_level not in {0, 1, 2, 3}:
        raise ValueError("optimization_level must be between 0 and 3")
    if config.statevector_qubit_limit < 1:
        raise ValueError("statevector_qubit_limit must be >= 1")
    if config.max_qubits is not None and config.max_qubits < 1:
        raise ValueError("max_qubits must be >= 1")

    available = set(list_all())
    unknown = sorted(set(config.comparators) - available)
    if unknown:
        raise KeyError(f"Unknown comparators: {', '.join(unknown)}")


def make_config(
    *,
    n_min: int,
    n_max: int,
    comparators: Sequence[str] | None = None,
    repeats: int = 5,
    warmups: int = 1,
    shots: int = 1024,
    A: int = 0,
    B: int = 1,
    seed: int = 1,
    optimization_level: int = 1,
    basis_gates: Sequence[str] | None = None,
    simulation_method: str = "automatic",
    statevector_qubit_limit: int = 24,
    max_qubits: int | None = None,
) -> BenchmarkConfig:
    """Create and validate a benchmark configuration."""
    selected = tuple(comparators) if comparators is not None else tuple(list_all())
    if len(set(selected)) != len(selected):
        raise ValueError("comparators must not contain duplicates")
    config = BenchmarkConfig(
        comparators=selected,
        n_min=n_min,
        n_max=n_max,
        repeats=repeats,
        warmups=warmups,
        shots=shots,
        a_value=A,
        b_value=B,
        seed=seed,
        optimization_level=optimization_level,
        basis_gates=tuple(basis_gates) if basis_gates else None,
        simulation_method=simulation_method,
        statevector_qubit_limit=statevector_qubit_limit,
        max_qubits=max_qubits,
    )
    _validate_config(config)
    return config


def _build_comparator(config: BenchmarkConfig, comparator: str, n: int) -> BuildOutput:
    implementation = get(comparator)
    input_values = {"a": config.a_value, "b": config.b_value}
    constant_operands = tuple(
        getattr(implementation.meta, "constant_operands", ())
    )
    if len(set(constant_operands)) != len(constant_operands):
        raise ValueError(f"Comparator {comparator!r} declares duplicate constant operands")
    unknown_constants = sorted(set(constant_operands) - set(_OPERANDS))
    if unknown_constants:
        raise ValueError(
            f"Comparator {comparator!r} declares unsupported constant operands: "
            f"{', '.join(unknown_constants)}"
        )

    build_kwargs = {
        f"{operand}_value": input_values[operand]
        for operand in constant_operands
    }
    try:
        return implementation.build(n, **build_kwargs)
    except ValueError as exc:
        raise UnsupportedBenchmarkPoint(str(exc)) from exc


def _prepare_inputs(
    built: BuildOutput,
    config: BenchmarkConfig,
    comparator: str,
    n: int,
) -> QuantumCircuit:
    qc = built.circuit
    prefix = QuantumCircuit(*qc.qregs, *qc.cregs, name="prep")
    input_values = {"a": config.a_value, "b": config.b_value}
    implementation = get(comparator)
    constant_operands = tuple(
        getattr(implementation.meta, "constant_operands", ())
    )
    for operand in _OPERANDS:
        value = input_values[operand]
        if value < 0 or value >= 1 << n:
            raise UnsupportedBenchmarkPoint(
                f"{operand.upper()}={value} does not fit in {n} bits"
            )
        if operand in constant_operands:
            continue
        if operand not in built.regs:
            raise KeyError(
                f"Comparator {comparator!r} has no register for operand {operand!r}"
            )
        load_int(prefix, built.regs[operand], value)
    return prefix.compose(qc)


def _run_iteration(
    config: BenchmarkConfig,
    comparator: str,
    n: int,
) -> _IterationResult:
    total_start = time.perf_counter_ns()

    build_start = time.perf_counter_ns()
    built = _build_comparator(config, comparator, n)
    build_end = time.perf_counter_ns()

    prepared = _prepare_inputs(built, config, comparator, n)
    if config.max_qubits is not None and prepared.num_qubits > config.max_qubits:
        raise UnsupportedBenchmarkPoint(
            f"circuit has {prepared.num_qubits} qubits, above max_qubits="
            f"{config.max_qubits}"
        )

    backend = _simulation_backend(
        prepared.num_qubits,
        seed=config.seed,
        simulation_method=config.simulation_method,
        statevector_qubit_limit=config.statevector_qubit_limit,
    )

    transpile_start = time.perf_counter_ns()
    transpiled = transpile(
        prepared,
        backend=backend,
        basis_gates=(
            list(config.basis_gates) if config.basis_gates is not None else None
        ),
        optimization_level=config.optimization_level,
        seed_transpiler=config.seed,
    )
    transpile_end = time.perf_counter_ns()

    simulate_start = time.perf_counter_ns()
    counts = backend.run(transpiled, shots=config.shots).result().get_counts()
    simulate_end = time.perf_counter_ns()

    if built.decode_counts is not None:
        built.decode_counts(counts)
    elif not counts:
        raise ValueError("simulator returned no measurement counts")

    total_end = time.perf_counter_ns()
    return _IterationResult(
        timings_ns={
            "build": build_end - build_start,
            "transpile": transpile_end - transpile_start,
            "simulate": simulate_end - simulate_start,
            "total": total_end - total_start,
        },
        backend_method=str(backend.options.method),
        num_qubits=prepared.num_qubits,
    )


def _benchmark_point(
    config: BenchmarkConfig,
    comparator: str,
    n: int,
) -> BenchmarkPoint:
    try:
        for _ in range(config.warmups):
            _run_iteration(config, comparator, n)

        samples = {stage: [] for stage in TIMING_STAGES}
        backend_methods: set[str] = set()
        num_qubits: set[int] = set()
        for _ in range(config.repeats):
            iteration = _run_iteration(config, comparator, n)
            for stage in TIMING_STAGES:
                samples[stage].append(iteration.timings_ns[stage])
            backend_methods.add(iteration.backend_method)
            num_qubits.add(iteration.num_qubits)

        if len(backend_methods) != 1 or len(num_qubits) != 1:
            raise RuntimeError("benchmark configuration changed between repetitions")
        return BenchmarkPoint(
            comparator=comparator,
            n=n,
            status="ok",
            timings={
                stage: TimingStats.from_samples(samples[stage])
                for stage in TIMING_STAGES
            },
            backend_method=next(iter(backend_methods)),
            num_qubits=next(iter(num_qubits)),
        )
    except UnsupportedBenchmarkPoint as exc:
        return BenchmarkPoint(
            comparator=comparator,
            n=n,
            status="skipped",
            error=str(exc) or "unsupported configuration",
        )
    except Exception as exc:
        return BenchmarkPoint(
            comparator=comparator,
            n=n,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_benchmark(
    config: BenchmarkConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    """Benchmark every selected comparator across the configured width range."""
    _validate_config(config)
    total_points = len(config.comparators) * len(config.widths)
    points: list[BenchmarkPoint] = []
    current = 0
    for comparator in config.comparators:
        for n in config.widths:
            current += 1
            if progress is not None:
                progress(
                    f"[{current}/{total_points}] Benchmarking {comparator} at n={n}"
                )
            points.append(_benchmark_point(config, comparator, n))
    return BenchmarkReport(config=config, points=tuple(points))


def _limited_rankings(
    report: BenchmarkReport,
    *,
    stage: str,
    top: int | None,
    allow_partial: bool,
) -> list[ComparatorRanking]:
    _validate_stage(stage)
    if top is not None and top < 1:
        raise ValueError("top must be >= 1")
    rankings = report.rankings(stage, allow_partial=allow_partial)
    return rankings if top is None else rankings[:top]


def _selected_comparators(
    report: BenchmarkReport,
    rankings: Sequence[ComparatorRanking],
    top: int | None,
) -> set[str]:
    if top is None:
        return set(report.config.comparators)
    return {ranking.comparator for ranking in rankings}


def _format_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in text_rows)])


def format_table(
    report: BenchmarkReport,
    *,
    stage: str = "total",
    top: int | None = None,
    allow_partial: bool = False,
) -> str:
    """Render a human-readable ranking and per-size timing matrix."""
    rankings = _limited_rankings(
        report,
        stage=stage,
        top=top,
        allow_partial=allow_partial,
    )
    selected = _selected_comparators(report, rankings, top)

    ranking_rows = [
        (
            ranking.rank,
            ranking.comparator,
            f"{ranking.completed_sizes}/{ranking.requested_sizes}",
            f"{_milliseconds(ranking.mean_median_ns):.3f}",
            (
                f"{_milliseconds(ranking.minimum_ns):.3f}–"
                f"{_milliseconds(ranking.maximum_ns):.3f}"
            ),
        )
        for ranking in rankings
    ]
    if not ranking_rows:
        ranking_rows = [("-", "No eligible comparators", "-", "-", "-")]

    timing_rows: list[tuple[object, ...]] = []
    ordered_keys = [ranking.comparator for ranking in rankings]
    if top is None:
        ordered_keys.extend(
            comparator
            for comparator in report.config.comparators
            if comparator not in ordered_keys
        )
    for comparator in ordered_keys:
        values: list[object] = [comparator]
        for n in report.config.widths:
            point = report.point(comparator, n)
            if point.status == "ok":
                values.append(f"{_milliseconds(point.timings[stage].median_ns):.3f}")
            elif point.status == "skipped":
                values.append("SKIP")
            else:
                values.append("ERROR")
        timing_rows.append(tuple(values))

    failure_rows = [
        (point.comparator, point.n, point.status.upper(), point.error or "")
        for point in report.points
        if point.comparator in selected and point.status != "ok"
    ]

    config = report.config
    sections = [
        "qcmp runtime benchmark",
        (
            f"Stage: {stage} | n: {config.n_min}..{config.n_max} | "
            f"repeats: {config.repeats} | warmups: {config.warmups} | "
            f"shots: {config.shots} | A: {config.a_value} | B: {config.b_value}"
        ),
        "",
        "Ranking (milliseconds)",
        _format_table(
            ranking_rows,
            ("Rank", "Comparator", "Coverage", "Mean median", "Observed range"),
        ),
        "",
        f"Median {stage} time by input width (milliseconds)",
        _format_table(
            timing_rows,
            ("Comparator", *(f"n={n}" for n in config.widths)),
        ),
    ]
    if failure_rows:
        sections.extend(
            [
                "",
                "Skipped or failed measurements",
                _format_table(
                    failure_rows,
                    ("Comparator", "n", "Status", "Reason"),
                ),
            ]
        )
    return "\n".join(sections)


def format_json(
    report: BenchmarkReport,
    *,
    stage: str = "total",
    top: int | None = None,
    allow_partial: bool = False,
) -> str:
    return json.dumps(
        report.to_dict(stage=stage, top=top, allow_partial=allow_partial),
        indent=2,
        sort_keys=True,
    )


def format_csv(
    report: BenchmarkReport,
    *,
    stage: str = "total",
    top: int | None = None,
    allow_partial: bool = False,
) -> str:
    """Render one CSV row per comparator and input width."""
    rankings = _limited_rankings(
        report,
        stage=stage,
        top=top,
        allow_partial=allow_partial,
    )
    selected = _selected_comparators(report, rankings, top)
    ranking_by_key = {ranking.comparator: ranking for ranking in rankings}
    output = io.StringIO(newline="")
    fieldnames = [
        "rank",
        "comparator",
        "n",
        "status",
        "stage",
        "backend_method",
        "num_qubits",
        "repeats",
        "shots",
        "minimum_ms",
        "first_quartile_ms",
        "median_ms",
        "third_quartile_ms",
        "maximum_ms",
        "mean_median_ms",
        "error",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for point in report.points:
        if point.comparator not in selected:
            continue
        ranking = ranking_by_key.get(point.comparator)
        row: dict[str, object] = {
            "rank": ranking.rank if ranking else "",
            "comparator": point.comparator,
            "n": point.n,
            "status": point.status,
            "stage": stage,
            "backend_method": point.backend_method or "",
            "num_qubits": point.num_qubits if point.num_qubits is not None else "",
            "repeats": report.config.repeats,
            "shots": report.config.shots,
            "mean_median_ms": (
                _milliseconds(ranking.mean_median_ns) if ranking else ""
            ),
            "error": point.error or "",
        }
        if point.timings is not None:
            stats = point.timings[stage]
            row.update(
                minimum_ms=_milliseconds(stats.minimum_ns),
                first_quartile_ms=_milliseconds(stats.first_quartile_ns),
                median_ms=_milliseconds(stats.median_ns),
                third_quartile_ms=_milliseconds(stats.third_quartile_ns),
                maximum_ms=_milliseconds(stats.maximum_ns),
            )
        writer.writerow(row)
    return output.getvalue()


def plot_report(
    report: BenchmarkReport,
    path: str | Path,
    *,
    stage: str = "total",
    top: int | None = None,
    allow_partial: bool = False,
    log_y: bool = False,
) -> Path:
    """Write a line chart with interquartile timing bands."""
    rankings = _limited_rankings(
        report,
        stage=stage,
        top=top,
        allow_partial=allow_partial,
    )
    selected = _selected_comparators(report, rankings, top)
    ordered = [ranking.comparator for ranking in rankings]
    if top is None:
        ordered.extend(
            comparator
            for comparator in report.config.comparators
            if comparator in selected and comparator not in ordered
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Graphical output requires matplotlib; install qcmp with the "
            "'plot' optional dependency"
        ) from exc

    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = 0
    for comparator in ordered:
        points = [report.point(comparator, n) for n in report.config.widths]
        if not any(point.status == "ok" for point in points):
            continue
        missing = float("nan")
        x_values = [point.n for point in points]
        medians = [
            (
                _milliseconds(point.timings[stage].median_ns)
                if point.status == "ok"
                else missing
            )
            for point in points
        ]
        lower = [
            (
                _milliseconds(point.timings[stage].first_quartile_ns)
                if point.status == "ok"
                else missing
            )
            for point in points
        ]
        upper = [
            (
                _milliseconds(point.timings[stage].third_quartile_ns)
                if point.status == "ok"
                else missing
            )
            for point in points
        ]
        line = axis.plot(x_values, medians, marker="o", linewidth=1.8, label=comparator)[0]
        axis.fill_between(x_values, lower, upper, color=line.get_color(), alpha=0.15)
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        raise ValueError("no successful benchmark points are available to plot")

    axis.set_xlabel("Input width n")
    axis.set_ylabel(f"Median {stage} time (ms)")
    axis.set_title("qcmp runtime benchmark")
    axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    if log_y:
        axis.set_yscale("log")
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def format_report(
    report: BenchmarkReport,
    output_format: str,
    *,
    stage: str = "total",
    top: int | None = None,
    allow_partial: bool = False,
) -> str:
    if output_format == "table":
        return format_table(
            report,
            stage=stage,
            top=top,
            allow_partial=allow_partial,
        )
    if output_format == "json":
        return format_json(
            report,
            stage=stage,
            top=top,
            allow_partial=allow_partial,
        )
    if output_format == "csv":
        return format_csv(
            report,
            stage=stage,
            top=top,
            allow_partial=allow_partial,
        )
    raise ValueError("output_format must be 'table', 'json', or 'csv'")
