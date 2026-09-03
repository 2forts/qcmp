from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Sequence

from qcmp.metrics.estimator import (
    RESOURCE_METRIC_ALIASES,
    RESOURCE_METRICS,
    estimate_resources,
)
from qcmp.metrics.profiles import DEFAULT_GATE_PROFILES, GateProfileRegistry


@dataclass(frozen=True)
class MetricEquation:
    metric: str
    expression: str | None
    valid_from: int
    exceptions: dict[int, int]
    samples: dict[int, int]
    degree: int | None
    model: str = "polynomial"
    status: str = "inferred"
    validation_samples: dict[int, int] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "expression": self.expression,
            "domain": f"N >= {self.valid_from}" if self.expression is not None else "sampled N only",
            "degree": self.degree,
            "model": self.model,
            "status": self.status,
        }
        if self.exceptions:
            result["exceptions"] = {str(n): value for n, value in self.exceptions.items()}
        result["samples"] = {str(n): value for n, value in self.samples.items()}
        if self.validation_samples:
            result["validation_samples"] = {
                str(n): value for n, value in self.validation_samples.items()
            }
        if self.message:
            result["message"] = self.message
        return result


@dataclass(frozen=True)
class EquationReport:
    comparator: str
    variable: str
    equations: dict[str, MetricEquation]
    gate_profiles: dict[str, dict[str, Any]]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "comparator": self.comparator,
            "variable": self.variable,
            "equations": {
                name: equation.to_dict() for name, equation in self.equations.items()
            },
            "gate_profiles": self.gate_profiles,
        }
        if self.warnings:
            result["warnings"] = list(self.warnings)
        return result


def _poly_add(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    size = max(len(a), len(b))
    out = [Fraction(0) for _ in range(size)]
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_mul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0) for _ in range(len(a) + len(b) - 1)]
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _interpolate(points: Sequence[tuple[int, int]]) -> list[Fraction]:
    result = [Fraction(0)]
    for i, (x_i, y_i) in enumerate(points):
        basis = [Fraction(1)]
        denominator = Fraction(1)
        for j, (x_j, _) in enumerate(points):
            if i == j:
                continue
            basis = _poly_mul(basis, [Fraction(-x_j), Fraction(1)])
            denominator *= x_i - x_j
        scaled = [coefficient * Fraction(y_i, 1) / denominator for coefficient in basis]
        result = _poly_add(result, scaled)
    return result


def _evaluate(coefficients: Sequence[Fraction], x: int) -> Fraction:
    total = Fraction(0)
    for coefficient in reversed(coefficients):
        total = total * x + coefficient
    return total


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"({value.numerator}/{value.denominator})"


def _format_polynomial(coefficients: Sequence[Fraction], variable: str = "N") -> str:
    terms: list[tuple[str, str]] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if coefficient == 0:
            continue

        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        if power == 0:
            body = _format_fraction(magnitude)
        else:
            variable_part = variable if power == 1 else f"{variable}^{power}"
            if magnitude == 1:
                body = variable_part
            else:
                body = f"{_format_fraction(magnitude)}*{variable_part}"
        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    text = ("-" if first_sign == "-" else "") + first_body
    for sign, body in terms[1:]:
        text += f" {sign} {body}"
    return text


def _infer_tail_polynomial(
    points: Sequence[tuple[int, int]],
    *,
    max_degree: int,
    validation_points: int,
    validation_samples: Sequence[tuple[int, int]] = (),
) -> tuple[list[Fraction], int, int]:
    if len(points) < validation_points + 1:
        raise ValueError("Not enough samples to infer an equation")

    for start_index in range(len(points)):
        tail = points[start_index:]
        for degree in range(max_degree + 1):
            required = degree + 1 + validation_points
            if len(tail) < required:
                continue
            coefficients = _interpolate(tail[: degree + 1])
            if all(_evaluate(coefficients, x) == y for x, y in tail) and all(
                _evaluate(coefficients, x) == y for x, y in validation_samples
            ):
                return coefficients, degree, start_index

    raise ValueError(
        f"Could not infer a polynomial of degree <= {max_degree}; "
        "increase sample_max or max_degree"
    )


def _infer_tail_logarithm(
    points: Sequence[tuple[int, int]],
    *,
    validation_points: int,
    validation_samples: Sequence[tuple[int, int]],
) -> tuple[str, int, str]:
    """Try affine functions of integer ceil/floor(log2(N)).

    Repeated points on a plateau are not independent evidence: require
    enough distinct logarithm levels, including held-out validation levels.
    The fit uses training points only, never the held-out samples.
    """
    transforms = (
        (lambda n: (n - 1).bit_length(), "ceil(log2(N))", "ceil_log2"),
        (lambda n: n.bit_length() - 1, "floor(log2(N))", "floor_log2"),
    )
    for start_index in range(len(points)):
        tail = points[start_index:]
        for transform, variable, model in transforms:
            levels: dict[int, int] = {}
            for n, value in tail:
                level = transform(n)
                if level in levels and levels[level] != value:
                    break
                levels[level] = value
            else:
                if len(levels) < 2:
                    continue
                coefficients = _interpolate(list(levels.items())[:2])
                all_points = [*tail, *validation_samples]
                distinct_levels = {transform(n) for n, _ in all_points}
                if len(distinct_levels) < 2 + validation_points:
                    continue
                if all(_evaluate(coefficients, transform(n)) == value for n, value in all_points):
                    return _format_polynomial(coefficients, variable=variable), start_index, model
    raise ValueError("No validated affine ceil/floor-logarithmic model")


def generate_equations(
    comparator: Any,
    *,
    metrics: Iterable[str] | None = None,
    sample_max: int = 12,
    max_degree: int = 4,
    validation_points: int = 3,
    gate_profiles: GateProfileRegistry | None = None,
    build_kwargs: dict[str, Any] | None = None,
) -> EquationReport:
    """Generate and validate resource equations in the variable ``N``.

    The comparator is built for every valid N up to ``sample_max``. Polynomial
    and affine ceil/floor-logarithmic models are tried, with independent checks
    just beyond the sample range and the next power-of-two boundary. Additional
    builds are structural only: they do not simulate a quantum state.

    A metric with no supported model retains its samples and has
    ``status='not_inferred'`` and ``expression=None``. It does not abort the
    other metrics. Exceptional small-N cases are emitted explicitly. Inference
    from finite samples is evidence, not a mathematical proof of a formula.
    """

    selected = tuple(metrics or RESOURCE_METRICS)
    accepted = set(RESOURCE_METRICS) | set(RESOURCE_METRIC_ALIASES)
    unknown = sorted(set(selected) - accepted)
    if unknown:
        raise KeyError(f"Unknown resource metrics: {', '.join(unknown)}")
    if sample_max < 4:
        raise ValueError("sample_max must be >= 4")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    if validation_points < 1:
        raise ValueError("validation_points must be >= 1")

    registry = gate_profiles or DEFAULT_GATE_PROFILES
    kwargs = dict(build_kwargs or {})
    sample_rows: list[tuple[int, dict[str, int]]] = []
    seen_valid = False

    for n in range(1, sample_max + 1):
        try:
            built = comparator.build(n, **kwargs)
        except ValueError:
            if seen_valid:
                raise
            continue
        seen_valid = True
        sample_rows.append((n, estimate_resources(built, gate_profiles=registry).to_dict()))

    if not sample_rows:
        raise ValueError("Comparator did not accept any sampled N")

    # Sparse held-out builds prevent, for example, extrapolating a tree-depth
    # plateau at N=9..12 as a constant for every N>=9. Do not let the fitter
    # move its training window into these points and overfit a new plateau.
    next_power_of_two = 1 << (sample_max - 1).bit_length()
    probe_widths = sorted({sample_max + 1, next_power_of_two + 1})
    validation_rows: list[tuple[int, dict[str, int]]] = []
    warnings: list[str] = []
    for n in probe_widths:
        try:
            built = comparator.build(n, **kwargs)
        except ValueError as exc:
            warnings.append(f"Comparator rejected held-out validation N={n}: {exc}")
            continue
        validation_rows.append((n, estimate_resources(built, gate_profiles=registry).to_dict()))

    equations: dict[str, MetricEquation] = {}
    for metric in selected:
        canonical_metric = RESOURCE_METRIC_ALIASES.get(metric, metric)
        points = [(n, row[canonical_metric]) for n, row in sample_rows]
        probes = [(n, row[canonical_metric]) for n, row in validation_rows]
        try:
            coefficients, degree, start_index = _infer_tail_polynomial(
                points, max_degree=max_degree, validation_points=validation_points,
                validation_samples=probes,
            )
            expression = _format_polynomial(coefficients)
            model = "polynomial"
        except ValueError:
            try:
                expression, start_index, model = _infer_tail_logarithm(
                    points, validation_points=validation_points, validation_samples=probes,
                )
                degree = None
            except ValueError:
                message = (
                    f"{metric}: no validated polynomial of degree <= {max_degree} "
                    "or affine ceil/floor-logarithmic model. Samples are retained; "
                    "no equation is asserted. Increasing the degree can overfit."
                )
                warnings.append(message)
                equations[metric] = MetricEquation(
                    metric=metric, expression=None, valid_from=points[0][0],
                    exceptions={}, samples=dict(points), degree=None,
                    model="unresolved", status="not_inferred",
                    validation_samples=dict(probes), message=message,
                )
                continue
        valid_from = points[start_index][0]
        exceptions = {n: value for n, value in points[:start_index]}
        equations[metric] = MetricEquation(
            metric=metric,
            expression=expression,
            valid_from=valid_from,
            exceptions=exceptions,
            samples=dict(points),
            degree=degree,
            model=model,
            validation_samples=dict(probes),
        )

    return EquationReport(
        comparator=comparator.meta.key,
        variable="N",
        equations=equations,
        gate_profiles=registry.to_dict(),
        warnings=tuple(warnings),
    )
