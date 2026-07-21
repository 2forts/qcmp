from __future__ import annotations

from dataclasses import dataclass
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
    expression: str
    valid_from: int
    exceptions: dict[int, int]
    samples: dict[int, int]
    degree: int

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "expression": self.expression,
            "domain": f"N >= {self.valid_from}",
            "degree": self.degree,
        }
        if self.exceptions:
            result["exceptions"] = {str(n): value for n, value in self.exceptions.items()}
        result["samples"] = {str(n): value for n, value in self.samples.items()}
        return result


@dataclass(frozen=True)
class EquationReport:
    comparator: str
    variable: str
    equations: dict[str, MetricEquation]
    gate_profiles: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator": self.comparator,
            "variable": self.variable,
            "equations": {
                name: equation.to_dict() for name, equation in self.equations.items()
            },
            "gate_profiles": self.gate_profiles,
        }


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
            if all(_evaluate(coefficients, x) == y for x, y in tail):
                return coefficients, degree, start_index

    raise ValueError(
        f"Could not infer a polynomial of degree <= {max_degree}; "
        "increase sample_max or max_degree"
    )


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

    The comparator is built for every valid N up to ``sample_max``. A polynomial
    is inferred from early samples and checked against independent later samples.
    Exceptional small-N cases are emitted explicitly.
    """

    selected = tuple(metrics or RESOURCE_METRICS)
    accepted = set(RESOURCE_METRICS) | set(RESOURCE_METRIC_ALIASES)
    unknown = sorted(set(selected) - accepted)
    if unknown:
        raise KeyError(f"Unknown resource metrics: {', '.join(unknown)}")
    if sample_max < 4:
        raise ValueError("sample_max must be >= 4")

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

    equations: dict[str, MetricEquation] = {}
    for metric in selected:
        canonical_metric = RESOURCE_METRIC_ALIASES.get(metric, metric)
        points = [(n, row[canonical_metric]) for n, row in sample_rows]
        coefficients, degree, start_index = _infer_tail_polynomial(
            points,
            max_degree=max_degree,
            validation_points=validation_points,
        )
        valid_from = points[start_index][0]
        exceptions = {n: value for n, value in points[:start_index]}
        equations[metric] = MetricEquation(
            metric=metric,
            expression=_format_polynomial(coefficients),
            valid_from=valid_from,
            exceptions=exceptions,
            samples=dict(points),
            degree=degree,
        )

    return EquationReport(
        comparator=comparator.meta.key,
        variable="N",
        equations=equations,
        gate_profiles=registry.to_dict(),
    )
