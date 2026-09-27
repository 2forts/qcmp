from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys

import pytest

from qcmp.__main__ import main
from qcmp.benchmark import (
    BenchmarkConfig,
    BenchmarkPoint,
    BenchmarkReport,
    TimingStats,
    format_csv,
    format_json,
    format_table,
    make_config,
    plot_report,
    run_benchmark,
)


def _stats(*milliseconds: float) -> TimingStats:
    return TimingStats.from_samples(
        round(value * 1_000_000) for value in milliseconds
    )


def _successful_point(comparator: str, n: int, milliseconds: float) -> BenchmarkPoint:
    timings = {
        stage: _stats(milliseconds - 0.1, milliseconds, milliseconds + 0.1)
        for stage in ("build", "transpile", "simulate", "total")
    }
    return BenchmarkPoint(
        comparator=comparator,
        n=n,
        status="ok",
        timings=timings,
        backend_method="automatic",
        num_qubits=2 * n + 1,
    )


def _sample_report() -> BenchmarkReport:
    config = BenchmarkConfig(
        comparators=("alpha", "beta", "partial"),
        n_min=2,
        n_max=3,
        repeats=3,
        warmups=1,
        shots=64,
    )
    return BenchmarkReport(
        config=config,
        points=(
            _successful_point("alpha", 2, 10.0),
            _successful_point("alpha", 3, 20.0),
            _successful_point("beta", 2, 5.0),
            _successful_point("beta", 3, 40.0),
            _successful_point("partial", 2, 1.0),
            BenchmarkPoint(
                comparator="partial",
                n=3,
                status="skipped",
                error="unsupported width",
            ),
        ),
    )


def test_timing_statistics_use_inclusive_quartiles():
    stats = TimingStats.from_samples([1, 2, 3, 4])
    assert stats.minimum_ns == 1
    assert stats.first_quartile_ns == 1.75
    assert stats.median_ns == 2.5
    assert stats.third_quartile_ns == 3.25
    assert stats.maximum_ns == 4
    with pytest.raises(ValueError, match="at least one"):
        TimingStats.from_samples([])


def test_ranking_uses_mean_of_per_size_medians_and_requires_full_coverage():
    report = _sample_report()
    ranking = report.rankings("total")
    assert [entry.comparator for entry in ranking] == ["alpha", "beta"]
    assert ranking[0].mean_median_ns == 15_000_000
    assert ranking[0].minimum_ns == 9_900_000
    assert ranking[0].maximum_ns == 20_100_000

    partial_ranking = report.rankings("total", allow_partial=True)
    assert [entry.comparator for entry in partial_ranking] == [
        "partial",
        "alpha",
        "beta",
    ]


def test_top_filter_is_consistent_across_output_formats():
    report = _sample_report()

    table = format_table(report, top=1)
    assert "alpha" in table
    assert "beta" not in table
    assert "partial" not in table

    payload = json.loads(format_json(report, top=1))
    assert [entry["comparator"] for entry in payload["ranking"]] == ["alpha"]
    assert {entry["comparator"] for entry in payload["results"]} == {"alpha"}

    rows = list(csv.DictReader(io.StringIO(format_csv(report, top=1))))
    assert rows
    assert {row["comparator"] for row in rows} == {"alpha"}
    assert {row["rank"] for row in rows} == {"1"}


def test_table_reports_skipped_points_when_no_top_filter_is_used():
    table = format_table(_sample_report())
    assert "SKIP" in table
    assert "unsupported width" in table


def test_real_benchmark_pipeline_records_every_stage():
    config = make_config(
        comparators=["lt_berti_qpc"],
        n_min=2,
        n_max=2,
        repeats=2,
        warmups=0,
        shots=8,
    )
    report = run_benchmark(config)
    point = report.point("lt_berti_qpc", 2)
    assert point.status == "ok"
    assert point.num_qubits == 5
    assert set(point.timings) == {"build", "transpile", "simulate", "total"}
    assert all(len(stats.samples_ns) == 2 for stats in point.timings.values())
    assert all(stats.median_ns >= 0 for stats in point.timings.values())


def test_max_qubits_skips_oversized_points_without_aborting_report():
    config = make_config(
        comparators=["lt_berti_qpc"],
        n_min=2,
        n_max=3,
        repeats=1,
        warmups=0,
        shots=8,
        max_qubits=5,
    )
    report = run_benchmark(config)
    assert report.point("lt_berti_qpc", 2).status == "ok"
    skipped = report.point("lt_berti_qpc", 3)
    assert skipped.status == "skipped"
    assert "above max_qubits=5" in skipped.error


def test_benchmark_cli_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qcmp",
            "benchmark",
            "--cmp",
            "lt_berti_qpc",
            "--n-min",
            "2",
            "--n-max",
            "2",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--shots",
            "8",
            "--format",
            "json",
            "--quiet",
        ],
    )
    main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["stage"] == "total"
    assert payload["config"]["comparators"] == ["lt_berti_qpc"]
    assert payload["ranking"][0]["rank"] == 1
    assert payload["results"][0]["status"] == "ok"


def test_invalid_benchmark_configuration_is_rejected():
    with pytest.raises(ValueError, match="n_max"):
        make_config(n_min=3, n_max=2)
    with pytest.raises(ValueError, match="repeats"):
        make_config(n_min=1, n_max=2, repeats=0)
    with pytest.raises(KeyError, match="Unknown comparators"):
        make_config(n_min=1, n_max=2, comparators=["not_registered"])


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib is an optional dependency",
)
def test_plot_output(tmp_path):
    destination = tmp_path / "benchmark.png"
    result = plot_report(_sample_report(), destination, top=2, log_y=True)
    assert result == destination
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
