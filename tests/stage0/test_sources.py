"""Stage 0 source loaders: pinned data files, real-schema rows and retries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gemmaloop.stage0 import sources as sources_module
from gemmaloop.stage0.sources import (
    APPS_PARQUET_REVISION,
    APPS_REVISION,
    APPS_SOURCE_REVISION,
    CODECONTESTS_REVISION,
    DATASET_REVISIONS,
    POOL_SOURCES,
    SOURCES_BY_NAME,
    TACO_REVISION,
    convert_row,
    iter_source_rows,
    source_with_loader,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "stage0"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _code_row(question: str = "Add two numbers.") -> dict:
    return {
        "question": question,
        "solutions": '["print(3)"]',
        "input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]}),
        "starter_code": "",
    }


def _load_script(name: str):
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_stage0_pool.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pins and loader wiring
# ---------------------------------------------------------------------------
def test_apps_pins_the_convert_parquet_commit():
    assert APPS_SOURCE_REVISION == "21e74ddf8de1a21436da12e3e653065c5213e9d1"
    assert APPS_PARQUET_REVISION == "0f10e424e13e1c2a69f851e153097b71b6734a1f"
    assert APPS_REVISION == APPS_PARQUET_REVISION
    source = SOURCES_BY_NAME["apps"]
    assert source.revision == APPS_PARQUET_REVISION
    assert source.data_files == ("all/train/0000.parquet",)
    assert source.load_mode == "refs/convert/parquet"
    entry = DATASET_REVISIONS["apps"]
    assert entry["revision"] == APPS_PARQUET_REVISION
    assert entry["source_revision"] == APPS_SOURCE_REVISION
    assert entry["load_mode"] == "refs/convert/parquet"


def test_taco_pins_the_parquet_export_at_the_dataset_commit():
    source = SOURCES_BY_NAME["taco"]
    assert source.revision == TACO_REVISION
    assert source.data_files == ("ALL/train-*.parquet",)
    assert source.load_mode == "data_files"
    entry = DATASET_REVISIONS["taco"]
    assert entry["revision"] == TACO_REVISION
    assert entry["data_files"] == ["ALL/train-*.parquet"]


def test_apps_load_uses_the_pinned_parquet_files(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return ["sentinel"]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    loaded = SOURCES_BY_NAME["apps"].load()
    assert loaded == ["sentinel"]
    path, kwargs = calls[0]
    assert path == "parquet"
    assert kwargs["streaming"] is True
    assert kwargs["split"] == "train"
    files = kwargs["data_files"]["train"]
    assert files == [
        "hf://datasets/codeparrot/apps@0f10e424e13e1c2a69f851e153097b71b6734a1f/"
        "all/train/0000.parquet"
    ]


def test_taco_load_uses_the_pinned_parquet_files(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return ["sentinel"]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    loaded = SOURCES_BY_NAME["taco"].load()
    assert loaded == ["sentinel"]
    path, kwargs = calls[0]
    assert path == "parquet"
    files = kwargs["data_files"]["train"]
    assert files == [
        "hf://datasets/BAAI/TACO@d593ed0a2becbbc952230bb89be09189bf1056dc/"
        "ALL/train-*.parquet"
    ]


def test_codecontests_pin_and_language_filter_are_recorded():
    assert SOURCES_BY_NAME["codecontests"].revision == CODECONTESTS_REVISION
    entry = DATASET_REVISIONS["codecontests"]
    assert entry["date_window"] == "pre_2023_05"
    assert entry["language_filter"] == "PYTHON3 then PYTHON"


# ---------------------------------------------------------------------------
# Real-schema row conversion
# ---------------------------------------------------------------------------
def test_taco_convert_real_parquet_row():
    row = _fixture("taco_row.json")
    problem = convert_row("taco", row)
    assert problem is not None
    assert problem["kind"] == "code"
    assert problem["tests"] == [{"input": "5 3 10\n1 2 3 4 5\nRGBRR\n", "output": "2\n", "type": "stdin"}]
    # solutions is a JSON string in the parquet export, not a list.
    assert problem["metadata"]["reference_solution"].startswith("INF = 10000000000.0")
    assert not problem["metadata"]["reference_solution"].startswith("[")


def test_apps_convert_real_parquet_row():
    row = _fixture("apps_row.json")
    problem = convert_row("apps", row)
    assert problem is not None
    assert problem["source"] == "apps"
    assert problem["tests"] == [{"input": "4\n4\n0001\n1000\n0011\n0111\n", "output": "1\n3 \n-1\n0\n", "type": "stdin"}]
    assert problem["metadata"]["reference_solution"].startswith("for _ in range")
    assert problem["metadata"]["raw_id"] == 0


def test_codecontests_convert_real_parquet_row_prefers_python3():
    row = _fixture("codecontests_row.json")
    problem = convert_row("codecontests", row)
    assert problem is not None
    assert problem["source"] == "codecontests"
    # language enum 3 is PYTHON3 in contest_problem.proto.
    assert problem["metadata"]["reference_solution"].startswith("for _ in range(int(input()))")
    assert [test["input"] for test in problem["tests"]] == [
        "3\n((()))\n(())()\n()(()",
        "3\n((()))\n(())()\n()())",
    ]
    assert all(test["type"] == "stdin" for test in problem["tests"])
    assert problem["metadata"]["date_window"] == "pre_2023_05"


def test_codecontests_rejects_rows_without_a_python_solution():
    assert convert_row("codecontests", _fixture("codecontests_no_python_row.json")) is None


# ---------------------------------------------------------------------------
# Bounded retries
# ---------------------------------------------------------------------------
def test_iter_source_rows_retries_transient_errors_and_resumes():
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        attempt = calls["count"]

        def stream():
            yield _code_row("A.")
            if attempt == 1:
                raise OSError("peer closed connection without sending complete message body")
            yield _code_row("B.")
            yield _code_row("C.")

        return stream()

    source = source_with_loader("taco", loader)
    rows = list(
        iter_source_rows(
            source,
            retry_attempts=3,
            retry_backoff=0.0,
        )
    )
    assert [index for index, _ in rows] == [0, 1, 2]
    assert calls["count"] == 2


def test_iter_source_rows_does_not_hide_persistent_transient_errors():
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        raise OSError("network down")

    source = source_with_loader("taco", loader)
    with pytest.raises(OSError, match="network down"):
        list(iter_source_rows(source, retry_attempts=3, retry_backoff=0.0))
    assert calls["count"] == 3


def test_iter_source_rows_does_not_retry_non_transient_errors():
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        raise ValueError("bad schema")

    source = source_with_loader("taco", loader)
    with pytest.raises(ValueError, match="bad schema"):
        list(iter_source_rows(source, retry_attempts=3, retry_backoff=0.0))
    assert calls["count"] == 1


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------
def test_dry_run_sources_prints_accepted_counts(capsys):
    module = _load_script("build_pool_dry_run_under_test")
    sources = [
        source_with_loader(
            "taco",
            lambda: [_code_row("A."), _code_row("B."), {"question": "unusable"}],
            quota=10,
        ),
        source_with_loader(
            "apps",
            lambda: [_code_row("C.")],
            quota=10,
        ),
    ]
    code = module.dry_run_sources(rows=10, sources=sources)
    assert code == 0
    out = capsys.readouterr().out
    assert "taco" in out and "accepted=2" in out
    assert "apps" in out and "accepted=1" in out


def test_dry_run_sources_reports_failures(capsys):
    module = _load_script("build_pool_dry_run_failure_under_test")

    def broken():
        raise RuntimeError("Dataset scripts are no longer supported")

    sources = [source_with_loader("taco", broken, quota=10)]
    code = module.dry_run_sources(rows=10, sources=sources)
    assert code == 1
    out = capsys.readouterr().out
    assert "taco" in out and "ERROR" in out
