"""Recover original APPS/TACO/CodeContests records for frozen code problems.

Only the already pinned revisions registered in
:mod:`gemmaloop.stage0.sources` are read.  Every source -- including
CodeContests, whose tests are literal proto strings rather than JSON -- is
recovered from its pinned record, bound to its locator and problem identity,
and replayed through the old converter before its tests may be authenticated.
A missing or mismatched record excludes the whole problem; there is no
literal-source passthrough.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from gemmaloop.stage0.interpretation import (
    INTERPRETATION_MANIFEST_NAME,
    build_interpretation_manifest,
    write_interpretation_manifest,
)
from gemmaloop.stage0.sources import SOURCES_BY_NAME, convert_row

Loader = Callable[[str, set[int]], Iterable[tuple[int, Mapping[str, Any]]]]


def _default_loader(
    source_name: str, needed_indices: set[int]
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    source = SOURCES_BY_NAME[source_name]
    if not needed_indices:
        return
    maximum = max(needed_indices)
    for index, row in enumerate(source.load()):
        if index in needed_indices:
            yield index, row
        if index >= maximum:
            return


def _locator(
    source_name: str,
    source_index: int,
    problem: Mapping[str, Any],
    *,
    raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = problem.get("metadata") or {}
    actual_raw_id = None
    if isinstance(raw, Mapping):
        if source_name == "apps":
            actual_raw_id = raw.get("problem_id")
        elif source_name == "taco":
            actual_raw_id = raw.get("url")
        elif source_name == "codecontests":
            actual_raw_id = raw.get("name") or raw.get("cf_contest_id")
    locator = {
        "source": source_name,
        "row_index": int(source_index),
        "raw_id": metadata.get("raw_id") if actual_raw_id is None else actual_raw_id,
        "frozen_raw_id": metadata.get("raw_id"),
        "hf_revision": metadata.get("hf_revision"),
        "source_index": metadata.get("source_index"),
    }
    if source_name == "apps":
        locator.update(
            {
                "hf_id": "codeparrot/apps",
                "load_mode": "refs/convert/parquet",
                "data_files": ["all/train/0000.parquet"],
            }
        )
    elif source_name == "taco":
        locator.update(
            {
                "hf_id": "BAAI/TACO",
                "load_mode": "data_files",
                "data_files": ["ALL/train-*.parquet"],
            }
        )
    elif source_name == "codecontests":
        locator.update(
            {
                "hf_id": "deepmind/code_contests",
                "load_mode": "parquet",
                "language_filter": "PYTHON3 then PYTHON",
            }
        )
    return locator


def recover_records(
    frozen_rows: Sequence[Mapping[str, Any]],
    problems_by_id: Mapping[str, Mapping[str, Any]],
    *,
    loader: Loader | None = None,
    sources: Sequence[str] = ("apps", "taco", "codecontests"),
) -> dict[str, dict[str, Any]]:
    """Return ``problem_id -> recovered record`` for the requested sources.

    Every recovered entry records the source, the actual loaded row, its
    locator (including the resolved row index and raw id), the pinned revision
    and whether the old converter reproduces the frozen problem identity.
    """

    loader = loader or _default_loader
    needed: dict[str, set[int]] = {}
    by_source: dict[str, list[tuple[str, int, Mapping[str, Any]]]] = {}
    for frozen_row in frozen_rows:
        problem_id = str(frozen_row["problem_id"])
        problem = problems_by_id.get(problem_id)
        if problem is None:
            continue
        source = str(problem.get("source") or "")
        if source not in sources:
            continue
        source_index = (problem.get("metadata") or {}).get("source_index")
        if not isinstance(source_index, int):
            continue
        needed.setdefault(source, set()).add(source_index)
        by_source.setdefault(source, []).append((problem_id, source_index, problem))

    records: dict[str, dict[str, Any]] = {}
    for source, entries in by_source.items():
        found: dict[int, Mapping[str, Any]] = {}
        for index, row in loader(source, set(needed.get(source) or set())):
            found[int(index)] = row
        for problem_id, source_index, problem in entries:
            raw = found.get(int(source_index))
            revision = (problem.get("metadata") or {}).get("hf_revision")
            converted_id = None
            identity_ok = False
            identity_error = None
            if raw is None:
                identity_error = (
                    f"no {source} record at pinned row index {int(source_index)}"
                )
            else:
                try:
                    converted = convert_row(source, raw, int(source_index))
                except Exception as exc:  # noqa: BLE001 - malformed original record
                    converted = None
                    identity_error = (
                        f"old converter raised {type(exc).__name__}: {exc}"
                    )
                if converted is None:
                    identity_error = identity_error or (
                        f"record at row index {int(source_index)} does not convert"
                    )
                else:
                    converted_id = str(converted.get("problem_id"))
                    identity_ok = converted_id == problem_id
                    if not identity_ok:
                        identity_error = (
                            "recovered identity "
                            f"{converted_id!r} does not match frozen {problem_id!r}"
                        )
            records[problem_id] = {
                "source": source,
                "raw": raw,
                "locator": _locator(
                    source, int(source_index), problem, raw=raw
                ),
                "revision": revision,
                "identity_ok": identity_ok,
                "identity_error": identity_error,
                "converted_problem_id": converted_id,
            }
    return records


def recover_source_types(
    frozen_rows: Sequence[Mapping[str, Any]],
    problems_by_id: Mapping[str, Mapping[str, Any]],
    *,
    out_dir: Any,
    loader: Loader | None = None,
    sources: Sequence[str] = ("apps", "taco", "codecontests"),
    created_utc: str | None = None,
) -> dict[str, Any]:
    records = recover_records(
        frozen_rows, problems_by_id, loader=loader, sources=sources
    )
    return write_interpretation_manifest(
        out_dir,
        frozen_rows,
        problems_by_id,
        records,
        created_utc=created_utc,
    )


__all__ = [
    "INTERPRETATION_MANIFEST_NAME",
    "Loader",
    "recover_records",
    "recover_source_types",
]
