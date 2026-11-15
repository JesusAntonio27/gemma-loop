import json

import pytest

from gemmaloop.eval.results import ItemResult, RunResult


def _item(item_id: str, sample_idx: int, correct: bool) -> ItemResult:
    return ItemResult(
        item_id=item_id,
        sample_idx=sample_idx,
        correct=correct,
        completion_tokens=123,
        finish_reason="stop",
    )


def _run(items: list[ItemResult]) -> RunResult:
    return RunResult(
        variant="L",
        size="E2B",
        n_loops=2,
        bench="gpqa_diamond",
        prereg_tag="prereg-v1",
        items=items,
    )


def test_item_result_fields():
    item = _item("q1", 0, True)
    assert item.item_id == "q1"
    assert item.sample_idx == 0
    assert item.correct is True
    assert item.completion_tokens == 123
    assert item.finish_reason == "stop"


def test_accuracy_and_per_item_mean_with_multiple_samples():
    run = _run(
        [
            _item("q1", 0, True),
            _item("q1", 1, False),
            _item("q2", 0, True),
            _item("q2", 1, True),
            _item("q3", 0, False),
        ]
    )
    assert run.accuracy() == pytest.approx(3 / 5)
    assert run.per_item_mean() == {"q1": pytest.approx(0.5), "q2": 1.0, "q3": 0.0}


def test_single_sample_per_item_matches_accuracy():
    run = _run([_item("q1", 0, True), _item("q2", 0, False)])
    assert run.per_item_mean() == {"q1": 1.0, "q2": 0.0}
    assert run.accuracy() == pytest.approx(0.5)


def test_empty_run():
    run = _run([])
    assert run.accuracy() == 0.0
    assert run.per_item_mean() == {}


def test_json_round_trip(tmp_path):
    run = _run([_item("q1", 0, True), _item("q1", 1, False), _item("q2", 0, True)])
    path = tmp_path / "nested" / "run.json"
    run.save(path)

    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["variant"] == "L"
    assert raw["items"][0]["item_id"] == "q1"

    loaded = RunResult.load(path)
    assert loaded == run
    assert loaded.accuracy() == pytest.approx(run.accuracy())
    assert loaded.per_item_mean() == run.per_item_mean()


def test_load_accepts_string_path(tmp_path):
    run = _run([_item("q1", 0, True)])
    path = tmp_path / "run.json"
    run.save(str(path))
    assert RunResult.load(str(path)) == run


def test_invalid_correct_value_is_rejected():
    with pytest.raises(ValueError):
        ItemResult(
            item_id="q1",
            sample_idx=0,
            correct=2,  # type: ignore[arg-type]
            completion_tokens=1,
            finish_reason="stop",
        )
