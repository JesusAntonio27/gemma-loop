"""Tests for agreement, blind human rating and brief freezing (Task 19).

The agreement numbers are compared against ``sklearn.metrics.cohen_kappa_score``
on a hand-checked example, and the generated rating page is checked to be
truly blind: neither candidate names nor source paths may reach the browser.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from gemmaloop.judge.agreement import agreement

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_task19_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def human_rate():
    return _load_script("human_rate")


@pytest.fixture(scope="module")
def make_briefs():
    return _load_script("make_briefs")


# ---------------------------------------------------------------------------
# agreement()
# ---------------------------------------------------------------------------


JUDGE = ["A", "A", "B", "B", "tie", "A", "B", "tie", "A", "B"]
HUMAN = ["A", "B", "B", "A", "tie", "A", "B", "tie", "B", "B"]


def test_agreement_matches_sklearn_on_known_example():
    result = agreement(JUDGE, HUMAN, iters=400, seed=0)

    assert result["p_o"] == pytest.approx(7 / 10)
    assert result["kappa"] == pytest.approx(cohen_kappa_score(JUDGE, HUMAN))
    lo, hi = result["kappa_ci95"]
    assert lo <= hi

    confusion = result["confusion"]
    assert set(confusion) == {"A", "B", "tie"}
    assert confusion["A"]["A"] == 2
    assert confusion["A"]["B"] == 1
    assert confusion["B"]["A"] == 2
    assert confusion["B"]["B"] == 3
    assert confusion["tie"]["tie"] == 2
    assert sum(sum(row.values()) for row in confusion.values()) == len(JUDGE)


def test_agreement_bootstrap_is_reproducible_with_seed():
    first = agreement(JUDGE, HUMAN, iters=300, seed=7)
    second = agreement(JUDGE, HUMAN, iters=300, seed=7)

    assert first == second


def test_agreement_rejects_malformed_input():
    with pytest.raises(ValueError):
        agreement([], [])
    with pytest.raises(ValueError):
        agreement(["A"], ["A", "B"])
    with pytest.raises(ValueError):
        agreement(["A", ""], ["A", "B"])
    with pytest.raises(ValueError):
        agreement(JUDGE, HUMAN, iters=0)


def test_agreement_undefined_kappa_is_nan_not_a_crash():
    result = agreement(["A", "A", "A"], ["A", "A", "A"], iters=50, seed=0)

    assert math.isnan(result["kappa"])
    assert math.isnan(result["p_o"]) is False
    assert result["p_o"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# scripts/human_rate.py — the browser must never learn the variant names
# ---------------------------------------------------------------------------


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _pair_file(tmp_path: Path) -> tuple[Path, list[dict]]:
    # Deliberately leak the variant names everywhere the browser must not see
    # them: in the pair id and in the source directory names.
    x_desktop = tmp_path / "O-H" / "desktop.png"
    x_mobile = tmp_path / "O-H" / "mobile.png"
    y_desktop = tmp_path / "L" / "desktop.png"
    y_mobile = tmp_path / "L" / "mobile.png"
    _write_png(x_desktop, (255, 0, 0))
    _write_png(x_mobile, (255, 0, 0))
    _write_png(y_desktop, (0, 0, 255))
    _write_png(y_mobile, (0, 0, 255))

    records = [
        {
            "pair_id": "O-H-vs-loop-001",
            "x": {"desktop": str(x_desktop), "mobile": str(x_mobile)},
            "y": {"desktop": str(y_desktop), "mobile": str(y_mobile)},
        },
        {
            "pair_id": "loop-vs-O-H-002",
            "x": [str(x_desktop)],
            "y": [str(y_desktop)],
        },
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path, records


def test_rating_page_never_contains_variant_names(tmp_path, human_rate):
    pairs_path, _records = _pair_file(tmp_path)

    pairs = human_rate.load_pairs(pairs_path)
    batches = human_rate.prepare_batches(
        pairs, tmp_path / "pages", seed=0, batch_size=4
    )

    html = batches[0].html_path.read_text(encoding="utf-8")
    for name in ("O-H", "L", "loop"):
        assert name not in html
    assert "http://" not in html and "https://" not in html


def test_rating_page_shows_only_a_b_and_tie(tmp_path, human_rate):
    pairs_path, _records = _pair_file(tmp_path)

    pairs = human_rate.load_pairs(pairs_path)
    batch = human_rate.prepare_batches(
        pairs, tmp_path / "pages", seed=0, batch_size=4
    )[0]
    html = batch.html_path.read_text(encoding="utf-8")

    assert "<h2>A</h2>" in html
    assert "<h2>B</h2>" in html
    assert "A is better" in html
    assert "B is better" in html
    assert "Tie" in html
    srcs = re.findall(r'src="([^"]+)"', html)
    assert len(srcs) == 6  # 2 pairs x 2 candidates x (2 + 1) images
    for src in srcs:
        assert src.startswith("/assets/")
    # every asset the page references was copied under the page directory
    assets = {src.rsplit("/", 1)[1] for src in srcs}
    copied = {path.name for path in (tmp_path / "pages").rglob("*.*")}
    assert assets <= copied


def test_display_order_is_seeded_and_answers_map_back(tmp_path, human_rate):
    pairs_path, _records = _pair_file(tmp_path)
    pairs = human_rate.load_pairs(pairs_path)

    first = human_rate.prepare_batches(
        pairs, tmp_path / "page_a", seed=3, batch_size=1
    )
    second = human_rate.prepare_batches(
        pairs, tmp_path / "page_b", seed=3, batch_size=1
    )
    assert [batch.pairs[0].order for batch in first] == [
        batch.pairs[0].order for batch in second
    ]

    displayed = first[0].pairs[0]
    record = human_rate.answer_record(displayed, "A", batch_index=0, timestamp="t")
    assert record["choice"] == "A"
    assert record["winner"] == displayed.order[0]
    assert record["display"] == {"A": displayed.order[0], "B": displayed.order[1]}

    tie = human_rate.answer_record(displayed, "tie", batch_index=0, timestamp="t")
    assert tie["winner"] == "tie"


def test_rating_tokens_are_opaque_and_unique(tmp_path, human_rate):
    pairs_path, _records = _pair_file(tmp_path)
    pairs = human_rate.load_pairs(pairs_path)

    batches = human_rate.prepare_batches(
        pairs, tmp_path / "pages", seed=0, batch_size=1
    )
    tokens = [batch.pairs[0].token for batch in batches]
    assert len(tokens) == len(set(tokens))
    assert all(token.startswith("p") for token in tokens)
    assert not any(name in token for token in tokens for name in ("O-H", "L", "loop"))


# ---------------------------------------------------------------------------
# scripts/make_briefs.py — validation and freezing (generation deferred)
# ---------------------------------------------------------------------------


def _brief(index: int, kind: str = "page") -> dict:
    return {
        "brief_id": f"brief-{index:03d}",
        "kind": kind,
        "brief": "Build a responsive landing page for a coffee subscription.",
        "required_sections": ["hero", "pricing", "footer"],
        "required_copy": ["Start your trial"],
        "forbidden": ["lorem ipsum", "placeholder"],
    }


def test_briefs_validation_accepts_the_frozen_format(make_briefs):
    records = [_brief(i, "page") for i in range(3)]

    summary = make_briefs.validate_briefs(records)

    assert summary["n"] == 3
    assert summary["kinds"] == {"page": 3, "component": 0, "svg": 0}


def test_briefs_validation_rejects_bad_records(make_briefs):
    with pytest.raises(make_briefs.BriefError):
        make_briefs.validate_briefs([{**_brief(1), "kind": "deck"}])
    with pytest.raises(make_briefs.BriefError):
        make_briefs.validate_briefs([{**_brief(1), "required_copy": "not a list"}])
    with pytest.raises(make_briefs.BriefError):
        make_briefs.validate_briefs([{**_brief(1), "unexpected": True}])
    with pytest.raises(make_briefs.BriefError):
        make_briefs.validate_briefs([_brief(1), _brief(1)])


def test_briefs_mix_is_60_25_15(make_briefs):
    mixed = (
        [_brief(i, "page") for i in range(90)]
        + [_brief(100 + i, "component") for i in range(38)]
        + [_brief(200 + i, "svg") for i in range(22)]
    )

    counts = make_briefs.check_mix(mixed)

    assert counts == {"page": 90, "component": 38, "svg": 22}
    with pytest.raises(make_briefs.BriefError):
        make_briefs.check_mix([_brief(i, "page") for i in range(10)])


def test_make_briefs_hashes_file_bytes(tmp_path, make_briefs, capsys):
    path = tmp_path / "briefs_eval.jsonl"
    path.write_text(
        "\n".join(json.dumps(_brief(i)) for i in range(2)) + "\n",
        encoding="utf-8",
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert make_briefs.hash_file(path) == expected
    assert make_briefs.main(["--briefs", str(path)]) == 0
    assert expected in capsys.readouterr().out
