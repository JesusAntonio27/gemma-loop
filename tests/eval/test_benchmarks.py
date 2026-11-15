"""Tests for the benchmark adapters.

Scoring rules are checked with small in-memory fixtures.  The only test that
touches the network is ``test_gpqa_load_items_small``; it is marked ``slow``
and skips cleanly while the gated GPQA dataset has not been accepted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gemmaloop.eval.benchmarks import (
    aime,
    extract_choice,
    extract_choice_status,
    extract_gold_letter,
    gpqa,
    lcb,
    load_rows,
    mmlupro,
    stable_sample,
)
from gemmaloop.eval.benchmarks import (
    cvbench,
    mathvision,
    mmmupro,
    spatialeval,
)


# ---------------------------------------------------------------------------
# AIME
# ---------------------------------------------------------------------------
def test_aime_last_boxed():
    it = aime.Item("1", "p", "204", {})
    assert aime.score(it, r"... \boxed{12} ... final \boxed{204}")
    assert not aime.score(it, r"\boxed{2 04}x \boxed{205}")


def test_aime_strips_reasoning_first():
    it = aime.Item("1", "p", "7", {})
    assert not aime.score(it, "<|channel>thought\n\\boxed{7}<channel|>The answer is 8")


def test_aime_fbox_and_normalization():
    it = aime.Item("1", "p", "42", {})
    assert aime.score(it, r"so the result is $\fbox{ 42 }$")


# ---------------------------------------------------------------------------
# MMLU-Pro
# ---------------------------------------------------------------------------
def test_mmlupro_regex():
    it = mmlupro.Item("1", "p", "C", {})
    assert mmlupro.score(it, "blah. The answer is (C)")
    assert mmlupro.score(it, "The answer is C")
    assert not mmlupro.score(it, "The answer is (B)")


def test_mmlupro_uses_zero_shot_prompt_and_official_cot_cue():
    it = mmlupro.Item(
        "1",
        mmlupro.format_prompt("What is 2+2?", ["3", "4", "5"], "B"),
        "B",
        {"options": ["3", "4", "5"]},
    )
    messages = mmlupro.build_messages(it)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "Question:\nWhat is 2+2?" in content
    assert "A. 3" in content
    assert "B. 4" in content
    assert "Answer: Let's think step by step." in content
    assert mmlupro.COT_CUE in content


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------
def test_gpqa_shuffle_stable_across_processes():
    code = (
        "from gemmaloop.eval.benchmarks import gpqa; "
        "print(gpqa.shuffle_map('q1', 0), gpqa.shuffle_map('q2', 0))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": s},
        ).stdout
        for s in ("1", "2")
    }
    assert len(outs) == 1 and outs.pop().strip()


def test_gpqa_shuffle_map_is_permutation_and_seed_dependent():
    first = gpqa.shuffle_map("q1", 0)
    assert sorted(first) == [0, 1, 2, 3]
    assert first != gpqa.shuffle_map("q1", 1)
    assert first == gpqa.shuffle_map("q1", 0)


def test_gpqa_score_uses_shuffled_letter_map():
    it = gpqa.Item("q1", "p", "B", {"letter_map": [2, 0, 3, 1]})
    assert gpqa.score(it, "<|channel>thought\nThe answer is (A)<channel|>The answer is (B)")
    assert gpqa.score(it, "The answer is B")
    assert not gpqa.score(it, "The answer is (A)")


@pytest.mark.slow
def test_gpqa_load_items_small():
    try:
        items = gpqa.load_items(limit=2)
    except Exception as exc:  # gated dataset / missing HF login
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(
            marker in text
            for marker in ("401", "gated", "authentication", "unauthorized", "access to dataset")
        ):
            pytest.skip(f"GPQA is gated and not authenticated: {exc}")
        raise
    assert len(items) == 2
    for item in items:
        assert sorted(item.meta["letter_map"]) == [0, 1, 2, 3]
        assert item.answer in "ABCD"
        assert item.meta["options"][ord(item.answer) - ord("A")] == item.meta["correct_answer"]


# ---------------------------------------------------------------------------
# LiveCodeBench
# ---------------------------------------------------------------------------
def test_lcb_extract_code():
    assert (
        lcb.extract_code("x\n```python\nprint(1)\n```\n```python\nprint(2)\n```")
        == "print(2)"
    )


def test_lcb_extract_code_without_block():
    assert lcb.extract_code("print(3)") == ""


def test_lcb_extract_code_any_language_and_last_block():
    assert lcb.extract_code(
        "x\n```py\nprint(1)\n```\n```Python\nprint(2)\n```"
    ) == "print(2)"
    assert lcb.extract_code("x\n```\nprint(3)\n```") == "print(3)"
    assert lcb.extract_code("```python\nonly one fence") == ""


def test_lcb_official_prompt_has_stdin_stdout_instruction():
    prompt = lcb.format_prompt("Add two numbers", "")
    assert prompt.startswith("### Question:\nAdd two numbers")
    assert "### Format:" in prompt
    assert "Read the inputs from stdin" in prompt
    assert "write the answer to stdout" in prompt
    assert "```python\n# YOUR CODE HERE\n```" in prompt
    assert prompt.endswith(
        "### Answer: (use the provided format with backticks)\n\n"
    )

    item = lcb.Item("q1", prompt, None, {})
    messages = lcb.build_messages(item)
    assert messages[0] == {"role": "system", "content": lcb.SYSTEM_MESSAGE_GENERIC}
    assert messages[1]["role"] == "user" and messages[1]["content"] == prompt


def test_lcb_post_cut_set_and_full_release_google_sample(monkeypatch):
    rows = [
        {"question_id": f"post-{index}", "contest_date": "2025-02-01"}
        for index in range(3)
    ] + [
        {"question_id": f"pre-{index}", "contest_date": "2024-12-01"}
        for index in range(4)
    ]
    monkeypatch.setattr(lcb, "_download_release", lambda release: tuple(rows))
    lcb.clear_release_cache()

    post = lcb.load_items(limit=1, seed=0, post_cut=True)
    assert [item.item_id for item in post] == ["post-0", "post-1", "post-2"]
    assert all(item.meta["post_cut"] is True for item in post)

    # lcb_google: stable sample of the FULL release, all dates included.
    first = lcb.load_items(limit=3, seed=7, post_cut=None)
    again = lcb.load_items(limit=3, seed=7, post_cut=None)
    assert [item.item_id for item in first] == [item.item_id for item in again]
    assert len(first) == 3
    assert {item.meta["post_cut"] for item in first} <= {True, False}
    # The post-cut subgroup is recoverable from per-item metadata.
    assert {item.item_id for item in first if item.meta["post_cut"]} == {
        item.item_id for item in first if item.item_id.startswith("post-")
    }


def test_eval_adapters_pin_the_same_revisions_as_prepare_heal_data():
    from scripts import prepare_heal_data as prepare

    expected = {
        "aime": "79037aebdb6580008fb960d17cb21fd3099083e3",
        "lcb": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "mmmupro": "0d7426df4ccb3d8704a992fe6850abee7c262127",
        "mathvision": "2837ddb3f13abaf6b3997c12d80753e5470bd46a",
        "spatialeval": "59ce0450bf65ae8dfa3590062442cf642474d7fe",
        "cvbench": "bc284db50d036958861cb60cdd7b77612052ce0d",
    }
    assert aime.REVISION == expected["aime"]
    assert lcb.REVISION == expected["lcb"]
    assert mmmupro.REVISION == expected["mmmupro"]
    assert mathvision.REVISION == expected["mathvision"]
    assert spatialeval.REVISION == expected["spatialeval"]
    assert cvbench.REVISION == expected["cvbench"]
    assert prepare.AIME_REVISION == expected["aime"]
    assert prepare.LCB_REVISION == expected["lcb"]
    assert prepare.MMMU_PRO_REVISION == expected["mmmupro"]
    assert prepare.MATH_VISION_REVISION == expected["mathvision"]
    assert prepare.SPATIAL_EVAL_REVISION == expected["spatialeval"]
    assert prepare.CV_BENCH_REVISION == expected["cvbench"]


def test_lcb_resolve_uses_post_cut_and_full_release_views():
    from gemmaloop.eval.runner import resolve_bench

    assert resolve_bench("lcb").load_kwargs == {"post_cut": True}
    assert resolve_bench("lcb_google").load_kwargs == {"post_cut": None}
    assert resolve_bench("lcb_google").kind == "lcb"




def test_lcb_load_items_clears_release_cache(tmp_path, monkeypatch):
    lcb.clear_release_cache()
    row = {
        "question_id": "q1",
        "question_content": "add two numbers",
        "starter_code": "",
        "question_title": "A+B",
        "platform": "atcoder",
        "difficulty": "easy",
        "contest_date": "2025-02-01",
    }
    release_file = tmp_path / "test.jsonl"
    release_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        lcb, "hf_hub_download", lambda **kwargs: str(release_file)
    )

    try:
        items = lcb.load_items(limit=2, seed=0)

        assert len(items) == 2
        assert lcb._download_release.cache_info().currsize == 0
        assert lcb._release_question_ids.cache_info().currsize == 0
    finally:
        lcb.clear_release_cache()


def test_lcb_run_official_evaluator_sends_only_sampled_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(lcb, "_lcb_repo", lambda: tmp_path)
    monkeypatch.setattr(lcb, "_lcb_python", lambda repo: tmp_path / "python.exe")
    (tmp_path / "python.exe").write_bytes(b"")
    script = tmp_path / "lcb_grade.py"
    script.write_text("# fake grader", encoding="utf-8")
    monkeypatch.setattr(lcb, "_lcb_grade_script", lambda: script)

    seen = {}

    def fake_run(cmd, cwd, check, capture_output, text, timeout=None):
        samples = json.loads(Path(cmd[2]).read_text(encoding="utf-8"))
        seen["samples"] = samples
        seen["outer_timeout"] = timeout
        Path(cmd[3]).write_text(
            json.dumps(
                {
                    "grades": {"a": [True], "b": [False]},
                    "metadata": {"a": [{}], "b": [{"error_code": -2}]},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(lcb.subprocess, "run", fake_run)
    result = lcb.run_official_evaluator(
        [
            {"question_id": "a", "code_list": ["code-a"]},
            {"question_id": "b", "code_list": ["code-b"]},
        ]
    )
    assert seen["samples"] == {"a": ["code-a"], "b": ["code-b"]}
    assert seen["outer_timeout"] is not None and seen["outer_timeout"] >= 120
    assert result == {"a": [True], "b": [False]}


def test_lcb_run_official_evaluator_requires_equal_sample_counts():
    with pytest.raises(ValueError):
        lcb.run_official_evaluator(
            [
                {"question_id": "a", "code_list": ["one"]},
                {"question_id": "b", "code_list": ["one", "two"]},
            ]
        )


@pytest.mark.slow
def test_lcb_official_grader_real_two_post_cut_problems():
    repo = lcb._lcb_repo()
    python = lcb._lcb_python(repo)
    if not python.exists():
        pytest.skip("LiveCodeBench .venv-lcb is not installed")
    result = lcb.run_official_evaluator(
        [
            {
                "question_id": "abc387_a",
                "code_list": ["a, b = map(int, input().split())\nprint((a + b) ** 2)"],
            },
            {"question_id": "abc388_a", "code_list": ["print(0)"]},
        ]
    )
    assert result["abc387_a"] == [True]
    assert result["abc388_a"] == [False]


@pytest.mark.slow
def test_lcb_official_grader_enforces_per_test_time_limit():
    repo = lcb._lcb_repo()
    python = lcb._lcb_python(repo)
    if not python.exists():
        pytest.skip("LiveCodeBench .venv-lcb is not installed")
    correct = "a, b = map(int, input().split())\nprint((a + b) ** 2)"
    infinite = "while True:\n    pass"
    # 1.5 s per test stays inside LCB's global watchdog budget of
    # (timeout+1)*N+5 = 2N+5 seconds, but exceeds the 1 s per-test limit.  Thus
    # this sample PASSES with a no-op alarm and FAILS (TLE) with the real one.
    sleepy = (
        "import time\n"
        "end = time.time() + 1.5\n"
        "while time.time() < end:\n"
        "    pass\n"
        "a, b = map(int, input().split())\n"
        "print((a + b) ** 2)"
    )
    grades, metadata = lcb.run_official_evaluator_with_metadata(
        [{"question_id": "abc387_a", "code_list": [correct, infinite, sleepy]}],
        timeout=1,
    )
    assert grades["abc387_a"] == [True, False, False]
    assert metadata["abc387_a"][0].get("error_code") is None
    assert metadata["abc387_a"][1]["error_code"] == -3
    assert metadata["abc387_a"][2]["error_code"] == -3


# ---------------------------------------------------------------------------
# MMMU-Pro
# ---------------------------------------------------------------------------
def test_mmmupro_score_shuffled_options():
    item = mmmupro.Item(
        "test_History_1",
        "p",
        "D",
        {"options": ["x", "y", "z", "w"], "option_order": [3, 1, 0, 2]},
    )
    assert mmmupro.score(item, "The answer is (D)")
    assert not mmmupro.score(item, "The answer is (C)")


def test_mmmupro_standard_only_includes_referenced_images():
    row = {
        "id": "test_History_1",
        "question": "Only <image 1> matters here.",
        "options": "['a', 'b']",
        "answer": "A",
        "image_1": Image.new("RGB", (2, 2), "red"),
        "image_2": Image.new("RGB", (2, 2), "green"),
        "subject": "History",
    }
    item = mmmupro.item_from_row(row)
    assert item.meta["image_names"] == ["image_1"]
    parts = mmmupro.build_messages(item)[0]["content"]
    assert [part["type"] for part in parts] == ["text", "image", "text"]


def test_mmmupro_interleaves_multiple_images_at_their_placeholders():
    row = {
        "id": "test_History_1",
        "question": "Compare <image 1> with <image 3>.",
        "options": "['a', 'b']",
        "answer": "(B)",
        "image_1": Image.new("RGB", (2, 2), "red"),
        "image_2": Image.new("RGB", (2, 2), "green"),
        "image_3": Image.new("RGB", (2, 2), "blue"),
        "subject": "History",
    }
    item = mmmupro.item_from_row(row)
    assert item.answer == "B"
    assert item.meta["image_names"] == ["image_1", "image_3"]
    parts = mmmupro.build_messages(item)[0]["content"]
    assert [part["type"] for part in parts] == ["text", "image", "text", "image", "text"]
    assert parts[1]["image"].getpixel((0, 0)) == (255, 0, 0)
    assert parts[3]["image"].getpixel((0, 0)) == (0, 0, 255)
    assert "A. a" in parts[-1]["text"]
    assert mmmupro.STANDARD_POST_PROMPT in parts[-1]["text"]


def test_mmmupro_vision_uses_official_vision_only_prompt():
    row = {
        "id": "test_Design_1",
        "image": Image.new("RGB", (2, 2), "white"),
        "options": "['a', 'b']",
        "answer": "A",
        "subject": "Design",
    }
    item = mmmupro.item_from_row(row, config=mmmupro.VISION_CONFIG)
    assert item.prompt == mmmupro.VISION_POST_PROMPT
    assert "A. a" not in item.prompt
    parts = mmmupro.build_messages(item)[0]["content"]
    assert [part["type"] for part in parts] == ["image", "text"]
    assert parts[1]["text"] == mmmupro.VISION_POST_PROMPT


# ---------------------------------------------------------------------------
# MATH-Vision
# ---------------------------------------------------------------------------
def test_mathvision_multiple_choice_official_exact_match():
    item = mathvision.Item(
        "1", "Which is correct?", "C", {"options": ["10", "20", "30"], "subject": "arithmetic"}
    )
    assert mathvision.score(item, "So the answer is (C).")
    assert not mathvision.score(item, "So the answer is (B).")


def test_mathvision_uses_official_boxed_prompt():
    prompt = mathvision.format_prompt("What is 2+2?", ["3", "4", "5"])
    assert prompt.startswith(mathvision.BOXED_INSTRUCTION)
    assert mathvision.BOXED_INSTRUCTION in prompt
    assert "(A) 3" in prompt
    assert "(B) 4" in prompt
    assert "single word or phrase" not in prompt

    open_prompt = mathvision.format_prompt("Compute 2+2.", [])
    assert mathvision.BOXED_INSTRUCTION in open_prompt
    assert "(A)" not in open_prompt


def test_aime_nested_boxed_braces():
    slash = chr(92)
    item = aime.Item("1", "p", "70", {})
    assert aime.score(item, "Therefore " + slash + "boxed{" + slash + "text{70}}.")
    assert (
        aime.score(item, "Therefore " + slash + "boxed{" + slash + "frac{140}{2}}.")
        is False
    )


def test_mathvision_nested_boxed_braces():
    slash = chr(92)
    item = mathvision.Item("2", "p", "70", {"options": []})
    assert mathvision.score(item, "Therefore " + slash + "boxed{" + slash + "text{70}}.")
    half = mathvision.Item("3", "p", slash + "frac{1}{2}", {"options": []})
    assert mathvision.score(
        half, "Therefore " + slash + "boxed{" + slash + "frac{1}{2}}."
    )


def test_mathvision_open_answer_normalization():
    item = mathvision.Item("2", "Compute.", r"\frac{1}{2}", {"options": [], "subject": "algebra"})
    assert mathvision.score(item, r"Therefore \boxed{\frac{1}{2}}")
    assert mathvision.score(item, r"Therefore \boxed{ 0.5 }")
    assert not mathvision.score(item, r"Therefore \boxed{\frac{1}{3}}")


def test_open_answer_adapters_expose_parse_answer():
    slash = chr(92)
    answer, status = aime.parse_answer(
        "Therefore " + slash + "boxed{" + slash + "text{70}}."
    )
    assert status == "ok" and answer == slash + "text{70}"
    assert aime.parse_answer("no box") == (None, "unparsed")

    answer, status = mathvision.parse_answer(
        "Therefore " + slash + "boxed{" + slash + "frac{1}{2}}."
    )
    assert status == "ok" and answer == slash + "frac{1}{2}"
    assert mathvision.parse_answer("") == (None, "unparsed")

    code, status = lcb.parse_answer("x" + chr(10) + "```py" + chr(10) + "print(1)" + chr(10) + "```")
    assert status == "ok" and code == "print(1)"
    assert lcb.parse_answer("no fence") == (None, "unparsed")


def test_mathvision_ignores_thinking_block():
    item = mathvision.Item("3", "Compute.", "60", {"options": [], "subject": "arithmetic"})
    assert not mathvision.score(item, r"<|channel>thought\n\boxed{60}<channel|>\boxed{61}")


@pytest.mark.slow
def test_mathvision_official_scorer_demo():
    repo = mathvision._mathv_repo()
    python = mathvision._mathv_python(repo)
    if not python.exists():
        pytest.skip("MATH-V .venv-mathv is not installed")
    grades = mathvision.run_official_scorer(
        [
            {
                "id": "fixture-symbolic",
                "completion": r"Therefore \boxed{\sqrt{12}}.",
                "answer": r"2\sqrt{3}",
                "options": [],
            },
            {
                "id": "fixture-mc",
                "completion": "The answer is (C).",
                "answer": "C",
                "options": ["10", "20", "30"],
            },
            {
                "id": "fixture-wrong",
                "completion": r"Therefore \boxed{6}.",
                "answer": "5",
                "options": [],
            },
        ]
    )
    assert grades == {
        "fixture-symbolic": True,
        "fixture-mc": True,
        "fixture-wrong": False,
    }


@pytest.mark.slow
def test_mathvision_official_scorer_strips_thinking_distractor():
    repo = mathvision._mathv_repo()
    python = mathvision._mathv_python(repo)
    if not python.exists():
        pytest.skip("MATH-V .venv-mathv is not installed")
    grades = mathvision.run_official_scorer(
        [
            {
                "id": "fixture-thinking",
                "completion": (
                    "<|channel>thought\nA distractor: \\boxed{7}\n<channel|>"
                    "The final answer is \\frac{1}{2}."
                ),
                "answer": r"\frac{1}{2}",
                "options": [],
            }
        ]
    )
    assert grades == {"fixture-thinking": True}


# ---------------------------------------------------------------------------
# SpatialEval
# ---------------------------------------------------------------------------
def _spatial_rows():
    image = Image.new("RGB", (4, 4), "white")
    vqa = {
        "id": "spatialmap.vqa.2000.0",
        "text": "Which direction? Available options:\nA. Northeast\nB. Southwest",
        "image": image,
        "oracle_answer": "Northeast",
        "oracle_option": "A",
        "oracle_full_answer": "A. Northeast",
    }
    tqa = {
        "id": "spatialmap.tqa.2000.0",
        "text": "Which direction? Available options:\nA. Northeast\nB. Southwest",
        "oracle_answer": "Northeast",
        "oracle_option": "A",
        "oracle_full_answer": "A. Northeast",
    }
    return vqa, tqa


def test_spatialeval_vqa_and_tqa_share_base_item_id():
    vqa_row, tqa_row = _spatial_rows()
    vqa_item = spatialeval.item_from_row(vqa_row, "vqa")
    tqa_item = spatialeval.item_from_row(tqa_row, "tqa")
    assert vqa_item.item_id != tqa_item.item_id
    assert vqa_item.meta["base_item_id"] == tqa_item.meta["base_item_id"]
    assert vqa_item.meta["base_item_id"] == "spatialmap.2000.0"


def test_spatialeval_vqa_has_image_tqa_does_not():
    vqa_row, tqa_row = _spatial_rows()
    vqa_item = spatialeval.item_from_row(vqa_row, "vqa")
    tqa_item = spatialeval.item_from_row(tqa_row, "tqa")
    vqa_parts = spatialeval.build_messages(vqa_item)[0]["content"]
    tqa_parts = spatialeval.build_messages(tqa_item)[0]["content"]
    assert [part["type"] for part in vqa_parts] == ["image", "text"]
    assert [part["type"] for part in tqa_parts] == ["text"]


def test_spatialeval_scores_against_oracle_option():
    vqa_row, tqa_row = _spatial_rows()
    for row, modality in ((vqa_row, "vqa"), (tqa_row, "tqa")):
        item = spatialeval.item_from_row(row, modality)
        assert spatialeval.score(item, "The answer is (A).")
        assert spatialeval.score(item, "A")
        assert not spatialeval.score(item, "The answer is (B).")


# ---------------------------------------------------------------------------
# CV-Bench
# ---------------------------------------------------------------------------
def test_cvbench_scores_parenthesized_letter():
    item = cvbench.Item("0", "How many organs are in the image?", "(C)", {})
    assert cvbench.score(item, "The answer is (C)")
    assert not cvbench.score(item, "The answer is (D)")


# ---------------------------------------------------------------------------
# Shared gold-answer extraction and stable sampling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A", "A"),
        ("(A)", "A"),
        ("A.", "A"),
        ("A. Northeast", "A"),
        ("(C) 1", "C"),
        ("B) down", "B"),
        ("Northeast", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_gold_letter_formats(raw, expected):
    assert extract_gold_letter(raw) == expected
CHOICE_CASES = [
    # Every string named in the evaluation audit.
    ("The answer is B. This answer is correct and consistent.", "B"),
    (
        """Answer: B

This answer is consistent with the evidence.""",
        "B",
    ),
    ("The answer is a hydrogen bond, so (C).", "C"),
    ("**B**", "B"),
    ("Answer: **B**", "B"),
    ("The correct answer is **B**.", "B"),
    ("B. 3.2 eV", "B"),
    ("**Final Answer:** B", "B"),
    # Twenty more realistic Gemma-style outputs.
    ("Answer: (C)", "C"),
    ("The (correct) answer is D", "D"),
    ("Final Answer: A", "A"),
    ("I first thought (A), but the final answer is D.", "D"),
    ("**Answer:** C", "C"),
    (
        """```
E
```""",
        "E",
    ),
    ("The option is (F)", "F"),
    ("Answer: $G$", "G"),
    ("The answer is _H_.", "H"),
    ("We choose I.", "I"),
    ("The answer is J", "J"),
    (chr(92) + "boxed{B}", "B"),
    ("B)", "B"),
    (
        """Answer: B
A distractor (D)
So the answer is C.""",
        "C",
    ),
    ("**Final Answer: (A)**", "A"),
    (
        """Answer: **D**

This matches the derivation.""",
        "D",
    ),
    ("The answer is B2", None),
    ("The answer is Box", None),
    ("the answer is b", None),
    ("I am not sure.", None),
    ("", None),
]


@pytest.mark.parametrize(("text", "expected"), CHOICE_CASES)
def test_extract_choice_formats(text, expected):
    assert extract_choice(text, "ABCDEFGHIJ") == expected


@pytest.mark.parametrize(("text", "expected"), CHOICE_CASES)
def test_extract_choice_status_reports_failures(text, expected):
    answer, status = extract_choice_status(text, "ABCDEFGHIJ")
    assert answer == expected
    assert status == ("ok" if expected is not None else "unparsed")


def test_extract_choice_last_explicit_statement_wins():
    assert extract_choice("First I said (A). The answer is B. Final Answer: C.") == "C"
    assert extract_choice("The answer is B. This answer is correct… (C)?") == "B"




def test_shared_gold_extractor_used_by_adapters():
    cv_item = cvbench.item_from_row(
        {"idx": 1, "question": "q", "choices": ["x", "y", "z"], "answer": "(C)"}
    )
    sp_item = spatialeval.item_from_row(
        {"id": "spatialmap.tqa.1.0", "text": "q", "oracle_option": "A."},
        "tqa",
    )
    mm_item = mmmupro.item_from_row(
        {
            "id": "test_x",
            "question": "Look at <image 1>.",
            "options": "['a', 'b']",
            "answer": "(B)",
            "image_1": Image.new("RGB", (2, 2), "red"),
        }
    )
    assert cv_item.answer == "C"
    assert sp_item.answer == "A"
    assert mm_item.answer == "B"


def test_stable_sample_is_deterministic_uniform_and_order_independent():
    ids = [f"item-{index}" for index in range(182)]
    first = stable_sample(ids, 20, seed=7)
    again = stable_sample(ids, 20, seed=7)
    other = stable_sample(ids, 20, seed=8)

    assert first == again
    assert first != other
    selected = {int(value.split("-")[1]) for value in first}
    assert len(selected) == 20
    assert max(selected) > 100  # not biased to the first ~64 streamed rows
    assert set(stable_sample(list(reversed(ids)), 20, seed=7)) == set(first)


def test_load_rows_uses_full_split_for_real_limits(monkeypatch):
    rows = [{"id": f"item-{index}"} for index in range(200)]
    calls = []

    def fake_load_dataset(dataset_id, **kwargs):
        calls.append(kwargs)
        return rows

    monkeypatch.setattr("gemmaloop.eval.benchmarks.load_dataset", fake_load_dataset)
    selected = load_rows("fake/dataset", limit=100, seed=7)
    assert len(selected) == 100
    assert "streaming" not in calls[0]
    assert selected == stable_sample(rows, 100, seed=7)
