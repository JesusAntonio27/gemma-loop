"""Stage 0 post-processing: verify samples, select prompts and build corpora.

Pipeline:

1. apply the registered Stage-0 sample filter -- ``finish_reason=stop``, the
   thought channel closed, a non-empty answer, rendered length <= 8192 and a
   reasoning share >= 75 % for code + math,
2. verify code with hidden tests and math with exact answers (chat is not
   verified),
3. compute per-problem pass rates and pick Stage 2 prompts from O's samples
   with 1-3 of 4 correct and median length <= 6,000 tokens,
4. write the Stage 1 heal rows (O's kept samples from the train bucket),
5. write verified-trace corpora for the pilot (bucket 50-59) and the NLL/KL
   test set (bucket 60-69), plus the long-trace bucket,
6. record counts and artefact SHA-256 values in the pool manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from gemmaloop.eval.benchmarks import lcb

MAX_LENGTH = 8192
MIN_REASONING_SHARE = 0.75
STAGE2_MEDIAN_MAX = 6000
STAGE2_CORRECT_RANGE = (1, 3)
SAMPLES_PER_PROMPT = 4
LONG_TRACE_MIN_TOKENS = 4096

_THOUGHT_OPEN_RE = re.compile(r"<\|channel>thought\b")
_THOUGHT_CLOSE = "<channel|>"


def _count_tokens(text: str, counter: Callable[[str], int] | None) -> int:
    if not text:
        return 0
    if counter is not None:
        return int(counter(text))
    return max(1, (len(text) + 3) // 4)


def thought_closed(sample: dict[str, Any]) -> bool:
    if "thought_closed" in sample and sample["thought_closed"] is not None:
        return bool(sample["thought_closed"])
    completion = str(sample.get("completion") or "")
    if _THOUGHT_OPEN_RE.search(completion):
        return _THOUGHT_CLOSE in completion
    return True


def _rendered_tokens(
    sample: dict[str, Any],
    problem: dict[str, Any],
    *,
    tokenizer: Any = None,
    renderer: Callable[[list[dict[str, Any]]], int] | None = None,
) -> int:
    if renderer is not None:
        return int(renderer(sample_messages(sample, problem)))
    if tokenizer is not None:
        from gemmaloop.train.labels import assistant_labels

        messages = sample_messages(sample, problem)
        output = assistant_labels(tokenizer, messages, MAX_LENGTH + 1)
        return len(output["input_ids"])
    stored = sample.get("rendered_tokens")
    if stored is not None:
        return int(stored)
    return int(sample.get("prompt_tokens") or 0) + int(sample.get("tokens") or 0)


def sample_messages(sample: dict[str, Any], problem: dict[str, Any]) -> list[dict[str, Any]]:
    """The rendered conversation for heal / verified-trace corpora."""

    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": str(sample.get("content") or ""),
    }
    reasoning = str(sample.get("reasoning") or "")
    if reasoning:
        assistant["reasoning"] = reasoning
    return [
        {"role": "user", "content": problem["prompt"]},
        assistant,
    ]


def reasoning_share(
    sample: dict[str, Any], counter: Callable[[str], int] | None = None
) -> float:
    reasoning_tokens = _count_tokens(str(sample.get("reasoning") or ""), counter)
    answer_tokens = _count_tokens(str(sample.get("content") or ""), counter)
    total = reasoning_tokens + answer_tokens
    return reasoning_tokens / total if total else 0.0


def keep_reason(
    sample: dict[str, Any],
    problem: dict[str, Any],
    *,
    tokenizer: Any = None,
    token_counter: Callable[[str], int] | None = None,
    renderer: Callable[[list[dict[str, Any]]], int] | None = None,
    max_length: int = MAX_LENGTH,
    min_reasoning_share: float = MIN_REASONING_SHARE,
) -> str | None:
    """Return the drop reason, or ``None`` when the sample is kept."""

    if str(sample.get("finish_reason") or "") != "stop":
        return "finish_reason"
    if not thought_closed(sample):
        return "thought_open"
    if not str(sample.get("content") or "").strip():
        return "empty_answer"
    rendered = _rendered_tokens(
        sample, problem, tokenizer=tokenizer, renderer=renderer
    )
    if rendered > int(max_length):
        return "too_long"
    if problem.get("kind") in {"code", "math"}:
        share = reasoning_share(sample, token_counter)
        if share < float(min_reasoning_share):
            return "reasoning_share"
    return None


def extract_code(sample: dict[str, Any]) -> str:
    completion = str(sample.get("completion") or sample.get("content") or "")
    code = lcb.extract_code(completion)
    return code or ""


@dataclass
class GradedSample:
    problem_id: str
    split: str
    kind: str
    sample_index: int
    kept: bool
    drop_reason: str | None
    rendered_tokens: int
    reasoning_share: float
    correct: bool | None
    grade: dict[str, Any] | None
    sample: dict[str, Any] = field(repr=False)

    def to_record(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "split": self.split,
            "kind": self.kind,
            "sample_index": self.sample_index,
            "kept": self.kept,
            "drop_reason": self.drop_reason,
            "rendered_tokens": self.rendered_tokens,
            "reasoning_share": self.reasoning_share,
            "correct": self.correct,
            "grade": self.grade,
        }


@dataclass
class PostProcessResult:
    samples: list[GradedSample]
    stage1_heal: list[dict[str, Any]]
    stage2_prompts: list[dict[str, Any]]
    pilot_verified: list[dict[str, Any]]
    nll_kl_verified: list[dict[str, Any]]
    long_traces: list[dict[str, Any]]
    manifest: dict[str, Any]


def _verify(
    sample: dict[str, Any], problem: dict[str, Any], grader: Any
) -> tuple[bool | None, dict[str, Any] | None]:
    kind = problem.get("kind")
    if kind == "code":
        code = extract_code(sample)
        if not code:
            return False, {"passed": False, "error": "no fenced code block"}
        result = grader.grade_code(code, problem.get("tests") or [])
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        return bool(getattr(result, "passed", payload.get("passed", False))), payload
    if kind == "math":
        prediction = str(sample.get("completion") or sample.get("content") or "")
        result = grader.grade_math(prediction, str(problem.get("answer") or ""))
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        return bool(getattr(result, "correct", payload.get("correct", False))), payload
    return None, None


def _trace_record(
    graded: GradedSample, problem: dict[str, Any]
) -> dict[str, Any]:
    return {
        "problem_id": graded.problem_id,
        "source": problem.get("source"),
        "kind": graded.kind,
        "split": graded.split,
        "pilot_subsplit": problem.get("pilot_subsplit"),
        "sample_index": graded.sample_index,
        "messages": sample_messages(graded.sample, problem),
        "rendered_tokens": graded.rendered_tokens,
        "reasoning_share": graded.reasoning_share,
        "engine": graded.sample.get("engine"),
        "model_revision": graded.sample.get("model_revision"),
    }


def postprocess(
    problems: Iterable[dict[str, Any]],
    samples: Iterable[dict[str, Any]],
    *,
    grader: Any,
    tokenizer: Any = None,
    token_counter: Callable[[str], int] | None = None,
    renderer: Callable[[list[dict[str, Any]]], int] | None = None,
    max_length: int = MAX_LENGTH,
    min_reasoning_share: float = MIN_REASONING_SHARE,
    stage2_median_max: int = STAGE2_MEDIAN_MAX,
    token_counter_method: str = "chars/4",
) -> PostProcessResult:
    problems_by_id = {problem["problem_id"]: problem for problem in problems}
    graded: list[GradedSample] = []
    drop_counts: Counter[str] = Counter()
    by_problem: dict[str, list[GradedSample]] = defaultdict(list)
    all_by_problem: dict[str, list[GradedSample]] = defaultdict(list)

    for sample in samples:
        problem_id = str(sample.get("problem_id") or "")
        problem = problems_by_id.get(problem_id)
        if problem is None:
            drop_counts["unknown_problem"] += 1
            continue
        reason = keep_reason(
            sample,
            problem,
            tokenizer=tokenizer,
            token_counter=token_counter,
            renderer=renderer,
            max_length=max_length,
            min_reasoning_share=min_reasoning_share,
        )
        rendered = _rendered_tokens(sample, problem, tokenizer=tokenizer, renderer=renderer)
        correct: bool | None = None
        grade: dict[str, Any] | None = None
        if reason is None:
            correct, grade = _verify(sample, problem, grader)
        else:
            drop_counts[reason] += 1
        entry = GradedSample(
            problem_id=problem_id,
            split=str(problem.get("split") or ""),
            kind=str(problem.get("kind") or ""),
            sample_index=int(sample.get("sample_index") or 0),
            kept=reason is None,
            drop_reason=reason,
            rendered_tokens=rendered,
            reasoning_share=reasoning_share(sample, token_counter),
            correct=correct,
            grade=grade,
            sample=sample,
        )
        graded.append(entry)
        all_by_problem[problem_id].append(entry)
        if entry.kept:
            by_problem[problem_id].append(entry)

    stage1_heal: list[dict[str, Any]] = []
    stage2_prompts: list[dict[str, Any]] = []
    pilot_verified: list[dict[str, Any]] = []
    nll_kl_verified: list[dict[str, Any]] = []
    long_traces: list[dict[str, Any]] = []
    pass_histogram: Counter[int] = Counter()

    for problem_id, entries in sorted(all_by_problem.items()):
        problem = problems_by_id[problem_id]
        kept = by_problem.get(problem_id, [])
        # The registered rule is "1-3 of 4 correct": count correctness over the
        # four generated samples (a dropped sample is not a correct sample), and
        # use the median over all four rendered lengths.
        n_generated = len(entries)
        correct_count = sum(1 for entry in entries if entry.kept and entry.correct)
        pass_histogram[correct_count] += 1
        lengths = [entry.rendered_tokens for entry in entries]
        median_length = float(statistics.median(lengths)) if lengths else 0.0

        for entry in kept:
            if entry.split == "train":
                stage1_heal.append(_trace_record(entry, problem))
            if entry.split == "pilot_dev" and entry.correct:
                pilot_verified.append(_trace_record(entry, problem))
            if entry.split == "nll_kl" and entry.correct:
                nll_kl_verified.append(_trace_record(entry, problem))
            if (
                entry.split == "long_traces"
                and entry.rendered_tokens >= LONG_TRACE_MIN_TOKENS
            ):
                long_traces.append(_trace_record(entry, problem))

        low, high = STAGE2_CORRECT_RANGE
        if (
            problem.get("kind") in {"code", "math"}
            and problem.get("split") == "train"
            and n_generated == SAMPLES_PER_PROMPT
            and low <= correct_count <= high
            and median_length <= float(stage2_median_max)
        ):
            stage2_prompts.append(
                {
                    "problem_id": problem_id,
                    "source": problem.get("source"),
                    "kind": problem.get("kind"),
                    "prompt": problem.get("prompt"),
                    "n_generated": n_generated,
                    "n_kept": len(kept),
                    "n_correct": correct_count,
                    "pass_rate": correct_count / SAMPLES_PER_PROMPT,
                    "median_length": median_length,
                }
            )

    graded_counts = Counter("correct" if item.correct else "incorrect" for item in graded if item.kept and item.correct is not None)
    manifest = {
        "stage": "stage0_postprocess",
        "version": 1,
        "filter": {
            "max_length": int(max_length),
            "min_reasoning_share": float(min_reasoning_share),
            "finish_reason": "stop",
            "thought_closed": True,
            "non_empty_answer": True,
            "stage2_correct_range": list(STAGE2_CORRECT_RANGE),
            "stage2_median_max": int(stage2_median_max),
            "stage2_rule": "exactly 4 generated samples; 1-3 correct; median rendered length <= limit",
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "reasoning_share_method": str(token_counter_method),
            "long_trace_min_tokens": LONG_TRACE_MIN_TOKENS,
        },
        "counts": {
            "samples_input": len(graded),
            "samples_kept": sum(1 for item in graded if item.kept),
            "samples_dropped": sum(1 for item in graded if not item.kept),
            "dropped_by_reason": dict(sorted(drop_counts.items())),
            "problems_with_samples": len(by_problem),
            "verified_correct": graded_counts["correct"],
            "verified_incorrect": graded_counts["incorrect"],
            "chat_unverified": sum(
                1 for item in graded if item.kept and item.correct is None
            ),
            "stage1_heal_rows": len(stage1_heal),
            "stage2_prompts": len(stage2_prompts),
            "pilot_verified": len(pilot_verified),
            "nll_kl_verified": len(nll_kl_verified),
            "long_traces": len(long_traces),
        },
        "pass_rate_histogram": {
            str(count): number for count, number in sorted(pass_histogram.items())
        },
        "artifacts": {},
    }
    return PostProcessResult(
        samples=graded,
        stage1_heal=stage1_heal,
        stage2_prompts=stage2_prompts,
        pilot_verified=pilot_verified,
        nll_kl_verified=nll_kl_verified,
        long_traces=long_traces,
        manifest=manifest,
    )


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = _canonical(row) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(tmp, path)
    return digest.hexdigest()


def _manifest_sha(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_postprocess(
    result: PostProcessResult,
    output_dir: Path | str,
    *,
    pool_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write every corpus and update the pool manifest in place."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sample_grades.jsonl": [item.to_record() for item in result.samples],
        "stage1_heal.jsonl": result.stage1_heal,
        "stage2_prompts.jsonl": result.stage2_prompts,
        "pilot_verified.jsonl": result.pilot_verified,
        "nll_kl_verified.jsonl": result.nll_kl_verified,
        "long_traces.jsonl": result.long_traces,
    }
    manifest = dict(result.manifest)
    manifest["artifacts"] = {}
    for name, rows in files.items():
        path = output_dir / name
        sha256 = _write_jsonl(path, rows)
        manifest["artifacts"][name] = {"sha256": sha256, "rows": len(rows)}

    manifest_path = output_dir / "postprocess_manifest.json"
    if pool_manifest_path is not None:
        pool_path = Path(pool_manifest_path)
        if pool_path.exists():
            pool_manifest = json.loads(pool_path.read_text(encoding="utf-8"))
            previous = pool_manifest.get("manifest_sha256")
            pool_manifest["postprocess"] = manifest
            pool_manifest.setdefault("artifacts", {})
            for name, entry in manifest["artifacts"].items():
                pool_manifest["artifacts"][f"postprocess/{name}"] = entry
            if previous:
                pool_manifest["previous_manifest_sha256"] = previous
            pool_manifest["manifest_sha256"] = _manifest_sha(pool_manifest)
            _write_json(pool_path, pool_manifest)
            manifest["pool_manifest_sha256"] = pool_manifest["manifest_sha256"]
    _write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "LONG_TRACE_MIN_TOKENS",
    "MAX_LENGTH",
    "MIN_REASONING_SHARE",
    "PostProcessResult",
    "SAMPLES_PER_PROMPT",
    "STAGE2_CORRECT_RANGE",
    "STAGE2_MEDIAN_MAX",
    "extract_code",
    "keep_reason",
    "postprocess",
    "reasoning_share",
    "sample_messages",
    "thought_closed",
    "write_postprocess",
]
