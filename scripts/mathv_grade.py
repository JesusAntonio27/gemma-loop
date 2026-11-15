"""Run MATH-V's official evaluation code on Gemma-Loop completions.

Usage (inside the eval-only ``.venv-mathv`` environment)::

    python scripts/mathv_grade.py records.json grades.json C:\\Users\\<user>\\ML\\MATH-V

``records.json`` is a list of ``{"id", "response", "answer", "options",
"subject", "level"}`` objects.  The script imports the cloned repo's
``evaluation/evaluate.py``, injects those records as its ``id_raw`` lookup table
and calls the official ``evaluate()`` function.  The output is
``{id: correct}``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: mathv_grade.py <records.json> <grades.json> <MATH-V repo>",
            file=sys.stderr,
        )
        return 2

    records_path = Path(sys.argv[1])
    grades_path = Path(sys.argv[2])
    repo = Path(sys.argv[3])
    evaluation_dir = repo / "evaluation"
    evaluate_path = evaluation_dir / "evaluate.py"
    if not evaluate_path.exists():
        print(f"official evaluator not found at {evaluate_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(evaluation_dir))
    spec = importlib.util.spec_from_file_location("mathv_official_evaluate", evaluate_path)
    assert spec is not None and spec.loader is not None
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    records = json.loads(records_path.read_text(encoding="utf-8"))
    answers = {
        str(record["id"]): {
            "id": str(record["id"]),
            "answer": record["answer"],
            "options": list(record.get("options") or []),
            "subject": record.get("subject", ""),
            "level": record.get("level", 0),
        }
        for record in records
    }
    official.id_raw = answers

    answer_file = records_path.with_suffix(".jsonl")
    with answer_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {"id": str(record["id"]), "response": record["response"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    official.evaluate(str(answer_file), regen_answer=True)
    graded = official.load_jsonl(str(answer_file))
    grades = {str(line["id"]): bool(line.get("correct", False)) for line in graded}
    grades_path.write_text(json.dumps(grades), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
