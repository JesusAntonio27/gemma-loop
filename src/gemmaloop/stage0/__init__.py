"""Stage 0 data pipeline.

This package implements the registered Stage 0 data work only:

* :mod:`gemmaloop.stage0.sources` -- pinned code/math/chat sources and the
  normalised problem schema,
* :mod:`gemmaloop.stage0.pool` -- decontaminated pool build and deterministic
  splits,
* :mod:`gemmaloop.stage0.grader` -- persistent CPU grader service (code tests
  and exact-answer math),
* :mod:`gemmaloop.stage0.deepseek` -- hidden-test writer for the official
  DeepSeek API,
* :mod:`gemmaloop.stage0.modal_plan` -- Modal cost model and GPU table,
* :mod:`gemmaloop.stage0.postprocess` -- sample verification, pass rates,
  heal rows and verified-trace corpora,
* :mod:`gemmaloop.stage0.local` -- the local Stage-0-lite sampler helpers
  (padded HF engine, per-request caps, resumable shards, manifest).

Nothing here trains a model.  ``modal`` is intentionally *not* a project
dependency; it is imported only by ``scripts/modal_stage0.py`` and executed with
``uvx modal run ...``.
"""

from __future__ import annotations

__all__ = [
    "deepseek",
    "grader",
    "local",
    "modal_plan",
    "pool",
    "postprocess",
    "sources",
]
