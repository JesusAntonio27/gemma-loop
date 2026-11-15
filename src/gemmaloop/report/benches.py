"""Canonical benchmark names, display labels and aliases.

Both the progress charts and the three-column table consume these helpers so
that a benchmark added to the pre-registration gets one spelling everywhere.
"""

from __future__ import annotations

BENCH_ORDER = (
    "lcb",
    "lcb_google",
    "aime",
    "gpqa",
    "mmlupro",
    "mmmupro",
    "mmmupro_vision",
    "mmmu_pro_overall",
    "mathvision",
    "spatialeval_vqa",
    "spatialeval_tqa",
    "cvbench",
)
BENCH_LABELS = {
    "lcb": "LiveCodeBench v6",
    "lcb_google": "LiveCodeBench v6 (full-v6 sample)",
    "aime": "AIME 2026",
    "gpqa": "GPQA Diamond",
    "mmlupro": "MMLU-Pro",
    "mmmupro": "MMMU-Pro Standard(10)",
    "mmmupro_vision": "MMMU-Pro Vision",
    "mmmu_pro_overall": "MMMU-Pro Overall",
    "mathvision": "MATH-Vision",
    "spatialeval_vqa": "SpatialEval VQA",
    "spatialeval_tqa": "SpatialEval TQA",
    "cvbench": "CV-Bench 2D",
}
EXPLORATORY_BENCHES = frozenset(
    {"spatialeval_vqa", "spatialeval_tqa", "cvbench"}
)

_BENCH_ALIASES = {
    "lcb": "lcb",
    "livecodebench": "lcb",
    "lcb_google": "lcb_google",
    "lcb_full_v6": "lcb_google",
    "livecodebench_google": "lcb_google",
    "live_code_bench": "lcb",
    "aime": "aime",
    "aime26": "aime",
    "aime_2026": "aime",
    "gpqa": "gpqa",
    "gpqa_diamond": "gpqa",
    "gpqa_d": "gpqa",
    "mmlupro": "mmlupro",
    "mmlu_pro": "mmlupro",
    "mathvision": "mathvision",
    "math_vision": "mathvision",
    "mathvision_test": "mathvision",
    "mmmupro": "mmmupro",
    "mmmu_pro": "mmmupro",
    "mmmu_pro_standard": "mmmupro",
    "mmmu_pro_10c": "mmmupro",
    "mmmupro_vision": "mmmupro_vision",
    "mmmu_pro_vision": "mmmupro_vision",
    "mmmu_pro_overall": "mmmu_pro_overall",
    "mmmupro_overall": "mmmu_pro_overall",
    "spatialeval": "spatialeval_vqa",
    "spatialeval_vqa": "spatialeval_vqa",
    "spatial_eval": "spatialeval_vqa",
    "spatial_eval_vqa": "spatialeval_vqa",
    "spatialeval_tqa": "spatialeval_tqa",
    "spatial_eval_tqa": "spatialeval_tqa",
    "cvbench": "cvbench",
    "cv_bench": "cvbench",
    "cvbench_2d": "cvbench",
    "cv_bench_2d": "cvbench",
}
_VARIANT_ALIASES = {
    "o": "O",
    "o_h": "O-H",
    "oh": "O-H",
    "o_h_ext": "O-H-ext",
    "oh_ext": "O-H-ext",
    "o_hext": "O-H-ext",
    "l": "L",
    "r": "R",
    "r_damped": "R-damped",
}


def _key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def canonical_bench(name: str) -> str:
    """Map a benchmark name or alias to its canonical key."""
    key = _key(name)
    return _BENCH_ALIASES.get(key, key)


def canonical_variant(name: str) -> str:
    """Map spelling variants (``oh``/``O-H``) to the display name."""
    key = _key(name)
    return _VARIANT_ALIASES.get(key, str(name).strip())


def bench_label(name: str) -> str:
    """Human-readable label for a benchmark key or alias."""
    canonical = canonical_bench(name)
    return BENCH_LABELS.get(canonical, canonical)
