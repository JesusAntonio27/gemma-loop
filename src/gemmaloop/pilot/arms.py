"""Parse the registered pilot arms from ``prereg.yaml``.

The pilot harness never accepts arm ids from the command line.  It reads the
arm strings registered under ``pilot.p2.arms`` and turns them into explicit
:class:`ArmSpec` records:

* ``L`` arms are the looped heals; the block, bridge and teacher are parsed
  from the registered text.
* A generic ``O-H + <teacher>`` entry expands into the block-matched controls
  the registered rule needs.  The E4B O-H expands to 25-29, 16-18 and 15-29
  (the two round-7 added short heals); the 12B O-H expands to 25-29, its
  teacher-step pair.
* The two ``single_exit_legacy`` arms are the descriptive anytime-tax
  references.  They are marked ``descriptive`` and ``eligible=False`` and may
  never enter rule B.

Everything else (budget, recipe, selection thresholds) stays in prereg.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ArmSpec",
    "block_stage_arms",
    "parse_arm_text",
    "parse_registered_arms",
    "pilot_arms",
    "teacher_stage_arms",
]

TEACHER_CODES = {
    "e4b": "e4b",
    "12b": "12b",
    "12b_qat_q4_0": "12b",
    "12b qat q4_0": "12b",
    "12b_qat": "12b",
}
BRIDGE_CODES = {
    "zero_init_per_channel_g": "gch",
    "zero_init_per_channel": "gch",
    "zero-init g": "gch",
    "zero_init_g": "gch",
    "zero init g": "gch",
    "gch": "gch",
    "legacy_scalar": "scalar",
    "legacy scalar": "scalar",
    "scalar": "scalar",
}
LOOP_BLOCKS = ("25-29", "16-18", "15-29")

TEACHER_STAGE_IDS = frozenset(
    {
        "L-25-29-gch-e4b",
        "L-25-29-gch-12b",
        "OH-25-29-e4b",
        "OH-25-29-12b",
    }
)

_BLOCK_RE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)$")


@dataclass(frozen=True)
class ArmSpec:
    """One registered pilot arm, fully resolved."""

    arm_id: str
    kind: str  # "L" or "O-H"
    block: tuple[int, int]
    bridge: str  # "gch" or "scalar"
    teacher: str  # "e4b" or "12b"
    registered_teacher: str
    single_exit: str | None = None
    descriptive: bool = False
    eligible: bool = True
    source: str = ""

    @property
    def is_loop(self) -> bool:
        return self.kind == "L"

    @property
    def objective(self) -> str:
        if self.single_exit:
            return "single_exit"
        return "dial" if self.is_loop else "heal"

    @property
    def runtime(self) -> dict:
        return {"teacher": self.teacher, "runtime": f"{self.teacher}_registered"}

    @property
    def block_size(self) -> int:
        return int(self.block[1]) - int(self.block[0]) + 1

    @property
    def flops_ratio_key(self) -> int:
        """The smaller block is the later tie-break; used by the reporter."""

        return self.block_size

    @property
    def single_exit_horizon(self) -> tuple[float, ...] | None:
        """The legacy single-exit N distribution, if any."""

        if self.single_exit == "n2_3_0901":
            # sample_n_loops enumerates with start=1: index 0 is N=1.
            return (0.0, 0.9, 0.1)
        if self.single_exit == "n4":
            # Length 4: index 0 is N=1, so the mass sits on N=4.
            return (0.0, 0.0, 0.0, 1.0)
        return None

    def to_dict(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "kind": self.kind,
            "block": list(self.block),
            "bridge": self.bridge,
            "teacher": self.teacher,
            "registered_teacher": self.registered_teacher,
            "single_exit": self.single_exit,
            "descriptive": self.descriptive,
            "eligible": self.eligible,
            "objective": self.objective,
            "runtime": self.runtime,
            "source": self.source,
        }


def _bridge(text: str) -> str:
    for code, value in BRIDGE_CODES.items():
        if code in text:
            return value
    raise ValueError(f"pilot arm does not name a registered bridge: {text!r}")


def _teacher(text: str) -> str:
    tokens = [part.strip() for part in text.split("+")]
    for token in tokens:
        key = token.lower()
        if key in TEACHER_CODES:
            return TEACHER_CODES[key]
    raise ValueError(f"pilot arm does not name a registered teacher: {text!r}")


def _single_exit(text: str) -> str | None:
    lowered = text.lower()
    if "single_exit_legacy" not in lowered and "single-exit legacy" not in lowered:
        return None
    if "n=4" in lowered or "n 4" in lowered:
        return "n4"
    if "n 2/3" in lowered or "n2/3" in lowered:
        return "n2_3_0901"
    raise ValueError(f"unrecognised single-exit legacy arm: {text!r}")


def _arm_id(kind: str, block: tuple[int, int], bridge: str, teacher: str, single_exit: str | None) -> str:
    prefix = "OH" if kind == "O-H" else "L"
    if kind == "O-H":
        arm_id = f"{prefix}-{block[0]}-{block[1]}-{teacher}"
    else:
        arm_id = f"{prefix}-{block[0]}-{block[1]}-{bridge}-{teacher}"
    if single_exit:
        arm_id += f"-single-exit-{'0901' if single_exit == 'n2_3_0901' else 'n4'}"
    return arm_id


def _make(
    kind: str,
    block: tuple[int, int],
    bridge: str,
    teacher: str,
    single_exit: str | None,
    source: str,
) -> ArmSpec:
    descriptive = single_exit is not None
    return ArmSpec(
        arm_id=_arm_id(kind, block, bridge, teacher, single_exit),
        kind=kind,
        block=(int(block[0]), int(block[1])),
        bridge=bridge,
        teacher=teacher,
        registered_teacher=teacher,
        single_exit=single_exit,
        descriptive=descriptive,
        eligible=not descriptive,
        source=source,
    )


def parse_arm_text(text: str) -> list[ArmSpec]:
    """Parse one registered arm string into one or more explicit arms."""

    raw = str(text).strip()
    if not raw:
        raise ValueError("empty pilot arm")
    tokens = [part.strip() for part in raw.split("+")]
    if not tokens:
        raise ValueError(f"invalid pilot arm: {raw!r}")
    head = tokens[0]
    teacher = _teacher(raw)
    single_exit = _single_exit(raw)

    compact = head.upper().replace(" ", "")
    if compact.startswith("O-H"):
        if single_exit:
            raise ValueError(f"single-exit arms are loop arms: {raw!r}")
        explicit = re.match(r"^O-H\[(\d+-\d+)\]$", compact)
        if explicit:
            return [
                _make(
                    "O-H",
                    _parse_block(explicit.group(1)),
                    "gch",
                    teacher,
                    None,
                    raw,
                )
            ]
        if compact != "O-H":
            raise ValueError(f"unrecognised O-H arm head: {raw!r}")
        # Compatibility path for the stale yaml's generic O-H strings.
        blocks = ["25-29", "16-18", "15-29"] if teacher == "e4b" else ["25-29"]
        return [
            _make("O-H", _parse_block(block), "gch", teacher, None, raw)
            for block in blocks
        ]

    match = _BLOCK_RE.match(head)
    if not match:
        raise ValueError(f"pilot arm does not start with a loop block: {raw!r}")
    block = (int(match.group("start")), int(match.group("end")))
    bridge = _bridge(raw)
    return [_make("L", block, bridge, teacher, single_exit, raw)]


def _parse_block(text: str) -> tuple[int, int]:
    match = _BLOCK_RE.match(str(text).strip())
    if not match:
        raise ValueError(f"invalid loop block: {text!r}")
    start, end = int(match.group("start")), int(match.group("end"))
    if end < start:
        raise ValueError(f"invalid loop block: {text!r}")
    return start, end


def parse_registered_arms(arm_texts: Iterable[str]) -> list[ArmSpec]:
    """Parse every registered arm string, in registration order."""

    arms: list[ArmSpec] = []
    for text in arm_texts:
        arms.extend(parse_arm_text(text))
    ids = [arm.arm_id for arm in arms]
    if len(set(ids)) != len(ids):
        duplicates = sorted({arm_id for arm_id in ids if ids.count(arm_id) > 1})
        raise ValueError(f"duplicate registered pilot arms: {duplicates}")
    return arms


def pilot_arms(prereg) -> list[ArmSpec]:
    """The registered pilot arms from a loaded :class:`gemmaloop.prereg.Prereg`."""

    if prereg.pilot is None or prereg.pilot.p2 is None:
        raise ValueError("prereg.yaml has no pilot.p2 section")
    arms = parse_registered_arms(prereg.pilot.p2.arms)
    if len(arms) < 7:
        raise ValueError("the registered pilot matrix needs at least seven arms")
    return arms


def teacher_stage_arms(arms: Sequence[ArmSpec]) -> list[ArmSpec]:
    """The four teacher-step arms, in registration order."""

    selected = [arm for arm in arms if arm.arm_id in TEACHER_STAGE_IDS]
    missing = TEACHER_STAGE_IDS - {arm.arm_id for arm in selected}
    if missing:
        raise ValueError(f"registered arms are missing the teacher step: {sorted(missing)}")
    return selected


def block_stage_arms(arms: Sequence[ArmSpec]) -> list[ArmSpec]:
    """Every registered arm that is not part of the teacher step."""

    return [arm for arm in arms if arm.arm_id not in TEACHER_STAGE_IDS]


def arms_by_id(arms: Iterable[ArmSpec]) -> dict[str, ArmSpec]:
    return {arm.arm_id: arm for arm in arms}


def teacher_from_arm_text(text: str) -> str:
    """Exposed for report labels: the teacher token of a registered arm."""

    return _teacher(str(text))
