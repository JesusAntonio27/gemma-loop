"""Fixed rubric for the blind pairwise vision judge.

The text is copied verbatim from ``references/gemma-loop/06`` Know-how §4 and
is frozen here before any result is seen.  It must stay at the beginning of
the request, exactly as written, so DeepSeek's automatic prefix cache can hit
across calls; the images (whose base64 payloads differ) come afterwards.

The ``single`` SVG layout removes the mobile viewport constraint, the mobile
JSON criterion and the second image per candidate; the ``full`` layout stays
byte-identical to the frozen template.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a senior front-end design reviewer. Judge two implementations of the same brief
from screenshots only. You do not see code, file names, or model names, and must not guess
them. Judge only visible evidence. Do not reward length, decoration, animation, or
confident copy. A tie is allowed.

Score each candidate 0-10 (integers):
1. Brief fidelity & content completeness (25): required sections, labels, CTAs and facts
   present; nothing required missing or invented.
2. Visual hierarchy & typography (20): reading order, type scale, line length, weight
   contrast.
3. Layout, spacing & composition (15): alignment, grid, whitespace, balance, no
   collisions/overflow.
4. Color, contrast & accessibility (15): coherent palette; body text >= 4.5:1; visible
   focus/hover; color not the only signal.
5. Craft & consistency (15): icons, radii, shadows, borders, spacing tokens consistent;
   no clipped text, broken images, lorem, default styles.
6. Responsive / mobile readiness (10): at 390x844 layout reflows, nav/CTAs usable, no
   horizontal scroll or overlap.

Anchors: 10 exceptional/shippable; 8 strong/minor; 6 competent/visible issues; 4 weak;
2 mostly broken; 0 absent.
Procedure: verify both rendered; list the brief's hard requirements; score A and B
independently with one sentence of visible evidence per criterion; weighted_total =
sum(score*weight)/10; winner = higher total, tie only if |A-B| < 0.5 and no criterion
differs by >= 2; confidence 0-100 reflects ambiguity; output JSON only, no markdown.
"""

FULL_IMAGES_LINE = (
    "CANDIDATE A: images 1 (desktop full) and 2 (mobile full). "
    "CANDIDATE B: images 3 and 4."
)
FULL_HARD_CONSTRAINTS = "HARD CONSTRAINTS: desktop 1440x900, mobile 390x844."
SINGLE_HARD_CONSTRAINTS = "HARD CONSTRAINTS: desktop 1440x900."
FULL_MOBILE_CRITERION = ',"mobile":{"A":0,"B":0,"note":""}'
SINGLE_IMAGE_LINE = (
    "CANDIDATE A: image 1. CANDIDATE B: image 2.\n"
    "Criterion 6 (responsive / mobile readiness) is not applicable: each "
    "candidate is a single SVG image with no mobile capture; do not score it."
)

USER_TEMPLATE = """\
BRIEF: {{brief}}
HARD CONSTRAINTS: desktop 1440x900, mobile 390x844.
Must include: {{required_sections}}; must appear verbatim: {{required_copy}}; forbidden: {{forbidden}}.
CANDIDATE A: images 1 (desktop full) and 2 (mobile full). CANDIDATE B: images 3 and 4.
Return exactly this JSON:
{"render_ok":{"A":true,"B":true},
 "criteria":{"brief_fidelity":{"A":0,"B":0,"note":""},"hierarchy":{"A":0,"B":0,"note":""},
             "layout":{"A":0,"B":0,"note":""},"color_a11y":{"A":0,"B":0,"note":""},
             "craft":{"A":0,"B":0,"note":""},"mobile":{"A":0,"B":0,"note":""}},
 "weighted":{"A":0.0,"B":0.0},"winner":"A|B|tie","confidence":0,
 "decisive_reason":"","flags":["both_bad|mobile_broken|a11y_issue|placeholder_text|render_error"]}
"""


def _as_list(values: list[str] | tuple[str, ...] | None) -> str:
    cleaned = [str(value).strip() for value in (values or []) if str(value).strip()]
    return ", ".join(cleaned) if cleaned else "none"


def user_text(
    brief: str,
    required_sections: list[str],
    required_copy: list[str],
    forbidden: list[str],
    *,
    image_layout: str = "full",
) -> str:
    """Fill the frozen user prompt for one pair.

    Lists are rendered as comma-separated values; an empty list becomes
    ``none`` so the judge never has to interpret a blank placeholder.

    ``image_layout="full"`` (the default, byte-identical to the frozen text)
    describes two captures per candidate.  ``image_layout="single"`` describes
    the SVG case: one image per candidate, with the mobile criterion marked
    not applicable so the judge cannot invent a mobile opinion.
    """
    if image_layout == "full":
        template = USER_TEMPLATE
    elif image_layout == "single":
        template = (
            USER_TEMPLATE.replace(FULL_IMAGES_LINE, SINGLE_IMAGE_LINE)
            .replace(FULL_HARD_CONSTRAINTS, SINGLE_HARD_CONSTRAINTS)
            .replace(FULL_MOBILE_CRITERION, "")
        )
    else:
        raise ValueError(
            f"unknown image layout {image_layout!r}; expected 'full' or 'single'"
        )

    return (
        template.replace("{{brief}}", str(brief).strip())
        .replace("{{required_sections}}", _as_list(required_sections))
        .replace("{{required_copy}}", _as_list(required_copy))
        .replace("{{forbidden}}", _as_list(forbidden))
    )
