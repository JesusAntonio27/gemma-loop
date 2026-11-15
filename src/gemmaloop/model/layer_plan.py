"""Layer split of a Gemma 4 text stack into prelude, looped block and coda.

Gemma 4 alternates sliding-attention and full-attention layers in a fixed
period (4 sliding + 1 full for E2B, 5 sliding + 1 full for E4B).  Gemma-Loop
reuses the last ``num_kv_shared_layers`` layers, where K/V are produced by the
first ``num_hidden_layers - num_kv_shared_layers`` layers and re-read from
``shared_kv_states``.  The looped block must live strictly inside that reuse
region, so the prelude is the producer region and both the prelude and the loop
are whole attention periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

FULL_ATTENTION = "full_attention"


@dataclass(frozen=True)
class LayerPlan:
    """Split of the layers into prelude, looped block and coda.

    Field order is part of the contract: later tasks build the plan
    positionally.
    """

    num_layers: int
    period: int
    full_layers: tuple[int, ...]
    first_shared: int
    prelude: range
    loop: range
    coda: range


def _period(layer_types: Sequence[str]) -> int:
    """Return the distance between consecutive full-attention layers."""
    full = [i for i, kind in enumerate(layer_types) if kind == FULL_ATTENTION]
    if len(full) < 2:
        raise ValueError(
            "layer_types must contain at least two full_attention layers to "
            "define a period"
        )
    distances = {b - a for a, b in zip(full, full[1:])}
    if len(distances) != 1:
        raise ValueError(f"full_attention layers are not evenly spaced: {full}")
    return distances.pop()


def parse_block(value: str | Sequence[int]) -> tuple[int, int]:
    """Parse ``"START-END"`` (inclusive) or a two-element sequence.

    The returned block is ``(start, end)`` with both endpoints inclusive.
    """
    if isinstance(value, str):
        parts = value.strip().split("-")
        if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
            raise ValueError(
                f"loop block must look like 'START-END' (inclusive), got {value!r}"
            )
        start, end = (int(part) for part in parts)
    else:
        try:
            start, end = (int(part) for part in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"loop block must be (start, end), got {value!r}"
            ) from exc
    if start > end:
        raise ValueError(f"loop block start {start} is after end {end}")
    return start, end


def make_layer_plan(
    text_config: Mapping[str, object],
    coda_periods: int = 1,
    block: tuple[int, int] | Sequence[int] | str | None = None,
) -> LayerPlan:
    """Compute the prelude / loop / coda split from a Gemma 4 text config.

    ``text_config`` may be the raw ``text_config`` object from a Hugging Face
    ``config.json`` or the dict returned by a transformers
    ``Gemma4TextConfig.to_dict()``.

    ``block`` is an optional inclusive ``(start, end)`` pair.  When omitted the
    legacy split is returned: the loop is every KV-consumer period except the
    last ``coda_periods`` periods (E2B: layers 15-29).  An explicit block must
    lie entirely inside the KV-consumer region and leave a non-empty coda,
    i.e. ``first_shared <= start <= end < num_layers - 1``.
    """
    num_layers = int(text_config["num_hidden_layers"])  # type: ignore[arg-type]
    layer_types = list(text_config["layer_types"])  # type: ignore[arg-type]
    num_kv_shared_layers = int(text_config["num_kv_shared_layers"])  # type: ignore[arg-type]

    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries, expected {num_layers}"
        )

    period = _period(layer_types)
    full_layers = tuple(
        i for i, kind in enumerate(layer_types) if kind == FULL_ATTENTION
    )
    first_shared = num_layers - num_kv_shared_layers

    if first_shared % period != 0:
        raise ValueError(
            f"first_shared={first_shared} is not a whole number of periods "
            f"(period={period})"
        )

    if block is not None:
        start, end = parse_block(block)
        if start < first_shared:
            raise ValueError(
                f"loop block {start}-{end} starts inside the KV-producer region "
                f"[0, {first_shared}); producers can never be looped"
            )
        if end >= num_layers - 1:
            raise ValueError(
                f"loop block {start}-{end} leaves an empty coda; the block must "
                f"end before layer {num_layers - 1} (num_layers={num_layers})"
            )
        return LayerPlan(
            num_layers=num_layers,
            period=period,
            full_layers=full_layers,
            first_shared=first_shared,
            prelude=range(0, start),
            loop=range(start, end + 1),
            coda=range(end + 1, num_layers),
        )

    coda_start = num_layers - coda_periods * period
    if coda_start < first_shared:
        raise ValueError(
            f"coda of {coda_periods} period(s) (period={period}) overlaps the "
            f"prelude ending at {first_shared}"
        )
    loop_size = coda_start - first_shared
    if loop_size % period != 0:
        raise ValueError(
            f"looped block of {loop_size} layers is not a whole number of "
            f"periods (period={period})"
        )

    return LayerPlan(
        num_layers=num_layers,
        period=period,
        full_layers=full_layers,
        first_shared=first_shared,
        prelude=range(0, first_shared),
        loop=range(first_shared, coda_start),
        coda=range(coda_start, num_layers),
    )


def schedule(plan: LayerPlan, n_loops: int) -> list[int]:
    """Return the layer execution order for ``n_loops`` loop iterations.

    ``schedule = prelude + loop * n_loops + coda``.
    """
    if n_loops < 1:
        raise ValueError(f"n_loops must be at least 1, got {n_loops}")
    return list(plan.prelude) + list(plan.loop) * n_loops + list(plan.coda)
