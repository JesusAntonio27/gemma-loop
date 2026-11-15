"""Blind human rating of rendered design pairs.

The author rates the same pairs as the AI judge, but never sees which variant
is which: for every pair the display order is drawn from ``--seed`` and the
two candidates are shown only as ``A`` and ``B``.  Neither the pair id nor the
source paths (which contain variant names such as ``O-H`` or ``L``) ever reach
the browser: images are copied under opaque names and the page refers to
opaque per-pair tokens.

Usage::

    uv run python scripts/human_rate.py --pairs pairs.jsonl --out human.jsonl

``pairs.jsonl`` has one object per line.  ``x`` and ``y`` are the canonical
candidates (``X`` = O-H, ``Y`` = L in Task 20); each may be a single path, a
list of paths (desktop, mobile) or an object with ``desktop``/``mobile`` (or
``desktop_png``/``mobile_png``) keys::

    {"pair_id": "brief-001", "x": {"desktop": "a.png", "mobile": "a-m.png"},
     "y": ["b.png", "b-m.png"]}

One self-contained HTML page is written per batch under ``--html-dir`` (by
default ``<out>_pages``).  A tiny HTTP server bound to ``127.0.0.1`` serves
those pages and writes each answer immediately; no external network is used.
The answer line keeps the clicked display label and maps it back to canonical
``X``/``Y``/``tie`` for :func:`gemmaloop.judge.agreement.agreement`::

    {"pair_id": "...", "choice": "A"|"B"|"tie", "winner": "X"|"Y"|"tie",
     "display": {"A": "X", "B": "Y"}, "batch": 0, "token": "p0000", "ts": "..."}
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import shutil
import sys
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_BATCH_SIZE = 20
DEFAULT_SEED = 0
CHOICES = ("A", "B", "tie")
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
HOST = "127.0.0.1"


@dataclass(frozen=True)
class RatingPair:
    """One pair of canonical candidates (``X`` and ``Y``)."""

    pair_id: str
    x: tuple[Path, ...]
    y: tuple[Path, ...]


@dataclass(frozen=True)
class DisplayedPair:
    """A pair as the browser sees it.

    ``order[0]`` is the canonical candidate shown as ``A`` and ``order[1]``
    the one shown as ``B``.  ``images`` maps the lowercase display slot
    (``"a"``/``"b"``) to the URLs of its captures.
    """

    token: str
    pair_id: str
    order: tuple[str, str]
    images: dict[str, tuple[str, ...]]

    @property
    def display(self) -> dict[str, str]:
        return {"A": self.order[0], "B": self.order[1]}


@dataclass(frozen=True)
class Batch:
    """A written rating page and the pairs it contains."""

    index: int
    pairs: tuple[DisplayedPair, ...]
    html_path: Path
    assets: dict[str, Path]


def _candidate_paths(value: object, pair_id: str, side: str) -> tuple[Path, ...]:
    if isinstance(value, str):
        raw: list[object] = [value]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    elif isinstance(value, dict):
        raw = []
        for keys in (("desktop", "mobile"), ("desktop_png", "mobile_png")):
            if any(key in value for key in keys):
                raw = [value[key] for key in keys if key in value]
                break
        else:
            images = value.get("images")
            if isinstance(images, str):
                raw = [images]
            elif isinstance(images, (list, tuple)):
                raw = list(images)
            else:
                raise ValueError(
                    f"pair {pair_id!r}: candidate {side!r} has no images"
                )
    else:
        raise ValueError(f"pair {pair_id!r}: candidate {side!r} is not image data")

    if not raw:
        raise ValueError(f"pair {pair_id!r}: candidate {side!r} has no images")

    paths = []
    for item in raw:
        if not isinstance(item, (str, Path)):
            raise ValueError(
                f"pair {pair_id!r}: candidate {side!r} contains a non-path"
            )
        paths.append(Path(item))
    return tuple(paths)


def load_pairs(path: str | Path) -> list[RatingPair]:
    """Read and normalise the pairs JSONL."""
    path = Path(path)
    pairs: list[RatingPair] = []
    seen: set[str] = set()

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from None
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: a pair must be a JSON object")

        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"{path}:{line_number}: missing string pair_id")
        if pair_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate pair_id {pair_id!r}")
        seen.add(pair_id)

        for side in ("x", "y"):
            if side not in record:
                raise ValueError(
                    f"{path}:{line_number}: pair {pair_id!r} has no {side!r}"
                )
        pairs.append(
            RatingPair(
                pair_id=pair_id,
                x=_candidate_paths(record["x"], pair_id, "x"),
                y=_candidate_paths(record["y"], pair_id, "y"),
            )
        )

    if not pairs:
        raise ValueError(f"{path} contains no pairs")
    return pairs


def prepare_batches(
    pairs: list[RatingPair],
    html_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Batch]:
    """Write one blind HTML page per batch and return their metadata.

    The display order is drawn pair by pair from ``seed`` and recorded in the
    returned :class:`DisplayedPair` objects; it is never derived from a variant
    name.  Captures are copied under opaque, lowercase filenames.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pairs = list(pairs)
    if not pairs:
        raise ValueError("prepare_batches needs at least one pair")

    html_dir = Path(html_dir)
    html_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    batches: list[Batch] = []

    for start in range(0, len(pairs), batch_size):
        index = len(batches)
        batch_dir = html_dir / f"batch_{index:03d}"
        asset_dir = batch_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)

        displayed: list[DisplayedPair] = []
        assets: dict[str, Path] = {}
        for offset, pair in enumerate(pairs[start : start + batch_size]):
            token = f"p{start + offset:04d}"
            order = ("X", "Y") if rng.random() < 0.5 else ("Y", "X")
            candidates = {"X": pair.x, "Y": pair.y}
            images: dict[str, tuple[str, ...]] = {}

            for slot, canonical in zip(("A", "B"), order):
                urls = []
                for image_index, source in enumerate(candidates[canonical]):
                    suffix = source.suffix.lower()
                    if suffix not in ALLOWED_SUFFIXES:
                        raise ValueError(
                            f"pair {pair.pair_id!r}: unsupported image type "
                            f"{source.suffix!r}"
                        )
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    filename = f"{token}_{slot.lower()}_{image_index}{suffix}"
                    target = asset_dir / filename
                    shutil.copyfile(source, target)
                    assets[filename] = target
                    urls.append(f"/assets/{index:03d}/{filename}")
                images[slot.lower()] = tuple(urls)

            displayed.append(
                DisplayedPair(
                    token=token,
                    pair_id=pair.pair_id,
                    order=order,
                    images=images,
                )
            )

        next_url = f"/batch/{index + 1:03d}" if index + 1 < total_batches else ""
        html = _render_page(displayed, next_url=next_url)
        html_path = batch_dir / "index.html"
        html_path.write_text(html, encoding="utf-8")
        batches.append(
            Batch(
                index=index,
                pairs=tuple(displayed),
                html_path=html_path,
                assets=assets,
            )
        )

    return batches


def answer_record(
    displayed: DisplayedPair,
    choice: str,
    *,
    batch_index: int,
    timestamp: str | None = None,
) -> dict:
    """Map a clicked ``A``/``B``/``tie`` back to the canonical ``X``/``Y``."""
    if choice not in CHOICES:
        raise ValueError(f"unknown choice {choice!r}; expected A, B or tie")
    if choice == "tie":
        winner = "tie"
    else:
        winner = displayed.order[0] if choice == "A" else displayed.order[1]
    return {
        "pair_id": displayed.pair_id,
        "choice": choice,
        "winner": winner,
        "display": displayed.display,
        "batch": batch_index,
        "token": displayed.token,
        "ts": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def _render_page(pairs: list[DisplayedPair], *, next_url: str) -> str:
    sections = []
    for pair in pairs:
        candidates = []
        for slot, label in (("a", "A"), ("b", "B")):
            figures = []
            for view, url in enumerate(pair.images[slot], start=1):
                figures.append(
                    f'<figure><img src="{url}" '
                    f'alt="candidate {label} view {view}">'
                    f"<figcaption>candidate {label} . view {view}</figcaption></figure>"
                )
            candidates.append(
                f'<div class="candidate"><h2>{label}</h2>'
                f'<div class="views">{"".join(figures)}</div></div>'
            )
        sections.append(
            f'<section class="pair" data-token="{pair.token}" hidden>'
            f'{"".join(candidates)}'
            '<div class="actions">'
            '<button type="button" data-choice="A">A is better</button>'
            '<button type="button" data-choice="B">B is better</button>'
            '<button type="button" data-choice="tie">Tie</button>'
            "</div></section>"
        )

    meta = json.dumps({"next": next_url, "total": len(pairs)}, ensure_ascii=False)
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind design rating</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: system-ui, sans-serif; background: #f4f4f5; color: #18181b; }}
header {{ position: sticky; top: 0; z-index: 10; display: flex; justify-content: space-between;
         align-items: baseline; gap: 1rem; padding: .9rem 1.4rem; background: #18181b; color: #fafafa; }}
header strong {{ font-size: 1rem; }}
header span {{ font-size: .85rem; opacity: .8; }}
main {{ padding: 1.4rem; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; align-items: start; }}
.candidate {{ min-width: 0; background: #fff; border: 1px solid #d4d4d8; border-radius: .6rem; padding: .8rem; }}
.candidate h2 {{ margin: 0 0 .6rem; font-size: 1rem; text-align: center; }}
.views {{ display: grid; gap: .8rem; }}
figure {{ margin: 0; }}
img {{ display: block; width: 100%; height: auto; max-height: 70vh; object-fit: contain;
      border: 1px solid #e4e4e7; background: #fff; }}
figcaption {{ margin-top: .25rem; font-size: .75rem; color: #52525b; text-align: center; }}
.actions {{ grid-column: 1 / -1; display: flex; justify-content: center; gap: .8rem; flex-wrap: wrap;
           position: sticky; bottom: 0; padding: .8rem; background: #f4f4f5; }}
button {{ font: inherit; font-size: .95rem; padding: .6rem 1.1rem; border-radius: .45rem;
         border: 1px solid #52525b; background: #fff; cursor: pointer; }}
button:hover {{ background: #e4e4e7; }}
button:focus-visible {{ outline: 3px solid #2563eb; outline-offset: 2px; }}
[hidden] {{ display: none !important; }}
</style>
<header><strong>Blind design rating</strong><span id="progress"></span></header>
<main id="app">
{"".join(sections)}
</main>
<script>
const meta = {meta};
const sections = Array.from(document.querySelectorAll("section.pair"));
const progress = document.getElementById("progress");
let index = 0;
let locked = false;

function show() {{
  sections.forEach(function (section, position) {{
    section.hidden = position !== index;
  }});
  progress.textContent = "pair " + (index + 1) + " / " + sections.length;
}}

function send(choice) {{
  if (locked) {{ return; }}
  locked = true;
  const section = sections[index];
  fetch("/answer", {{
    method: "post",
    headers: {{"content-type": "application/json"}},
    body: JSON.stringify({{token: section.dataset.token, choice: choice}})
  }}).then(function (response) {{
    if (!response.ok) {{ throw new Error("save failed"); }}
    return response.json();
  }}).then(function () {{
    locked = false;
    index = index + 1;
    if (index < sections.length) {{ show(); }} else {{ finish(); }}
  }}).catch(function () {{
    locked = false;
    window.alert("The answer could not be saved. Is the rating server still running?");
  }});
}}

document.querySelectorAll("button[data-choice]").forEach(function (button) {{
  button.onclick = function () {{ send(button.dataset.choice); }};
}});

document.onkeydown = function (event) {{
  if (event.key === "a" || event.key === "A") {{ send("A"); }}
  if (event.key === "b" || event.key === "B") {{ send("B"); }}
  if (event.key === "t" || event.key === "T") {{ send("tie"); }}
}};

function finish() {{
  progress.textContent = "batch complete";
  const main = document.getElementById("app");
  main.textContent = "";
  const heading = document.createElement("h2");
  heading.textContent = "This batch is complete";
  const note = document.createElement("p");
  main.appendChild(heading);
  main.appendChild(note);
  if (meta.next) {{
    note.textContent = "Continuing with the next batch...";
    window.setTimeout(function () {{ window.location.href = meta.next; }}, 400);
  }} else {{
    note.textContent = "All pairs are rated. You can close this tab.";
  }}
}}

show();
</script>
"""


class _RatingState:
    """Answers, tokens and output file shared by the HTTP handler threads."""

    def __init__(self, batches: list[Batch], out_path: Path):
        self.batches = batches
        self.tokens: dict[str, DisplayedPair] = {}
        self.batch_of: dict[str, int] = {}
        for batch in batches:
            for pair in batch.pairs:
                self.tokens[pair.token] = pair
                self.batch_of[pair.token] = batch.index
        self.answered: set[str] = set()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self._out = out_path.open("a", encoding="utf-8")

    def record(self, token: object, choice: object) -> int:
        with self.lock:
            if not isinstance(token, str) or token not in self.tokens:
                raise KeyError("unknown token")
            if choice not in CHOICES:
                raise ValueError("unknown choice")
            if token not in self.answered:
                displayed = self.tokens[token]
                record = answer_record(
                    displayed, choice, batch_index=self.batch_of[token]
                )
                self._out.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._out.flush()
                self.answered.add(token)
            remaining = len(self.tokens) - len(self.answered)
            if remaining == 0:
                self.done.set()
            return remaining

    def close(self) -> None:
        self._out.close()


class _RatingHandler(BaseHTTPRequestHandler):
    server_version = "GemmaLoopRating/1.0"

    @property
    def state(self) -> _RatingState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = urlsplit(self.path).path
        if path in ("", "/"):
            if not self.state.batches:
                self._send(404, b"no pairs", "text/plain; charset=utf-8")
                return
            self.send_response(302)
            self.send_header("location", "/batch/000")
            self.send_header("content-length", "0")
            self.end_headers()
            return

        if path.startswith("/batch/"):
            try:
                index = int(path.rsplit("/", 1)[1])
                batch = self.state.batches[index]
            except (ValueError, IndexError):
                self._send(404, b"unknown batch", "text/plain; charset=utf-8")
                return
            self._send(200, batch.html_path.read_bytes(), "text/html; charset=utf-8")
            return

        if path.startswith("/assets/"):
            parts = path.split("/")
            if len(parts) != 4:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            try:
                index = int(parts[2])
                batch = self.state.batches[index]
            except (ValueError, IndexError):
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            source = batch.assets.get(parts[3])
            if source is None:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            content_type = (
                mimetypes.guess_type(parts[3])[0] or "application/octet-stream"
            )
            self._send(200, source.read_bytes(), content_type)
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if urlsplit(self.path).path != "/answer":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000:
            self._send(400, b"bad length", "text/plain; charset=utf-8")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be an object")
            remaining = self.state.record(payload.get("token"), payload.get("choice"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError):
            self._json(400, {"ok": False})
            return
        self._json(200, {"ok": True, "remaining": remaining})


def serve(
    batches: list[Batch],
    out_path: str | Path,
    *,
    port: int = 0,
    open_browser: bool = True,
    timeout: float | None = None,
) -> int:
    """Serve the rating pages on ``127.0.0.1`` until every pair is rated."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = _RatingState(batches, out_path)
    server = ThreadingHTTPServer((HOST, port), _RatingHandler)
    server.state = state  # type: ignore[attr-defined]
    server.daemon_threads = True
    url = f"http://{HOST}:{server.server_address[1]}/"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"rating pages: {batches[0].html_path.parent.parent}")
    print(f"open {url} and rate {len(state.tokens)} pairs; answers -> {out_path}")

    try:
        if open_browser:
            webbrowser.open(url)
        finished = state.done.wait(timeout)
    except KeyboardInterrupt:
        finished = False
        print("\ninterrupted", file=sys.stderr)
    finally:
        server.shutdown()
        server.server_close()
        state.close()

    rated = len(state.answered)
    print(f"{rated}/{len(state.tokens)} pairs rated")
    return 0 if finished or rated == len(state.tokens) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, type=Path, help="pairs JSONL")
    parser.add_argument("--out", required=True, type=Path, help="answers JSONL")
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="where to write the rating pages (default: <out>_pages)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a non-empty answers file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds to wait (default: until every pair is rated)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists() and args.out.stat().st_size > 0 and not args.overwrite:
        print(
            f"error: {args.out} already has answers; pass --overwrite to replace it",
            file=sys.stderr,
        )
        return 2

    try:
        pairs = load_pairs(args.pairs)
        html_dir = args.html_dir or args.out.with_name(args.out.stem + "_pages")
        batches = prepare_batches(
            pairs, html_dir, seed=args.seed, batch_size=args.batch_size
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.write_text("", encoding="utf-8")
    try:
        return serve(
            batches,
            args.out,
            port=args.port,
            open_browser=not args.no_browser,
            timeout=args.timeout,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
