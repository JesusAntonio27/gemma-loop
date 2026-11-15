"""Deterministic, offline rendering of model-generated HTML and SVG.

The vision judge compares screenshots, so the same markup must produce the same
pixels on every run: fixed viewport and device scale factor, a frozen locale,
time zone, colour scheme and reduced-motion preference, CSS animations disabled
and — most important — no network access at all.  Any external request would
make the screenshots depend on the machine and the moment of the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

HTTP_PREFIXES = ("http://", "https://")

# Reasoning traces are emitted as <|channel>thought ... <channel|> blocks.
_THOUGHT_CHANNEL = re.compile(
    r"<\|channel>thought\b.*?<channel\|>", re.DOTALL | re.IGNORECASE
)
_THOUGHT_TAGS = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = "</think>"
_FENCED_HTML = re.compile(
    r"```[ \t]*html[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
)
_HTML_DOCUMENT = re.compile(r"<html\b[^>]*>.*?</html\s*>", re.IGNORECASE | re.DOTALL)

# Absolute URLs in attributes and CSS would leave the page even with
# ``offline=True`` (the request is aborted, but it is still a request).  They
# are neutralised to an empty data URL before the markup ever reaches Chromium.
_ATTR_URL = re.compile(
    r"(?P<prefix>\b(?:src|href|srcset|poster|data|action|formaction|cite|background)"
    r"\s*=\s*)"
    r"(?:(?P<quote>[\"'])(?:https?:)?//.*?(?P=quote)|(?:https?:)?//[^\s>]+)",
    re.IGNORECASE | re.DOTALL,
)
_CSS_URL = re.compile(
    r"url\(\s*(?P<quote>[\"']?)(?:https?:)?//.*?(?P=quote)\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_CSS_IMPORT = re.compile(
    r"(@import\s+)(?P<quote>[\"'])(?:https?:)?//.*?(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)

_READY_SENTINEL = "window.__RENDER_READY__ === true"


@dataclass
class RenderResult:
    """Outcome of a render.

    ``desktop_png``/``mobile_png`` are ``None`` unless the capture succeeded and
    was not blank.
    """

    ok: bool
    desktop_png: Path | None
    mobile_png: Path | None
    error: str | None


def _strip_reasoning(text: str) -> str:
    """Remove the chain-of-thought preamble and return the visible answer."""
    text = _THOUGHT_CHANNEL.sub("", text)
    text = _THOUGHT_TAGS.sub("", text)
    lowered = text.lower()
    if _THINK_CLOSE in lowered:
        end = lowered.rfind(_THINK_CLOSE) + len(_THINK_CLOSE)
        text = text[end:]
    # Malformed/unclosed channel markup: the answer starts after the last close.
    if "<channel|>" in text:
        text = text.rsplit("<channel|>", 1)[1]
    return text


def extract_html(completion: str) -> str | None:
    """Return the HTML contained in a model completion, or ``None``.

    The last ```` ```html ```` fenced block wins; without one, the last complete
    ``<html>…</html>`` document is used.  Reasoning is stripped first so a
    ``html`` block quoted inside a thought trace is never rendered.
    """
    if not completion:
        return None

    text = _strip_reasoning(completion)

    blocks = list(_FENCED_HTML.finditer(text))
    if blocks:
        return blocks[-1].group(1).strip()

    documents = list(_HTML_DOCUMENT.finditer(text))
    if documents:
        return documents[-1].group(0).strip()

    return None


def _neutralise_external_urls(markup: str) -> str:
    """Replace absolute http(s) references so Chromium never requests them."""

    def _attribute(match: re.Match[str]) -> str:
        quote = match.group("quote") or ""
        return f"{match.group('prefix')}{quote}data:,{quote}"

    markup = _ATTR_URL.sub(_attribute, markup)
    markup = _CSS_URL.sub('url("data:,")', markup)
    markup = _CSS_IMPORT.sub(r'\1"data:,"', markup)
    return markup


def _abort_external(route) -> None:
    """Abort every http(s) request; serve inline resources normally."""
    url = route.request.url
    if url.startswith(HTTP_PREFIXES):
        route.abort()
    else:
        route.continue_()


def _context_options(viewport: tuple[int, int]) -> dict:
    return {
        "viewport": {"width": viewport[0], "height": viewport[1]},
        "device_scale_factor": 1,
        "color_scheme": "light",
        "reduced_motion": "reduce",
        "locale": "en-US",
        "timezone_id": "UTC",
        "offline": True,
        "service_workers": "block",
    }


def _wait_until_ready(page: Page, timeout_ms: int) -> None:
    page.evaluate("document.fonts.ready")
    try:
        page.wait_for_function(_READY_SENTINEL, timeout=min(3000, timeout_ms))
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(300)


def _screenshot(page: Page, path: Path) -> None:
    page.screenshot(
        path=str(path),
        full_page=True,
        animations="disabled",
        caret="hide",
        scale="css",
    )


def _is_blank(path: Path) -> bool:
    """Return ``True`` when every pixel of the capture is (near) white."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
    return all(low >= 250 for low, _high in extrema)


def _record_requests(page: Page, sink: list[str]) -> None:
    """Attach a request listener (also the seam used to test the offline rule)."""

    def _handler(request) -> None:
        sink.append(request.url)

    page.on("request", _handler)


def render_html(
    html: str,
    out_dir: Path,
    desktop: tuple[int, int] = (1440, 900),
    mobile: tuple[int, int] = (390, 844),
    timeout_ms: int = 15000,
) -> RenderResult:
    """Render ``html`` to ``desktop.png`` and ``mobile.png`` under ``out_dir``.

    The browser is headless Chromium with networking disabled.  Returns
    ``ok=False`` (with ``error`` set) if navigation fails, no screenshot can be
    taken or the capture is blank.
    """
    out_dir = Path(out_dir)
    desktop_png = out_dir / "desktop.png"
    mobile_png = out_dir / "mobile.png"

    if not html or not html.strip():
        return RenderResult(False, None, None, "empty html")

    seen: list[str] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(**_context_options(desktop))
                context.route("**/*", _abort_external)
                page = context.new_page()
                page.set_default_timeout(timeout_ms)
                _record_requests(page, seen)
                page.set_content(
                    _neutralise_external_urls(html),
                    wait_until="load",
                    timeout=timeout_ms,
                )
                _wait_until_ready(page, timeout_ms)
                _screenshot(page, desktop_png)

                page.set_viewport_size(
                    {"width": mobile[0], "height": mobile[1]}
                )
                page.wait_for_timeout(300)
                _screenshot(page, mobile_png)
            finally:
                browser.close()

        if not desktop_png.exists() or not mobile_png.exists():
            return RenderResult(False, None, None, "screenshot was not written")
        if _is_blank(desktop_png):
            return RenderResult(False, None, None, "blank desktop screenshot")
        if _is_blank(mobile_png):
            return RenderResult(False, None, None, "blank mobile screenshot")
    except Exception as exc:  # noqa: BLE001 - surface every failure to the judge
        return RenderResult(False, None, None, f"{type(exc).__name__}: {exc}")

    return RenderResult(True, desktop_png, mobile_png, None)


def render_svg(svg: str, out_png: Path) -> RenderResult:
    """Render an SVG document into ``out_png`` with the same fixed environment."""
    out_png = Path(out_png)

    if not svg or not svg.strip():
        return RenderResult(False, None, None, "empty svg")

    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        html = (
            "<!doctype html><meta charset='utf-8'>"
            "<body style='margin:0;background:#fff'>"
            f"{_neutralise_external_urls(svg)}</body>"
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(**_context_options((1440, 900)))
                context.route("**/*", _abort_external)
                page = context.new_page()
                page.set_default_timeout(15000)
                page.set_content(html, wait_until="load")
                _wait_until_ready(page, 15000)
                page.evaluate(
                    "document.querySelectorAll("
                    "'animate, animateTransform, animateMotion, set'"
                    ").forEach(element => element.remove())"
                )
                locator = page.locator("svg")
                if locator.count() == 0:
                    return RenderResult(False, None, None, "no <svg> element")
                locator.first.screenshot(
                    path=str(out_png), animations="disabled", caret="hide"
                )
            finally:
                browser.close()

        if not out_png.exists():
            return RenderResult(False, None, None, "screenshot was not written")
        if _is_blank(out_png):
            return RenderResult(False, None, None, "blank svg screenshot")
    except Exception as exc:  # noqa: BLE001
        return RenderResult(False, None, None, f"{type(exc).__name__}: {exc}")

    return RenderResult(True, out_png, None, None)
