"""Tests for deterministic, offline rendering (Task 17)."""

from __future__ import annotations

from PIL import Image
from playwright.sync_api import Page

import gemmaloop.judge.render as render_module
from gemmaloop.judge.render import extract_html, render_html, render_svg


def test_render_html_red_page_center_is_red(tmp_path):
    html = '<div style="background:#ff0000;width:100vw;height:100vh"></div>'

    result = render_html(html, tmp_path)

    assert result.ok, result.error
    assert result.desktop_png is not None and result.desktop_png.exists()
    assert result.mobile_png is not None and result.mobile_png.exists()
    with Image.open(result.desktop_png) as image:
        center = image.getpixel((image.width // 2, image.height // 2))
    assert center[:3] == (255, 0, 0)


def test_external_image_makes_no_network_request(tmp_path, monkeypatch):
    seen: list[str] = []
    original_on = Page.on

    def spying_on(page, event, handler):
        if event == "request":

            def wrapped(request):
                seen.append(request.url)
                return handler(request)

            return original_on(page, event, wrapped)
        return original_on(page, event, handler)

    monkeypatch.setattr(Page, "on", spying_on)

    html = (
        '<body style="margin:0;background:#fff">'
        '<img src="https://example.com/x.png" alt="external">'
        '<p>visible</p></body>'
    )

    result = render_html(html, tmp_path)

    assert result.ok, result.error
    assert [url for url in seen if url.startswith(("http://", "https://"))] == []
    assert result.desktop_png is not None and result.desktop_png.exists()


def test_empty_html_is_blank(tmp_path):
    result = render_html("   \n\t ", tmp_path)

    assert result.ok is False
    assert result.error
    assert result.desktop_png is None
    assert result.mobile_png is None


def test_non_empty_but_blank_html_is_rejected(tmp_path):
    result = render_html("<html><body></body></html>", tmp_path)

    assert result.ok is False
    assert result.error
    assert result.desktop_png is None
    assert result.mobile_png is None


def test_extract_html_takes_last_fenced_block():
    completion = (
        "```html\n<div>first</div>\n```\n\n"
        "some prose\n\n"
        "```html\n<div>second</div>\n```"
    )

    assert extract_html(completion) == "<div>second</div>"


def test_extract_html_drops_reasoning_first():
    completion = (
        "<|channel>thought\nMaybe ```html\n<div>wrong</div>\n```\n<channel|>"
        "```html\n<div>right</div>\n```"
    )

    assert extract_html(completion) == "<div>right</div>"


def test_extract_html_falls_back_to_last_html_document():
    completion = (
        "noise <html><body>one</body></html> more "
        "<HTML><body>two</body></HTML>"
    )

    assert extract_html(completion) == "<HTML><body>two</body></HTML>"


def test_extract_html_returns_none_without_html():
    assert extract_html("") is None
    assert extract_html("just prose, no markup") is None


def test_render_svg_writes_png(tmp_path):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect width="100" height="100" fill="#00ff00"/></svg>'
    )
    out_png = tmp_path / "art.png"

    result = render_svg(svg, out_png)

    assert result.ok, result.error
    assert result.desktop_png == out_png and out_png.exists()
    with Image.open(out_png) as image:
        center = image.getpixel((image.width // 2, image.height // 2))
    assert center[:3] == (0, 255, 0)


RED_PAGE = '<div style="background:#ff0000;width:100vw;height:100vh"></div>'
GREEN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<rect width="100" height="100" fill="#00ff00"/></svg>'
)


def test_html_blank_check_failure_is_reported_not_raised(tmp_path, monkeypatch):
    def broken(path):
        raise RuntimeError("cannot decode png")

    monkeypatch.setattr(render_module, "_is_blank", broken)

    result = render_html(RED_PAGE, tmp_path)

    assert result.ok is False
    assert result.error is not None and "cannot decode png" in result.error
    assert result.desktop_png is None and result.mobile_png is None


def test_svg_blank_check_failure_is_reported_not_raised(tmp_path, monkeypatch):
    def broken(path):
        raise ImportError("PIL is not available")

    monkeypatch.setattr(render_module, "_is_blank", broken)

    result = render_svg(GREEN_SVG, tmp_path / "art.png")

    assert result.ok is False
    assert result.error is not None and "PIL is not available" in result.error
    assert result.desktop_png is None


def test_html_invalid_out_dir_is_reported_not_raised(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    result = render_html(RED_PAGE, blocked)

    assert result.ok is False
    assert result.error
    assert result.desktop_png is None and result.mobile_png is None


def test_svg_invalid_out_dir_is_reported_not_raised(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    result = render_svg(GREEN_SVG, blocked / "art.png")

    assert result.ok is False
    assert result.error
    assert result.desktop_png is None
