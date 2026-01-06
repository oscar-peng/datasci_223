#!/usr/bin/env python3
"""Fetch XKCD images via explainxkcd file pages (2x originals when available).

Usage:
    ./fetch_xkcd_2x.py <id:Slug[:filename]> [...]
Example:
    ./fetch_xkcd_2x.py 1597:Git 1722:Debugging:xkcd_debugging.png

Flow per comic:
1) Article page: https://www.explainxkcd.com/wiki/index.php/<num>:_<Slug>
2) Grab the first image link (/wiki/index.php/File:...)
3) File page: follow to /wiki/images/... and download the linked file (highest res).
"""
from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent.parent / "lectures" / "01" / "media"


def fetch(url: str) -> str:
    with urllib.request.urlopen(url) as resp:  # nosec B310
        return resp.read().decode("utf-8", errors="ignore")


def article_to_file_page(article_html: str, article_url: str) -> str:
    match = re.search(r'href="(/wiki/index.php/File:[^"]+)"[^>]*class="image"', article_html)
    if not match:
        raise RuntimeError(f"Could not find file page link on {article_url}")
    rel = match.group(1)
    return urllib.parse.urljoin(article_url, rel)


def file_page_to_image(file_html: str, file_url: str) -> str:
    match = re.search(r'<div class="fullMedia"><a href="([^"]+)"', file_html)
    if not match:
        raise RuntimeError(f"Could not find fullMedia link on {file_url}")
    rel = match.group(1)
    return urllib.parse.urljoin(file_url, rel)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:  # nosec B310
        data = resp.read()
    dest.write_bytes(data)
    print(f"Saved {dest} ({len(data)} bytes) from {url}")


def parse_args(argv: list[str]) -> dict[str, tuple[str, str]]:
    """Allow overrides via args like 1597:Git[:custom_filename]."""
    if not argv:
        raise SystemExit("Usage: ./fetch_xkcd_2x.py <id:Slug[:filename]> [...]")
    parsed: dict[str, tuple[str, str]] = {}
    for arg in argv:
        parts = arg.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid comic spec '{arg}', expected id:Slug[:filename]")
        num, slug, *rest = parts
        filename = rest[0] if rest else f"xkcd_{slug.lower()}.png"
        parsed[num] = (slug, filename)
    return parsed


def main(argv: list[str]) -> None:
    comics = parse_args(argv)
    for number, (slug, filename) in comics.items():
        article_url = f"https://www.explainxkcd.com/wiki/index.php/{number}:_{slug}"
        article_html = fetch(article_url)
        file_page_url = article_to_file_page(article_html, article_url)
        file_html = fetch(file_page_url)
        image_url = file_page_to_image(file_html, file_page_url)
        download(image_url, TARGET_DIR / filename)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as err:  # pragma: no cover
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
