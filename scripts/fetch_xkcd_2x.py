#!/usr/bin/env python3
"""Fetch XKCD images via explainxkcd file pages (2x originals when available).

Usage:
    ./scripts/fetch_xkcd_2x.py [--lecture 02 | --outdir lectures/02/media] <id:Slug[:filename]> [...]
Example:
    ./scripts/fetch_xkcd_2x.py --lecture 02 1597:Git 1722:Debugging:xkcd_debugging.png

Flow per comic:
1) Article page: https://www.explainxkcd.com/wiki/index.php/<num>:_<Slug>
2) Grab the first image link (/wiki/index.php/File:...)
3) File page: follow to /wiki/images/... and download the linked file (highest res).
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_target_dir(outdir: Path | None, lecture: str | None) -> Path:
    if outdir is not None:
        return outdir

    if lecture is not None:
        return REPO_ROOT / "lectures" / lecture / "media"

    return Path.cwd()


def fetch(url: str) -> str:
    with urllib.request.urlopen(url) as resp:  # nosec B310
        return resp.read().decode("utf-8", errors="ignore")


def article_to_file_page(article_html: str, article_url: str) -> str:
    match = re.search(
        r'href="(/wiki/index.php/File:[^"]+)"[^>]*class="image"', article_html
    )
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


def parse_comic_specs(specs: list[str]) -> dict[str, tuple[str, str]]:
    """Allow overrides via args like 1597:Git[:custom_filename]."""
    parsed: dict[str, tuple[str, str]] = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid comic spec '{spec}', expected id:Slug[:filename]"
            )
        num, slug, *rest = parts
        filename = rest[0] if rest else f"xkcd_{slug.lower()}.png"
        parsed[num] = (slug, filename)
    return parsed


def _parse_cli(argv: list[str]) -> tuple[list[str], Path | None, str | None]:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch XKCD images via explainxkcd file pages (prefers 2x originals)."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory for downloaded images. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--lecture",
        default=None,
        help=("Two-digit lecture folder (e.g., 02) to write into `lectures/02/media`."),
    )
    parser.add_argument("comics", nargs="+", help="Comic specs: id:Slug[:filename]")

    ns = parser.parse_args(argv)

    lecture = ns.lecture
    if lecture is not None:
        if lecture.isdigit():
            lecture = f"{int(lecture):02d}"
        if not re.fullmatch(r"\d\d", lecture):
            raise SystemExit("--lecture must be a two-digit value like 02")

    return ns.comics, ns.outdir, lecture


def main(argv: list[str]) -> None:
    specs, outdir, lecture = _parse_cli(argv)
    target_dir = _resolve_target_dir(outdir=outdir, lecture=lecture)

    if outdir is None and lecture is None:
        print(
            f"Defaulting output directory to {target_dir}. "
            "Use --lecture 02 or --outdir lectures/02/media to target a lecture.",
            file=sys.stderr,
        )
    else:
        print(f"Writing images to {target_dir}", file=sys.stderr)

    comics = parse_comic_specs(specs)
    for number, (slug, filename) in comics.items():
        article_url = f"https://www.explainxkcd.com/wiki/index.php/{number}:_{slug}"
        article_html = fetch(article_url)
        file_page_url = article_to_file_page(article_html, article_url)
        file_html = fetch(file_page_url)
        image_url = file_page_to_image(file_html, file_page_url)
        download(image_url, target_dir / filename)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as err:  # pragma: no cover
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
