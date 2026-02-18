#!/usr/bin/env python3
"""Fetch XKCD images via the xkcd JSON API, preferring 2x versions.

Usage:
    ./scripts/fetch_xkcd_2x.py [--lecture 02 | --outdir lectures/02/media] <id[:filename]> [...]
Example:
    ./scripts/fetch_xkcd_2x.py --lecture 07 1619 2237:xkcd_ai_hiring.png

Flow per comic:
1) JSON API: https://xkcd.com/<id>/info.0.json → get image URL
2) Try 2x variant (insert _2x before extension)
3) Fall back to standard resolution if 2x unavailable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_target_dir(outdir: Path | None, lecture: str | None) -> Path:
    if outdir is not None:
        return outdir
    if lecture is not None:
        return REPO_ROOT / "lectures" / lecture / "media"
    return Path.cwd()


def _get_comic_info(comic_id: int) -> dict:
    url = f"https://xkcd.com/{comic_id}/info.0.json"
    with urllib.request.urlopen(url) as resp:  # nosec B310
        return json.loads(resp.read())


def _make_2x_url(img_url: str) -> str:
    stem, ext = img_url.rsplit(".", 1)
    return f"{stem}_2x.{ext}"


def _url_exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as resp:  # nosec B310
            return resp.status == 200
    except Exception:
        return False


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:  # nosec B310
        data = resp.read()
    dest.write_bytes(data)
    print(f"  Saved {dest.name} ({len(data):,} bytes)")


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def parse_comic_specs(specs: list[str]) -> list[tuple[int, str | None]]:
    parsed = []
    for spec in specs:
        parts = spec.split(":")
        comic_id = int(parts[0])
        filename = parts[1] if len(parts) > 1 else None
        parsed.append((comic_id, filename))
    return parsed


def _parse_cli(argv: list[str]) -> tuple[list[str], Path | None, str | None]:
    parser = argparse.ArgumentParser(
        description="Fetch XKCD images via the JSON API (prefers 2x versions)."
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory for downloaded images.",
    )
    parser.add_argument(
        "--lecture", default=None,
        help="Two-digit lecture folder (e.g., 02) → lectures/02/media.",
    )
    parser.add_argument(
        "comics", nargs="+",
        help="Comic specs: id[:filename.png] (title auto-slugified if no filename given)",
    )
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
        print(f"Target: {target_dir}", file=sys.stderr)

    comics = parse_comic_specs(specs)
    for comic_id, filename_override in comics:
        info = _get_comic_info(comic_id)
        title = info["safe_title"]
        img_url = info["img"]
        ext = img_url.rsplit(".", 1)[-1]

        filename = filename_override or f"xkcd_{_slugify(title)}.{ext}"

        url_2x = _make_2x_url(img_url)
        if _url_exists(url_2x):
            print(f"#{comic_id} \"{title}\" → 2x")
            _download(url_2x, target_dir / filename)
        else:
            print(f"#{comic_id} \"{title}\" → 1x (no 2x available)")
            _download(img_url, target_dir / filename)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as err:  # pragma: no cover
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
