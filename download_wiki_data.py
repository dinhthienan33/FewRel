#!/usr/bin/env python3
"""Download FewRel wiki train/val JSON files into ./data."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

FILES = {
    "train_wiki.json": (
        "https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json"
    ),
    "val_wiki.json": (
        "https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json"
    ),
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  done ({size_mb:.2f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory (default: FewRel/data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    for name, url in FILES.items():
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"Skip (exists): {dest}")
            continue
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to download {name}: {exc}", file=sys.stderr)
            return 1

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
