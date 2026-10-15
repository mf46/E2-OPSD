"""Prepare the two release datasets, with pinned downloads and SHA-256 checks.

The 173-question validation parquet ships with the repository. Only the large
DeepMath training parquet is downloaded. Existing files are verified, never
overwritten. Run with --verify to check files without network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "manifest.json"
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "verl"


def verify(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Size mismatch: {path}; existing file was not changed")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}; existing file was not changed")


def prepare(entry: dict, dst_root: Path, verify_only: bool = False) -> Path:
    destination = dst_root / entry["path"]
    if destination.exists() or verify_only:
        verify(destination, entry)
        return destination
    if entry["distribution"] == "bundled":
        source = DEFAULT_LOCAL_ROOT / entry["path"]
    else:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(
            repo_id=entry["repo_id"], repo_type="dataset",
            revision=entry["revision"], filename=entry["path"],
        ))
    verify(source, entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem; link() refuses to replace an existing
    # file, including when another download finishes after our initial check.
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
            staged = Path(out.name)
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, out)
        verify(staged, entry)
        try:
            os.link(staged, destination)
        except FileExistsError:
            verify(destination, entry)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return destination


def main() -> int:
    entries = json.loads(MANIFEST.read_text())["datasets"]
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Prepare both release datasets (default)")
    selection.add_argument("--dataset", choices=[entry["dataset"] for entry in entries])
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--dst", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--verify", action="store_true", help="Verify existing files offline")
    args = parser.parse_args()
    if args.split and not args.dataset:
        parser.error("--split requires --dataset")
    selected = [entry for entry in entries if not args.dataset or entry["dataset"] == args.dataset]
    if args.split:
        selected = [entry for entry in selected if entry["split"] == args.split]
        if not selected:
            parser.error("The selected dataset does not provide that split")
    for entry in selected:
        try:
            path = prepare(entry, args.dst, args.verify)
            print(f"OK {entry['path']} ({entry['rows']} rows, SHA-256 verified) -> {path}")
        except Exception as exc:
            print(f"FAIL {entry['path']}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
