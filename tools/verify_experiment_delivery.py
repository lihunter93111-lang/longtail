"""Verify the versioned experiment delivery against its SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "docs/release/2026-11-07-integration-manifest.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--working-tree", action="store_true", help="Compare the current files with the historical manifest")
    mode.add_argument("--revision", help="Compare files at this Git revision; defaults to the manifest's own commit")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    revision = args.revision
    if not args.working_tree and revision is None:
        revision = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(args.manifest.resolve().relative_to(root))],
            cwd=root, text=True).strip()
        if not revision:
            parser.error("Manifest has no Git history; select --working-tree or --revision")
    errors = []
    for record in manifest["files"]:
        path = root / record["path"]
        try:
            contents = path.read_bytes() if args.working_tree else subprocess.check_output(
                ["git", "show", "{}:{}".format(revision, record["path"])], cwd=root, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            errors.append({"path": record["path"], "error": "missing"})
            continue
        digest = hashlib.sha256(contents).hexdigest()
        if digest != record["sha256"]:
            errors.append({"path": record["path"], "error": "sha256 mismatch"})
    print(json.dumps({"files": len(manifest["files"]), "revision": revision,
                      "mode": "working-tree" if args.working_tree else "historical-revision",
                      "errors": errors}, ensure_ascii=False))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
