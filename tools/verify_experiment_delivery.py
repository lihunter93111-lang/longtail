"""Verify the versioned experiment delivery against its SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "docs/release/2026-11-07-integration-manifest.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            errors.append({"path": record["path"], "error": "missing"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            errors.append({"path": record["path"], "error": "sha256 mismatch"})
    print(json.dumps({"files": len(manifest["files"]), "errors": errors}, ensure_ascii=False))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
