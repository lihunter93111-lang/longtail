"""Verify exported bytes; optionally verify the original files and external assets."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-source', type=Path)
    parser.add_argument('--check-artifacts', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'delivery/source_import_manifest.json').read_text())
    checked = 0
    for row in manifest['files']:
        path = ROOT/row['destination']
        assert path.is_file(), f'Missing exported file: {path}'
        assert sha(path) == row['export_sha256'], f'Export hash mismatch: {path}'
        if args.check_source:
            original = args.check_source/row['source']
            assert sha(original) == row['source_sha256'], f'Source changed: {original}'
        checked += 1
    artifacts = 0
    if args.check_artifacts:
        index = json.loads((ROOT/'delivery/artifact_manifest.json').read_text())
        for row in index['files']:
            path = Path(row['original_path'])
            assert path.stat().st_size == row['bytes'], f'Artifact size mismatch: {path}'
            assert sha(path) == row['sha256'], f'Artifact hash mismatch: {path}'
            artifacts += 1
    print(json.dumps({'export_files_verified': checked,
                      'source_verified': bool(args.check_source),
                      'external_artifacts_verified': artifacts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
