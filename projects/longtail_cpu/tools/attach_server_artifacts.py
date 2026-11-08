"""Attach explicitly supplied local historical artifacts without copying large files."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def link(source, destination):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f'Existing mapping differs: {destination}')
    elif destination.exists():
        raise FileExistsError(f'Will not replace existing path: {destination}')
    else:
        destination.symlink_to(source.resolve(), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--datasets', required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    datasets = args.datasets.resolve()
    if source == ROOT or not (source/'runs').is_dir() or not datasets.is_dir():
        raise ValueError('Supply the original checkout with runs and the existing datasets root')
    run_root = ROOT/'runs'
    run_root.mkdir(exist_ok=True)
    mapped = []
    for suite in sorted((source/'runs').iterdir()):
        if suite.is_dir():
            link(suite, run_root/suite.name)
            mapped.append(suite.name)
    link(datasets, ROOT.parent/'datasets')
    if (source/'backup').is_dir():
        link(source/'backup', ROOT/'backup')
    repository = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip())
    exclude = Path(subprocess.check_output(['git', 'rev-parse', '--git-path', 'info/exclude'], cwd=repository, text=True).strip())
    if not exclude.is_absolute():
        exclude = repository/exclude
    pattern = '/'+str((ROOT.parent/'datasets').relative_to(repository))
    contents = exclude.read_text() if exclude.exists() else ''
    if pattern not in contents.splitlines():
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(contents.rstrip()+'\n'+pattern+'\n')
    record = {'source': str(source), 'datasets': str(datasets), 'historical_suites': mapped,
              'warning': 'Read historical suites only. Use a new suite name for any new training.'}
    (ROOT/'.local-artifacts.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
