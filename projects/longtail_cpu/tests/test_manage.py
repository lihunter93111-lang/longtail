from pathlib import Path
import pytest
from longtail.data import manage
from longtail.io import write_json, read_json


def test_backup_moves_preserve_original_bytes_and_are_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, 'verify', lambda _: {'verified': True})
    root = tmp_path / 'dataset'
    source = root / 'closedset_6class'
    source.mkdir(parents=True)
    raw = source / 'original.mat'
    raw.write_bytes(b'original dataset bytes')
    inode = raw.stat().st_ino
    inventory = tmp_path / 'inventory.json'
    write_json(inventory, {'files': [{'path': 'closedset_6class/original.mat', 'bytes': raw.stat().st_size}]})
    receipt = manage.backup_unused(root, inventory)
    archived = root / 'backup/closedset_6class/original.mat'
    assert archived.read_bytes() == b'original dataset bytes'
    assert archived.stat().st_ino == inode
    assert not source.exists()
    assert receipt['deletion_performed'] is False
    assert manage.backup_unused(root, inventory) == receipt


def test_conflicting_backup_is_rejected_before_any_move(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, 'verify', lambda _: {'verified': True})
    root = tmp_path / 'dataset'
    (root / 'closedset_6class').mkdir(parents=True)
    (root / 'backup/closedset_6class').mkdir(parents=True)
    (root / 'closedset_6class/original.mat').write_bytes(b'keep')
    inventory = tmp_path / 'inventory.json'
    write_json(inventory, {'files': [{'path': 'closedset_6class/original.mat', 'bytes': 4}]})
    with pytest.raises(FileExistsError):
        manage.backup_unused(root, inventory)
    assert (root / 'closedset_6class/original.mat').read_bytes() == b'keep'
