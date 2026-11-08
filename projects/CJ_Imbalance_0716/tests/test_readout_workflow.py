"""Small end-to-end checks, independent of historical datasets and weights."""
from copy import deepcopy
import json
import h5py
import numpy as np
import pytest
import torch
from cjlt.data.ama_hdf5 import load_roles
from cjlt.data.label_sampling import sampling_probabilities
from cjlt.engine.longtail_evaluation import fit_thresholds
from cjlt.engine.longtail_workflow import train_job, verify_frozen
from cjlt.engine.model_bundle import load_bundle
from cjlt.engine.readout_trainer import collect_records, metrics
from cjlt.models.classwise_fusion import AMA_CLASSES


@pytest.fixture
def protocol(tmp_path):
    def write(name, labels, start, paired=False):
        path = tmp_path / name
        labels = np.asarray(labels, np.uint8)
        count = len(labels)
        generator = np.random.default_rng(start)
        signals = (generator.normal(size=(64, count)) + 1j * generator.normal(size=(64, count))).astype(np.complex64)
        with h5py.File(path, "w") as stream:
            stream.attrs["classes_json"] = json.dumps(AMA_CLASSES)
            stream["iq_complex"] = signals
            stream["label_selected"] = labels.T
            stream["sample_id"] = np.arange(start, start + count)[None]
            stream["receiver_noise_seed"] = (np.repeat(np.arange(count // 2), 2) + start if paired
                                               else np.arange(start, start + count))[None]
        return str(path)

    labels = ((np.arange(1, 64)[:, None] >> np.arange(6)) & 1)
    train = write("train.h5", np.tile(labels, (2, 1)), 1000)
    development = write("development.h5", np.tile(labels, (3, 1)), 2000)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(dict(fit=list(range(126)), calibration=list(range(63)),
                                    check=list(range(63, 126)), report=list(range(126, 189)))))
    pair_labels, target, gap = [], [], []
    for column in (2, 4):
        for stratum in range(3):
            negative = np.array([1, 0, 0, 0, 0, 0])
            positive = negative.copy()
            positive[column] = 1
            pair_labels.extend([negative, positive])
            target.extend([column] * 2)
            gap.extend([stratum] * 2)
    pairs = write("pairs.h5", pair_labels, 3000, paired=True)
    with h5py.File(pairs, "a") as stream:
        stream["pair_target"] = np.array(target)[None]
        stream["target_gap_bin"] = np.array(gap)[None]
        stream["pair_side"] = np.tile([0, 1], 6)[None]
    return dict(data=dict(classes=list(AMA_CLASSES), train=train, development=development, split=str(split)),
                backbone=dict(width=4, dropout=0.2),
                training=dict(epochs=2, steps_per_epoch=2, batch_size=8, lr=1e-3)), pairs


@pytest.mark.parametrize("sampler,loss", [("uniform", "bce"), ("label_aware", "weighted_bce"),
                                         ("combo_aware", "logit_adjustment")])
def test_train_freeze_readout_and_evaluate(protocol, tmp_path, sampler, loss):
    config, pairs = protocol
    config["training"].update(sampler=sampler, loss=loss)
    result = train_job(("baseline", config, 7, str(tmp_path / "base"), 1, "cpu"))
    assert result["optimizer_steps"] == 4
    config["reference"] = result["model"]
    config["data"]["pairs"] = pairs
    config["readout"] = dict(name="sawmft", targets=["SawFM", "MFT"], dimensions=8,
                              hidden=4, source="segment", pair_views=2)
    result = train_job(("readout", config, 7, str(tmp_path / "head"), 1, "cpu"))
    model, thresholds, info = load_bundle(result["model"])
    assert info["report_used"] is False
    assert all(row["paired_views"] == 4 for row in info["history"])
    roles = load_roles(config["data"])
    try:
        frozen = tmp_path / "head/seed7/frozen.json"
        verify_frozen(frozen, result["model"], config, roles)
        records = collect_records(model, roles["report"])
        assert metrics(records, records["logits"], thresholds)["samples"] == 63
        # An altered signal with identical identity metadata must fail verification.
        with h5py.File(config["data"]["development"], "a") as stream:
            stream["iq_complex"][0, 0] += 1
        with pytest.raises(ValueError, match="frozen protocol"):
            verify_frozen(frozen, result["model"], config, roles)
    finally:
        for dataset in roles.values():
            dataset.close()

    if sampler == "uniform":
        config["reference"] = result["model"]
        config["readout"].update(name="isrj", targets=["ISRJ"], source="global", pair_views=0)
        result = train_job(("readout", config, 7, str(tmp_path / "isrj"), 1, "cpu"))
        _, _, info = load_bundle(result["model"])
        assert all(row["paired_views"] == 0 for row in info["history"])


def test_role_leakage_label_order_and_calibration_are_rejected(protocol):
    config, _ = protocol
    split = config["data"]["split"]
    values = json.loads(open(split).read())
    values["check"][0] = values["calibration"][0]
    with open(split, "w") as stream:
        json.dump(values, stream)
    with pytest.raises(ValueError, match="leakage"):
        load_roles(config["data"])
    changed = deepcopy(config["data"])
    changed["classes"][1] = "VFT"
    with pytest.raises(ValueError, match="AMA order"):
        load_roles(changed)
    with pytest.raises(ValueError, match="calibration identities"):
        fit_thresholds(np.ones((2, 6)), np.zeros((2, 6)), source_role="report")


def test_combo_sampling_balances_combinations():
    labels = np.array([[1, 0]] * 100 + [[1, 1]] * 2)
    weights = sampling_probabilities(labels, "combo_aware")
    np.testing.assert_allclose([weights[:100].sum(), weights[100:].sum()], [0.5, 0.5])
