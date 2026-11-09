"""Small end-to-end checks, independent of historical datasets and weights."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
from cjlt.data.ama_hdf5 import load_roles, WeakViewDataset
from cjlt.data.label_sampling import sampling_probabilities
from cjlt.engine.longtail_evaluation import fit_thresholds
from cjlt.engine.longtail_evaluation import compare_predictions
from cjlt.engine.longtail_workflow import train_job, verify_frozen
from cjlt.engine.model_bundle import load_bundle, file_sha256
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


def test_weak_identity_schedule_matched_initialization_and_resume(protocol, tmp_path, monkeypatch):
    config, _ = protocol
    config["data"]["cache"] = "memory"
    config["training"].update(batch_size=7, steps_per_epoch=18, identity_order="fixed_generator",
                              schedule="endpoint_cosine", lr_min=1e-6, selection_metric="raw_bce")
    weak = tmp_path / "weak.h5"
    # Synthetic receipt tests the loader/schedule; physical arithmetic is checked below.
    with h5py.File(config["data"]["train"]) as source, h5py.File(weak, "w") as stream:
        for key in ("iq_complex", "label_selected", "sample_id", "receiver_noise_seed"):
            stream[key] = np.asarray(source[key])[:, :10]
        stream.attrs.update(single_target_delta_verified=True, parent_file_sha256=file_sha256(config["data"]["train"]))
    baseline = train_job(("baseline", config, 9, str(tmp_path / "natural"), 1, "cpu"))
    config["data"]["weak_views"] = str(weak)
    result = train_job(("baseline", config, 9, str(tmp_path / "weak"), 1, "cpu"))
    model, _, info = load_bundle(result["model"])
    _, _, control = load_bundle(baseline["model"])
    assert info["initial_state_sha256"] == control["initial_state_sha256"]
    assert [row["order_sha256"] for row in info["history"]] == [row["order_sha256"] for row in control["history"]]
    assert sum(row["augmented_presentations"] for row in info["history"]) == 10
    assert [row["lr"] for row in info["history"]] == pytest.approx([1e-3, 1e-6])
    assert all(row["independent_fit_identities"] == 126 for row in info["history"])
    import cjlt.engine.longtail_workflow as workflow
    original = workflow.collect_records
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption after the second training epoch")
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "collect_records", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_job(("baseline", config, 9, str(tmp_path / "interrupted"), 1, "cpu"))
    resumed = train_job(("baseline", config, 9, str(tmp_path / "interrupted"), 1, "cpu"))
    restored, _, _ = load_bundle(resumed["model"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_physical_weak_views_modify_only_one_component(tmp_path):
    generator_path = Path(__file__).resolve().parents[3] / "data_generation/python_longtail/generate.py"
    spec = importlib.util.spec_from_file_location("ama_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    root = tmp_path / "physical"
    generator.generate_dataset(root, np.full(63, 2, dtype=int), 1, 1, 1, 1234, weak_views=True)
    roles = load_roles(dict(classes=list(AMA_CLASSES), train=str(root / "train.h5"),
                           development=str(root / "development.h5"), split=str(root / "split.json")))
    views = WeakViewDataset(roles["fit"], root / "weak_fit.h5")
    assert not np.array_equal(roles["fit"].indices, np.arange(126))
    receipt = json.loads((root / "weak_fit.json").read_text())
    assert receipt["maximum_relative_delta_error"] < 1e-12
    assert receipt["eligible"] == len(views.weak)
    counts = np.zeros(len(views), dtype=int)
    for epoch in range(1, 41):
        views.epoch = epoch
        counts += (views.mapping >= 0) & ((np.arange(len(views)) + epoch) % 2 == 0)
    np.testing.assert_array_equal(counts[views.mapping >= 0], 20)
    with h5py.File(root / "weak_fit.h5") as stream:
        positions = np.asarray(stream["fit_position"]).reshape(-1)
        np.testing.assert_array_equal(roles["fit"].identities[roles["fit"].indices[positions]], views.weak.identities)
        target = np.asarray(stream["target_label"]).reshape(-1)
        relative = np.asarray(stream["relative_power_dB"]).T
        np.testing.assert_allclose(relative[np.arange(len(target)), target], -10, atol=1e-10)
    with h5py.File(root / "weak_fit.h5", "a") as stream:
        stream["fit_position"][0, 0] = (positions[0] + 1) % len(roles["fit"])
    with pytest.raises(ValueError, match="declared fit identity order"):
        WeakViewDataset(roles["fit"], root / "weak_fit.h5")
    for role in roles.values():
        role.close()
    views.weak.close()


def test_fusion_comparison_freezes_before_report_and_aligns_predictions(protocol, tmp_path, monkeypatch):
    from experiments.compare_longtail import run_comparison
    from cjlt.data.ama_hdf5 import AMAHDF5Dataset
    config, _ = protocol
    base = train_job(("baseline", config, 7, str(tmp_path / "base"), 1, "cpu"))
    config.update(reference=base["model"], specialist=base["model"],
                  fusion=dict(kind="affine", epochs=2, fpr_slack=0.0025))
    output = tmp_path / "comparison"
    original = AMAHDF5Dataset.__getitem__
    report_calls = []
    def checked_read(self, position):
        if self.path == config["data"]["development"] and self.indices[position] >= 126:
            assert (output / "all_models_frozen.json").exists()
            report_calls.append(position)
        return original(self, position)
    with monkeypatch.context() as patch:
        patch.setattr(AMAHDF5Dataset, "__getitem__", checked_read)
        frozen = run_comparison(config, output, "fusion", [7], report=False)
        assert len(frozen) == 4 and report_calls == []
        summary = run_comparison(config, output, "fusion", [7], report=True)
    assert len(report_calls) == 4 * 63
    assert set(summary["methods"]) == {"R_calibrated", "affine_all", "saw_gate", "tail_gate"}
    assert summary["comparisons"]["affine_all_vs_R_calibrated"]["exact_delta"]["t95"] is None
    path = output / "affine_all/seed7/model.pt"
    model, thresholds, info = load_bundle(path)
    assert info["report_used"] is False
    assert info["guard_policy"].startswith("diagnostic")
    roles = load_roles(config["data"])
    try:
        verify_frozen(path.with_name("frozen.json"), path, config, roles)
        records = collect_records(model, roles["report"])
        assert metrics(records, records["logits"], thresholds)["samples"] == 63
    finally:
        for dataset in roles.values():
            dataset.close()
    candidate = output / "evaluation/affine_all/seed7/predictions.npz"
    reference = output / "evaluation/R_calibrated/seed7/predictions.npz"
    with np.load(candidate) as stream:
        values = {key: stream[key] for key in stream.files}
    values["identity"] = values["identity"][::-1]
    bad = tmp_path / "misaligned.npz"
    np.savez_compressed(bad, **values)
    with pytest.raises(ValueError, match="differ in identity"):
        compare_predictions({7: bad}, {7: reference})
