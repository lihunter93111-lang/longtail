"""Prepare a separately versioned sampler trainer while budget runs stay frozen."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def change(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    source = (HERE / "training.py").read_text(encoding="utf-8")
    source = change(source, "    history = []; orders = []; best_bce = math.inf; best_ap = -math.inf", "    history = []; orders = []; sampling_counts = []; best_bce = math.inf; best_ap = -math.inf")
    source = change(source, '        history = recovery["history"]; orders = recovery["orders"]', '        history = recovery["history"]; orders = recovery["orders"]\n        sampling_counts = recovery["sampling_counts"]')
    source = change(source, "        order = torch.randperm(8000, generator=generator)\n        assert len(torch.unique(order)) == 8000", """        order = torch.multinomial(args.sample_probabilities, 8000, replacement=True, generator=generator)
        counts = torch.bincount(order, minlength=8000).numpy().astype(np.int32)
        sampling_counts.append(counts)
        unique_identities = int(np.count_nonzero(counts))
        assert counts.sum() == 8000""")
    source = change(source, '        objective_losses = []; hard_losses = []', """        weak_mask = torch.zeros(8000,dtype=torch.bool)
        weak_mask[selected_positions.cpu()] = True
        gap15_mask = torch.zeros(8000,dtype=torch.bool)
        gap15_mask[weak15_positions[augmented15].cpu()] = True
        actual_weak = int(weak_mask[order].sum())
        actual_gap15 = int(gap15_mask[order].sum())
        objective_losses = []; hard_losses = []""")
    source = change(source, '"augmented_presentations":augmented_count,"gap15_presentations":int(augmented15.sum())',
        '"augmented_presentations":actual_weak,"gap15_presentations":actual_gap15,"eligible_augmented_identities":augmented_count,"unique_sampled_identities":unique_identities,"repetition_fraction":1-unique_identities/8000')
    source = change(source, '"history":history,"orders":orders,"best_bce"', '"history":history,"orders":orders,"sampling_counts":sampling_counts,"best_bce"')
    source = change(source, '    receipt = {"arm":arm,', """    counts_array = np.stack(sampling_counts)
    np.savez_compressed(folder/"sample_counts.npz",counts=counts_array,probabilities=args.sample_probabilities.numpy())
    total_counts = counts_array.sum(axis=0).astype(np.float64)
    receipt = {"sampling":"cardinality_stratified_combo_mix_0.5","fit_identity_count":8000,
        "sample_counts_sha256":base.sha(folder/"sample_counts.npz"),
        "total_unique_identities":int(np.count_nonzero(total_counts)),
        "realized_effective_sample_size":float(total_counts.sum()**2/np.square(total_counts).sum()),
        "expected_weight_effective_sample_size":float(1/args.sample_probabilities.square().sum()),
        "arm":arm,""")
    source = change(source, '"unique_fit_per_epoch":8000', '"unique_fit_per_epoch":[row["unique_sampled_identities"] for row in history]')
    (HERE / "training_sampling.py").write_text(source, encoding="utf-8")
    runner = (HERE / "run_variant.py").read_text(encoding="utf-8")
    runner = change(runner, "import training\n", "import training_sampling as training\n")
    runner = change(runner, 'HERE/"02_BUDGET_GOAL.txt"', 'HERE/"03A_COMBINATION_SAMPLING_GOAL.txt"')
    runner = change(runner, 'source_files = [Path(__file__),HERE/"training.py",', 'source_files = [Path(__file__),Path(training.__file__),HERE/"training.py",')
    runner = change(runner, '    x,y = base.load(manifest["fit_path"],manifest["fit_A"])\n', '')
    needle = '    source_files = [Path(__file__)'
    insertion = """    assert args.coverage == "M"
    x,y = base.load(manifest["fit_path"],manifest["fit_A"])
    codes = y.astype(int)@(1<<np.arange(6))
    cardinality = y.sum(axis=1).astype(int)
    combination_counts = np.bincount(codes,minlength=64)
    cardinality_counts = np.bincount(cardinality,minlength=7)
    combinations_per_cardinality = np.bincount([int(code).bit_count() for code in np.flatnonzero(combination_counts)],minlength=7)
    assert (combination_counts[1:]>0).all()
    probabilities = .5/8000 + .5*cardinality_counts[cardinality]/(8000*combinations_per_cardinality[cardinality]*combination_counts[codes])
    probabilities = probabilities/probabilities.sum()
    expected_cardinality = np.bincount(cardinality,weights=probabilities,minlength=7)
    np.testing.assert_allclose(expected_cardinality,cardinality_counts/8000,rtol=1e-12,atol=1e-12)
    args.sample_probabilities = torch.from_numpy(probabilities)
    sampler = {"rule":"0.5/N +0.5*N_k/(N*C_k*n_c)","probabilities":probabilities.tolist(),
        "expected_label_fraction":(probabilities@y).tolist(),"expected_cardinality_fraction":expected_cardinality.tolist(),
        "expected_combination_visits_per_epoch":(8000*np.bincount(codes,weights=probabilities,minlength=64)).tolist(),
        "expected_unique_per_epoch":float(np.sum(1-(1-probabilities)**8000)),
        "weight_effective_sample_size":float(1/np.square(probabilities).sum())}
    sampler_path = args.output/"sampler.json"
    if sampler_path.exists():
        assert read(sampler_path)==sampler
    else:
        base.write(sampler_path,sampler)
"""
    assert runner.count(needle) == 1
    runner = runner.replace(needle, insertion + needle)
    runner = change(runner, '"initialization":args.initialization,"seeds":args.seeds,', '"initialization":args.initialization,"sampler_sha256":base.sha(sampler_path),"seeds":args.seeds,')
    runner = change(runner, '        assert sum(row["augmented_presentations"] for row in history)==(0 if args.coverage=="B" else (args.epochs//2)*len(p10))\n        assert sum(row["gap15_presentations"] for row in history)==((args.epochs//4)*len(p15) if args.coverage=="M" else 0)',
        '        assert len(history)==args.epochs\n        assert all(0<=row["gap15_presentations"]<=row["augmented_presentations"]<=8000 for row in history)')
    (HERE / "run_sampling.py").write_text(runner, encoding="utf-8")
    print("SAMPLING_ADAPTERS_PREPARED", flush=True)


if __name__ == "__main__":
    main()
