# LT-ABC epoch-budget diagnostic v1 — 2026-11-06

Goal: following the user's request, measure loss over epochs 10–150 and choose
a supported training horizon rather than extrapolating from eight epochs.

This is an exploratory training-budget revision of [LT-ABC-v1](../2026-11-06-01-lt-abc/results.md).
Keep that completed eight-epoch experiment and its screening decisions intact.

## Frozen inputs and scope

- Run ID: `20261106_epoch_diag_01`.
- Source run: `E:\LJQ_code\LJQ_code_experiment_storage\lt_abc_v1\20261106_local_01`.
- Reuse only its A fit (8000 identities) and early-stop (1008 identities), source
  manifest and shared F2 initialization. Calibration, development and locked-test
  waveforms/scores are not loaded by this diagnostic.
- A / ordinary BCE only; paired model seeds 2050, 2051, 2052; each starts from
  the original initialized state, not the eight-epoch best checkpoint.
- dfms_iq width=16, dropout=.2; whole-model fine-tuning; RMS normalization;
  AdamW lr=1e-4, weight_decay=1e-4, batch=32, grad clip=5, FP32, TF32 off;
  250 updates per epoch, 150 epochs, 37500 updates per seed, no LR schedule.
- Same torch permutation generator and initialization as the source experiment;
  verify initial-state and first-eight-epoch order hashes against its receipts.
- Cache fit and early-stop tensors on the same GPU to reduce transfer overhead;
  this does not change inputs, batch ordering, model, precision or optimizer.

## Logs, plots and selection fixed before the extended run

Log training minibatch BCE (dropout on), early-stop unweighted BCE (eval mode),
early-stop macro-AP and per-label AP every epoch. At epochs 10,20,...,150 also
log fit BCE in eval mode, comparable to early-stop BCE. Preserve raw seed curves,
three-seed mean and sample standard deviation, checkpoint and recovery state.

Choose among horizons 10,20,...,150. For each seed and horizon compute its best
early-stop macro-AP observed up to that horizon. Select the earliest horizon
whose three-seed mean is within **0.001 absolute AP (0.1 percentage point)** of
the mean best up to 150, AND whose individual seeds are each within **0.002 AP**
of their own best up to 150. These are engineering tolerances for this diagnostic,
not statistical equivalence tests. Checkpoints remain selected by maximum
early-stop macro-AP, with exact ties going to the earlier epoch.

Also report the minimum early-stop BCE epoch, each seed's AP peak epoch,
8-vs-selected-vs-150 losses and AP, and late (130–150) changes. Do not choose
epochs from declining training loss alone. If the rule only accepts 150, report
that this ceiling has not established an earlier plateau.

The resulting horizon is an A-baseline-supported budget proposal. It does not
establish B/C convergence or a new method ranking; an A/B/C rerun under one
shared revised budget is a separate next gate. Do not auto-run that gate.

## Required artifacts

`preflight.json`, per-seed `history.json`/`receipt.json`, `curves.csv`,
`checkpoints_10_150.csv`, `selection.json`, `loss_curves.png` and `.pdf`,
`report.md`, `status.json`. Large weights stay in the external output directory;
light tables and figures are copied into this experiment-log folder after QA.
