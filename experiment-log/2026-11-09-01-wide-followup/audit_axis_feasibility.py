"""Permute fixed within-cardinality quota multisets to balance label margins."""
import json
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import training

base = training.base
HERE = Path(__file__).resolve().parent
ROOT = Path("E:/LJQ_code/LJQ_code_experiment_storage/daytime_v1/20261109_01")
SOURCE = Path("E:/LJQ_code/LJQ_code_experiment_storage/overnight_v1/20261108_01/data/fit_A_distribution.json")


def main():
    original = np.array(json.loads(SOURCE.read_text(encoding="utf-8"))["counts"], dtype=int)
    codes = np.arange(1, 64)
    labels = ((codes[:, None] >> np.arange(6)) & 1).astype(int)
    cardinality = labels.sum(1)
    choices = [(index, int(quota)) for index in range(63) for quota in np.unique(original[cardinality == cardinality[index]])]
    variable_count = len(choices) + 1
    rows, lower, upper = [], [], []
    for index in range(63):
        row = np.zeros(variable_count)
        for variable, (destination, quota) in enumerate(choices):
            row[variable] = destination == index
        rows.append(row)
        lower.append(1)
        upper.append(1)
    for k in range(1, 7):
        values, counts = np.unique(original[cardinality == k], return_counts=True)
        for value, count in zip(values, counts):
            row = np.zeros(variable_count)
            for variable, (destination, quota) in enumerate(choices):
                row[variable] = cardinality[destination] == k and quota == value
            rows.append(row)
            lower.append(count)
            upper.append(count)
    target = float((labels.T @ original).sum() / 6)
    for label in range(6):
        row = np.r_[[quota * labels[destination, label] for destination, quota in choices], -1.0]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(target)
        row = np.r_[[-quota * labels[destination, label] for destination, quota in choices], -1.0]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(-target)
    result = milp(c=np.r_[np.zeros(variable_count - 1), 1.0],
        integrality=np.r_[np.ones(variable_count - 1), 0],
        bounds=Bounds(np.zeros(variable_count), np.r_[np.ones(variable_count - 1), 8000]),
        constraints=LinearConstraint(np.array(rows), np.array(lower), np.array(upper)),
        options={"time_limit": 30, "mip_rel_gap": 0})
    output = {"role": "future_data_design_feasibility", "new_model_training": False,
        "solver_status": int(result.status), "solver_message": str(result.message), "solver_success": bool(result.success),
        "binary_variables": variable_count - 1, "source_sha256": base.sha(Path(__file__)),
        "quota_source_sha256": base.sha(SOURCE), "prompt_sha256": base.sha(HERE / "07A_AXIS_FEASIBILITY_GOAL.txt")}
    if result.x is not None:
        assignment = np.rint(result.x[:-1]).astype(int)
        counts = np.zeros(63, dtype=int)
        for selected, (destination, quota) in zip(assignment, choices):
            counts[destination] += selected * quota
        assert counts.sum() == original.sum() == 8000 and (counts > 0).all()
        for k in range(1, 7):
            np.testing.assert_array_equal(np.sort(counts[cardinality == k]), np.sort(original[cardinality == k]))
        margins = labels.T @ counts
        deviation = float(np.max(np.abs(margins - target)))
        assert deviation <= result.fun + 1e-5
        output.update(feasible=True, quotas=counts.tolist(), positive_margins=margins.tolist(),
            label_IF=float(margins.max() / margins.min()), combo_IF=float(counts.max() / counts.min()),
            maximum_label_deviation=deviation, objective=float(result.fun), dual_bound=float(result.mip_dual_bound),
            mip_gap=float(result.mip_gap), within_cardinality_multisets_identical=True,
            cooccurrence=(labels.T @ (counts[:, None] * labels)).tolist())
        base.table = training.previous.table
        base.table(ROOT / "axis_feasible_quotas.csv", [{"code": int(code), "cardinality": int(k),
            "original_quota": int(old), "permuted_quota": int(new)} for code, k, old, new in zip(codes, cardinality, original, counts)])
    else:
        output.update(feasible_incumbent_found=False)
    base.write(ROOT / "axis_feasibility.json", output)
    print(json.dumps({key: value for key, value in output.items() if key not in ("quotas", "cooccurrence")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
