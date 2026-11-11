"""Construct feasible future quota cells with within-cardinality histogram controls."""
from common import *
from scipy.optimize import Bounds, LinearConstraint, milp


def stronger_counts(original,labels,cardinality):
    # quota, lower-anchor binary, upper-anchor binary, absolute displacement.
    size = 252
    rows,lower,upper = [],[],[]
    for basis in [*(cardinality==k for k in range(1,7)),*labels.T]:
        row = np.zeros(size)
        row[:63] = basis
        target = int(basis@original)
        rows.append(row); lower.append(target); upper.append(target)
    for index in range(63):
        row = np.zeros(size)
        row[index],row[63+index] = 1,450
        rows.append(row); lower.append(-np.inf); upper.append(480)
        row = np.zeros(size)
        row[index],row[126+index] = 1,-450
        rows.append(row); lower.append(30); upper.append(np.inf)
        for sign in (-1,1):
            row = np.zeros(size)
            row[index],row[189+index] = sign,-1
            rows.append(row); lower.append(-np.inf); upper.append(sign*original[index])
    for offset in (63,126):
        row = np.zeros(size)
        row[offset:offset+63] = 1
        rows.append(row); lower.append(1); upper.append(np.inf)
    result = milp(c=np.r_[np.zeros(189),np.ones(63)],integrality=np.r_[np.ones(189),np.zeros(63)],
        bounds=Bounds(np.r_[np.full(63,30),np.zeros(189)],np.r_[np.full(63,480),np.ones(126),np.full(63,8000)]),
        constraints=LinearConstraint(np.array(rows),lower,upper),options={"time_limit":30,"mip_rel_gap":0})
    assert result.x is not None,result.message
    counts = np.rint(result.x[:63]).astype(int)
    assert counts.min()==30 and counts.max()==480 and counts.sum()==8000
    np.testing.assert_array_equal(labels.T@counts,labels.T@original)
    for k in range(1,7):
        assert counts[cardinality==k].sum()==original[cardinality==k].sum()
    return counts,{"status":int(result.status),"message":str(result.message),"objective":float(result.fun),
                   "dual_bound":float(result.mip_dual_bound),"gap":float(result.mip_gap)}


def balance_permutation(original,labels,cardinality):
    choices = [(index,int(quota)) for index in range(63) for quota in np.unique(original[cardinality==cardinality[index]])]
    size = len(choices)+1
    rows,lower,upper = [],[],[]
    for index in range(63):
        rows.append([int(destination==index) for destination,quota in choices]+[0])
        lower.append(1); upper.append(1)
    for k in range(1,7):
        values,counts = np.unique(original[cardinality==k],return_counts=True)
        for value,count in zip(values,counts):
            rows.append([int(cardinality[destination]==k and quota==value) for destination,quota in choices]+[0])
            lower.append(int(count)); upper.append(int(count))
    target = float((labels.T@original).sum()/6)
    for label in range(6):
        for sign in (-1,1):
            rows.append([sign*quota*labels[destination,label] for destination,quota in choices]+[-1])
            lower.append(-np.inf); upper.append(sign*target)
    result = milp(c=np.r_[np.zeros(size-1),1.],integrality=np.r_[np.ones(size-1),0],
        bounds=Bounds(np.zeros(size),np.r_[np.ones(size-1),8000]),
        constraints=LinearConstraint(np.array(rows),lower,upper),options={"time_limit":30,"mip_rel_gap":0})
    status = {"status":int(result.status),"message":str(result.message),"feasible_found":result.x is not None}
    if result.x is None:
        return None,status
    selected = np.rint(result.x[:-1]).astype(int)
    counts = np.zeros(63,dtype=int)
    for value,(destination,quota) in zip(selected,choices):
        counts[destination] += value*quota
    for k in range(1,7):
        np.testing.assert_array_equal(np.sort(counts[cardinality==k]),np.sort(original[cardinality==k]))
    status.update(objective=float(result.fun),dual_bound=float(result.mip_dual_bound),gap=float(result.mip_gap))
    return counts,status


def main():
    evidence_path = ROOT/"analysis/dual_longtail_feasibility.json"
    evidence = read(evidence_path)
    original = np.asarray(evidence["original_counts"])
    labels = ((np.arange(1,64)[:,None]>>np.arange(6))&1).astype(int)
    cardinality = labels.sum(1)
    high,high_status = stronger_counts(original,labels,cardinality)
    balanced,balanced_status = balance_permutation(high,labels,cardinality)
    cells = {"CF7p85_IF3":original,"CF7p85_IF1":np.asarray(evidence["prior_high_CF_balanced_permutation"]["counts"]),
             "CF16_IF3":high}
    if balanced is not None:
        cells["CF16_near_IF1"] = balanced
    records = {}
    for name,counts in cells.items():
        margins = labels.T@counts
        records[name] = {"counts":counts.tolist(),"label_margins":margins.tolist(),
            "label_IF":float(margins.max()/margins.min()),"combination_IF":float(counts.max()/counts.min()),
            "cardinality_counts":[int(counts[cardinality==k].sum()) for k in range(1,7)],
            "cooccurrence":(labels.T@(counts[:,None]*labels)).tolist()}
    output = {"role":"future_data_design_feasibility","new_model_training":False,"cells":records,
        "high_CF_solver":high_status,"balanced_permutation_solver":balanced_status,
        "input_sha256":base.sha(evidence_path),"source_sha256":base.sha(Path(__file__)),
        "goal_sha256":base.sha(HERE/"06C_FEASIBLE_MATRIX_GOAL.txt"),
        "original_IF3_minimum_CF_bracket":[evidence["conditions"][0]["relaxed_minimum_combination_IF"],
                                            evidence["original_combination_IF"]],
        "interpretation":"within each CF level, cardinality-specific count histograms match; semantic frequencies and cooccurrences change",
        "time":stamp()}
    write(ROOT/"analysis/future_dual_tail_matrix.json",output)
    helper.table(ROOT/"analysis/future_dual_tail_quotas.csv",[{"code":code,"cardinality":int(cardinality[code-1]),
                 **{name:int(counts[code-1]) for name,counts in cells.items()}} for code in range(1,64)])
    print(json.dumps({"cells":{name:{k:r[k] for k in ("label_margins","label_IF","combination_IF")} for name,r in records.items()},
                      "high_CF_solver":high_status,"balanced_permutation_solver":balanced_status}),flush=True)


if __name__=="__main__":
    main()
