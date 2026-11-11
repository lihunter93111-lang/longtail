"""Match both label-margin vectors exactly across the two future CF levels."""
from common import *
from scipy.optimize import Bounds, LinearConstraint, milp


def main():
    path = ROOT/"analysis/future_dual_tail_matrix.json"
    prior = read(path)
    cells = prior["cells"]
    quotas = np.asarray(cells["CF16_IF3"]["counts"],dtype=int)
    target = np.asarray(cells["CF7p85_IF1"]["label_margins"],dtype=int)
    labels = ((np.arange(1,64)[:,None]>>np.arange(6))&1).astype(int)
    cardinality = labels.sum(1)
    choices = [(index,int(quota)) for index in range(63) for quota in np.unique(quotas[cardinality==cardinality[index]])]
    rows,bounds = [],[]
    for index in range(63):
        rows.append([int(destination==index) for destination,quota in choices])
        bounds.append(1)
    for k in range(1,7):
        values,counts = np.unique(quotas[cardinality==k],return_counts=True)
        for value,count in zip(values,counts):
            rows.append([int(cardinality[destination]==k and quota==value) for destination,quota in choices])
            bounds.append(int(count))
    for column in range(6):
        rows.append([quota*labels[destination,column] for destination,quota in choices])
        bounds.append(int(target[column]))
    result = milp(c=np.zeros(len(choices)),integrality=np.ones(len(choices)),bounds=Bounds(0,1),
        constraints=LinearConstraint(np.asarray(rows),bounds,bounds),options={"time_limit":30,"mip_rel_gap":0})
    output = {"role":"future_data_design_feasibility","new_model_training":False,"input_sha256":base.sha(path),
        "source_sha256":base.sha(Path(__file__)),"goal_sha256":base.sha(HERE/"06D_EXACT_MARGIN_MATRIX_GOAL.txt"),
        "status":int(result.status),"message":str(result.message),"feasible_found":result.x is not None,"time":stamp()}
    if result.x is not None:
        selected = np.rint(result.x).astype(int)
        np.testing.assert_array_equal(np.asarray(rows)@selected,bounds)
        counts = np.zeros(63,dtype=int)
        for decision,(destination,quota) in zip(selected,choices):
            counts[destination] += decision*quota
        np.testing.assert_array_equal(labels.T@counts,target)
        assert counts.sum()==8000 and counts.max()/counts.min()==16
        for k in range(1,7):
            np.testing.assert_array_equal(np.sort(counts[cardinality==k]),np.sort(quotas[cardinality==k]))
        exact_cells = {name:record for name,record in cells.items() if name!="CF16_near_IF1"}
        exact_cells["CF16_IF1_matched"] = {"counts":counts.tolist(),"label_margins":target.tolist(),
            "label_IF":float(target.max()/target.min()),"combination_IF":16.,
            "cardinality_counts":[int(counts[cardinality==k].sum()) for k in range(1,7)],
            "cooccurrence":(labels.T@(counts[:,None]*labels)).tolist()}
        output.update(cells=exact_cells,label_margins_exactly_matched_across_CF=True,
                      within_cardinality_histograms_matched_within_CF=True)
        helper.table(ROOT/"analysis/future_dual_tail_exact_quotas.csv",[{"code":code,"cardinality":int(cardinality[code-1]),
            **{name:record["counts"][code-1] for name,record in exact_cells.items()}} for code in range(1,64)])
    write(ROOT/"analysis/future_dual_tail_exact_matrix.json",output)
    print(json.dumps({k:v for k,v in output.items() if k!="cells"}),flush=True)


if __name__=="__main__":
    main()
