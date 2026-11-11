"""Continuous lower bounds and integer feasible quotas for future dual-tail data."""
from common import *
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


def main():
    counts = np.asarray(QUOTAS["A_counts"],dtype=int)
    codes = np.arange(1,64)
    labels = ((codes[:,None]>>np.arange(6))&1).astype(int)
    cardinality = labels.sum(1)
    size_rows = np.stack([cardinality==k for k in range(1,7)]).astype(int)
    constraints = np.concatenate((size_rows,labels.T),axis=0)
    margins = labels.T@counts
    balanced = np.full(6,margins.sum()//6,dtype=int)
    balanced[:margins.sum()%6] += 1
    result_rows = []
    for name,target_margins in (("original_IF3",margins),("near_balanced",balanced)):
        target = np.r_[size_rows@counts,target_margins]
        # u = quota/min(quota), t = 1/min(quota), r = max(u).
        equality = np.column_stack((constraints,-target,np.zeros(len(target))))
        upper = np.zeros((63,65))
        upper[:,:63] = np.eye(63)
        upper[:,-1] = -1
        objective = np.r_[np.zeros(64),1.]
        relaxed = linprog(objective,A_ub=upper,b_ub=np.zeros(63),A_eq=equality,b_eq=np.zeros(len(target)),
                          bounds=[(1,None)]*63+[(1/8000,None),(1,None)],method="highs")
        assert relaxed.success,relaxed.message
        lower_bound = float(relaxed.fun)
        continuous = relaxed.x[:63]/relaxed.x[63]
        np.testing.assert_allclose(constraints@continuous,target,atol=1e-6)
        trials = []
        feasible = None
        for factor in (1.0001,1.001,1.005,1.01,1.02):
            ratio = lower_bound*factor
            equality = np.column_stack((constraints,np.zeros(len(target))))
            inequalities = np.vstack((np.column_stack((np.eye(63),-np.ones(63))),
                                      np.column_stack((np.eye(63),-ratio*np.ones(63)))))
            limits = LinearConstraint(np.vstack((equality,inequalities)),
                np.r_[target,np.zeros(63),np.full(63,-np.inf)],np.r_[target,np.full(63,np.inf),np.zeros(63)])
            solution = milp(c=np.r_[np.zeros(63),-1.],integrality=np.r_[np.ones(63),0],
                bounds=Bounds(np.ones(64),np.full(64,8000.)),constraints=limits,
                options={"time_limit":10,"mip_rel_gap":0})
            trial = {"ratio_cap":ratio,"status":int(solution.status),"message":str(solution.message),"feasible_found":solution.x is not None}
            trials.append(trial)
            if solution.x is not None:
                candidate = np.rint(solution.x[:63]).astype(int)
                np.testing.assert_array_equal(constraints@candidate,target)
                assert candidate.min()>=1 and candidate.sum()==8000
                actual_ratio = float(candidate.max()/candidate.min())
                assert actual_ratio<=ratio+1e-6
                feasible = {"counts":candidate.tolist(),"combination_IF":actual_ratio,
                            "label_margins":(labels.T@candidate).tolist(),"minimum_count":int(candidate.min()),
                            "maximum_count":int(candidate.max()),"upper_minus_relaxed_lower":actual_ratio-lower_bound}
                break
        result_rows.append({"condition":name,"label_margins":target_margins.tolist(),
            "label_IF":float(target_margins.max()/target_margins.min()),"relaxed_minimum_combination_IF":lower_bound,
            "continuous_counts":continuous.tolist(),"integer_feasible":feasible,"integer_trials":trials})
    prior_path = SOURCE/"experiment-log/2026-11-09-01-wide-followup/results/axis_feasibility.json"
    prior = read(prior_path)
    permuted = np.asarray(prior["quotas"],dtype=int)
    assert permuted.sum()==8000
    for k in range(1,7):
        np.testing.assert_array_equal(np.sort(permuted[cardinality==k]),np.sort(counts[cardinality==k]))
    output = {"role":"future_data_design_feasibility","new_model_training":False,"new_waveforms_generated":False,
        "source_sha256":base.sha(Path(__file__)),"goal_sha256":base.sha(HERE/"06B_LONGTAIL_FEASIBILITY_GOAL.txt"),
        "original_counts":counts.tolist(),"original_cardinality_counts":(size_rows@counts).tolist(),
        "original_label_margins":margins.tolist(),"original_combination_IF":float(counts.max()/counts.min()),
        "necessary_bound":"combination_IF >= 2*label_IF-1 for six labels and all 63 nonempty combinations",
        "conditions":result_rows,"prior_high_CF_balanced_permutation":{"path":str(prior_path),"sha256":base.sha(prior_path),
            "counts":permuted.tolist(),"label_margins":(labels.T@permuted).tolist(),"label_IF":prior["label_IF"],
            "combination_IF":prior["combo_IF"],"within_cardinality_histogram_verified":True},"time":stamp()}
    write(ROOT/"analysis/dual_longtail_feasibility.json",output)
    print(json.dumps([{k:r[k] for k in ("condition","label_IF","relaxed_minimum_combination_IF","integer_feasible")} for r in result_rows]),flush=True)


if __name__=="__main__":
    main()
