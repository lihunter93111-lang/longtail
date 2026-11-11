"""Bind existing M-view feature caches to their source data and baseline models."""
from common import *


def main():
    setup()
    rows = []
    for split in CONFIG["splits"]:
        views = read(BASELINES/split/"w32/protocol.json")["views"]
        for seed in CONFIG["seeds"]:
            model,_,receipt = load_baseline(split,seed)
            with np.load(ROOT/split/"cache"/f"M_{seed}"/"fit.npz") as data:
                fit_targets = data["target"]
            for gap in (10,15):
                source = BASELINES/split/"data"/f"fit_gap{gap}.mat"
                cache = ROOT/split/"cache"/f"M_{seed}"/f"weak{gap}.npz"
                saved = read(cache.with_suffix(".json"))
                assert saved["sha256"]==base.sha(cache)
                assert saved["source_sha256"]==views[source.name]==base.sha(source)
                inputs,positions,targets = helper.load_views(source)
                replay = collect(model,inputs[:64])
                errors = {}
                with np.load(cache) as data:
                    np.testing.assert_array_equal(data["positions"],positions)
                    np.testing.assert_array_equal(data["targets"],targets)
                    np.testing.assert_array_equal(data["target"],fit_targets[positions])
                    for key,values in replay.items():
                        np.testing.assert_allclose(values,data[key][:64],rtol=1e-5,atol=3e-5)
                        errors[key] = float(abs(values-data[key][:64]).max())
                rows.append({"split":split,"seed":seed,"gap":gap,"source_sha256":saved["source_sha256"],
                    "cache_sha256":saved["sha256"],"backbone_checkpoint_sha256":receipt["checkpoint_sha256"],
                    "rows":len(positions),"normal_forward_replay_rows":64,"maximum_errors":errors})
            del model
    write(ROOT/"weak_cache_verification.json",{"passed":True,"caches":rows,"time":stamp()})
    print("WEAK_CACHE_BINDINGS_VERIFIED",len(rows),flush=True)


if __name__=="__main__":
    main()
