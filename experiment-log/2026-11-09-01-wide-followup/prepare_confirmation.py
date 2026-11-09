"""Version the reserved cohort generator, audit, and frozen evaluator."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-08-01-overnight"


def main():
    generator = (OLD / "generate_confirmation.m").read_text(encoding="utf-8")
    assert generator.count("3108310800") == 1
    (HERE / "generate_confirmation.m").write_text(generator.replace("3108310800", "3109310900"), encoding="utf-8")
    (HERE / "generate_confirmation_pairs.m").write_text(
        (OLD / "generate_confirmation_pairs.m").read_text(encoding="utf-8"), encoding="utf-8")
    prefix = "import sys\nsys.path.append(str(Path(__file__).resolve().parent.parent/'2026-11-08-01-overnight'))\n"
    for name in ("prepare_final.py", "run_confirmation.py"):
        source = (OLD / name).read_text(encoding="utf-8")
        assert source.count("import training\n") == 1
        source = source.replace("import training\n", prefix + "import training\n")
        if name == "prepare_final.py":
            needle = "    history = sorted(set(history))\n"
            replacement = """    previous_confirmation = read(args.root/"final_confirmation_data/manifest.json")
    history.append(Path(previous_confirmation["dev_path"]))
    additional = args.data.parent/"additional_identity_audit_sources.json"
    if additional.exists():
        history += [Path(path) for path in read(additional)]
    history = sorted(set(history))
"""
            assert source.count(needle) == 1
            source = source.replace(needle, replacement)
        (HERE / name).write_text(source, encoding="utf-8")
    print("CONFIRMATION_ADAPTERS_READY", flush=True)


if __name__ == "__main__":
    main()
