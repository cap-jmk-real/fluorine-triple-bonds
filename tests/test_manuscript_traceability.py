"""Drift check: manifest numbers must match values recomputed from results JSON."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def load_manifest():
    path = RESULTS / "manuscript_number_manifest.json"
    assert path.exists(), "Run scripts/emit_manuscript_macros.py first"
    with open(path) as f:
        return json.load(f)


def recompute_manifest():
    spec = importlib.util.spec_from_file_location(
        "emit_manuscript_macros", REPO / "scripts" / "emit_manuscript_macros.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.manifest_from_results()


def test_required_result_files_exist():
    required = [
        "ab_initio_results.json",
        "extreme_bo_results.json",
        "triple_bond_design_screen.json",
        "forensic_mnf_reproduction.json",
        "ligated_triple_bond_cores.json",
    ]
    for name in required:
        assert (RESULTS / name).exists(), f"Missing {name}"


def test_table4_manifest_matches_json():
    spec = importlib.util.spec_from_file_location(
        "emit_manuscript_macros", REPO / "scripts" / "emit_manuscript_macros.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = mod.table4_from_json()
    path = RESULTS / "table4_manifest.json"
    assert path.exists(), "Run emit_manuscript_macros.py"
    saved = json.loads(path.read_text())
    for key in ("DesignCrF5plus_re", "DesignOF3plus_BOpbe0", "DesignFCF2plus_re"):
        assert abs(float(saved[key]["value"]) - float(fresh[key]["value"])) < 1e-5, key


def test_headline_numbers_match_manifest():
    manifest = load_manifest()
    fresh = recompute_manifest()
    for key in (
        "OF3plus_BO_pbe0",
        "OF3plus_BO_hf",
        "OF3plus_re_ang",
        "MoF5plus_BO_pbe0",
        "MnF1_deltaE_eV",
        "forensic_bs_delta_eV",
    ):
        assert key in manifest, key
        assert key in fresh, key
        v0 = manifest[key]["value"]
        v1 = fresh[key]["value"]
        assert abs(float(v0) - float(v1)) < 1e-6, f"{key}: manifest {v0} != fresh {v1}"


def test_macros_regenerate_cleanly():
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "emit_manuscript_macros.py")],
        cwd=str(REPO),
        check=True,
    )
    assert (REPO / "paper" / "generated" / "macros.tex").exists()