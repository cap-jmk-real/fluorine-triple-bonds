"""Drift check: manifest numbers must match values recomputed from results JSON."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
PAPER = REPO / "paper"
PD_MANIFEST = RESULTS / "pd_matched_v3_manifest.json"
PD_CSV = REPO / "data" / "pd_matched_v3" / "corrected_summary.csv"


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
        "superacid_complexes_results.json",
        "pd_matched_v3_manifest.json",
    ]
    for name in required:
        assert (RESULTS / name).exists(), f"Missing {name}"
    assert PD_CSV.exists()
    assert (PAPER / "generated" / "pd_ecp_table.tex").exists()


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


def test_pd_manifest_keys_use_pdaudit_prefix():
    saved = json.loads(PD_MANIFEST.read_text())
    assert saved, "empty pd manifest"
    assert all(k.startswith("PdAudit_") for k in saved), list(saved)[:3]


def test_pd_manifest_matches_csv():
    spec = importlib.util.spec_from_file_location(
        "pd_matched_v3_table", REPO / "scripts" / "pd_matched_v3_table.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = mod.manifest_from_summary(mod.load_summary_rows())
    saved = json.loads(PD_MANIFEST.read_text())
    for key, meta in saved.items():
        assert key in fresh, f"Missing key in fresh manifest: {key}"
        assert fresh[key]["source"] == meta["source"]
        assert abs(float(meta["value"]) - float(fresh[key]["value"])) < 1e-5, key


def test_pd_table_tex_no_legacy_reversals():
  text = (PAPER / "generated" / "pd_ecp_table.tex").read_text()
  assert "2.025" not in text
  assert "1.838" not in text
  assert "1.976" not in text
  assert "Repulsive" in text


def test_main_tex_pd_numbers_trace_to_v3():
    tex = (PAPER / "main.tex").read_text()
    assert "pd_matched_v3" in tex.replace("\\_", "_")
    assert "binding reversals" not in tex.lower() or "retract" in tex.lower()
    assert "6--18" not in tex and "16.614" not in tex
    table = (PAPER / "generated" / "pd_ecp_table.tex").read_text()
    import csv

    with open(PD_CSV, newline="") as f:
        rows = { (r["species"], r["method"], r["2S+1"]): r for r in csv.DictReader(f) }
    for species, spin in (("PdF_2", "2"), ("PdF2+_4", "4")):
        for method in ("ECP_ccPP", "AE_X2C_ccDK", "AE_X2C_x2cTZ", "ECP_def2"):
            r = rows[(species, method, spin)]
            re_val = float(r["r_e_NEVPT2_lowest_root_A"])
            if re_val < 3.19:
                assert f"{re_val:.3f}" in table or f"{re_val:.2f}" in table, (
                    f"missing r_e {re_val} for {species}/{method}"
                )


def test_superacid_macros_in_tex():
    macros = (PAPER / "generated" / "macros.tex").read_text()
    tex = (PAPER / "main.tex").read_text()
    for m in re.findall(r"\\DataSuperacid\w+", tex):
        assert m in macros, f"Macro {m} missing from generated/macros.tex"


def test_macros_regenerate_cleanly():
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "emit_manuscript_macros.py")],
        cwd=str(REPO),
        check=True,
    )
    assert (PAPER / "generated" / "macros.tex").exists()


def test_bibliography_dois_present_in_tex():
    tex = (PAPER / "main.tex").read_text()
    assert len(re.findall(r"\\bibitem\{", tex)) >= 24
    assert "doi.org" in tex