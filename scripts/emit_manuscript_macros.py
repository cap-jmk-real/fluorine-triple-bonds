#!/usr/bin/env python3
"""Generate LaTeX macros and a JSON manifest of headline manuscript numbers from saved results."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from repo_paths import GENERATED_DIR, PAPER_DIR, RESULTS_DIR

HA_TO_EV = 27.211386245988
MANIFEST_JSON = RESULTS_DIR / "manuscript_number_manifest.json"
MACROS_TEX = GENERATED_DIR / "macros.tex"


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    with open(path) as f:
        return json.load(f)


def round_val(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def find_extreme(candidates: list, substr: str) -> dict:
    for c in candidates:
        if substr in c.get("name", ""):
            return c
    raise KeyError(f"No candidate matching {substr!r}")


def manifest_from_results() -> dict:
    ab = load_json("ab_initio_results.json")
    extreme = load_json("extreme_bo_results.json")
    forensic = load_json("forensic_mnf_reproduction.json")

    mnf7 = ab["MnF_neutral"]["7Sigma+"]
    mnf5 = ab["MnF_neutral"]["5Sigma+"]
    mnf1 = ab["MnF_neutral"]["1Sigma+"]
    e7 = mnf7["spec_nevpt2"]["e_min_ha"]
    e5 = mnf5["spec_nevpt2"]["e_min_grid"]
    e1 = mnf1["spec_nevpt2"]["e_min_grid"]
    de_5_vs_7 = (e5 - e7) * HA_TO_EV
    de_1_vs_7 = (e1 - e7) * HA_TO_EV

    at7 = mnf7["at_min"]["bond_cas"]
    cf = ab["main_group_benchmarks"]["CF+_1Sigma+"]["at_min"]["bond_cas"]
    nf3 = ab["main_group_benchmarks"]["NF_3Sigma-"]["at_min"]["bond_cas"]

    ex = extreme["candidates"]
    of3 = find_extreme(ex, "[OF]3+")

    def extreme_or_ligated(substr: str, ligated_name: str) -> dict:
        try:
            return find_extreme(ex, substr)
        except KeyError:
            lig = load_json("ligated_triple_bond_cores.json")
            for block in lig.get("triple_bond_complexes", {}).values():
                for row in block:
                    if row.get("name") == ligated_name:
                        return {
                            "r_e_ang": row["r_e"],
                            "pbe0": {
                                "mayer_total": row["BO"],
                                "mayer_pi": row.get("pi"),
                            },
                        }
            raise

    mof5 = extreme_or_ligated("[MoF]5+", "Bare [MoF]5+ (reference)")
    wf5 = extreme_or_ligated("[WF]5+", "Bare [WF]5+ (reference)")

    singlet_row = next(r for r in forensic["table1_rows"] if r["mult"] == 1)

    entries = {
        "MnF7_re_ang": {"value": mnf7["spec_nevpt2"]["r_e_ang"], "source": "ab_initio_results.json:MnF_neutral/7Sigma+/spec_nevpt2"},
        "MnF7_omega_cm1": {"value": mnf7["spec_nevpt2"]["omega_e_cm1"], "source": "ab_initio_results.json:MnF_neutral/7Sigma+/spec_nevpt2"},
        "MnF7_BO": {"value": round_val(at7["mayer_total"], 3), "source": "ab_initio_results.json:MnF_neutral/7Sigma+/at_min/bond_cas"},
        "MnF7_qF": {"value": round_val(at7["q_lowdin_F"], 3), "source": "ab_initio_results.json:MnF_neutral/7Sigma+/at_min/bond_cas"},
        "MnF5_deltaE_eV": {"value": round(de_5_vs_7, 3), "source": "ab_initio_results.json:derived NEVPT2 minima"},
        "MnF1_deltaE_eV": {"value": round(de_1_vs_7, 3), "source": "ab_initio_results.json:derived NEVPT2 minima"},
        "CFplus_re_ang": {"value": ab["main_group_benchmarks"]["CF+_1Sigma+"]["spectroscopic_nevpt2"]["r_e_ang"], "source": "ab_initio_results.json:CF+"},
        "CFplus_BO": {"value": round_val(cf["mayer_total"], 3), "source": "ab_initio_results.json:CF+ at_min"},
        "NF_re_ang": {"value": ab["main_group_benchmarks"]["NF_3Sigma-"]["spectroscopic_nevpt2"]["r_e_ang"], "source": "ab_initio_results.json:NF"},
        "NF_piBO": {"value": round_val(nf3["mayer_pi"], 3), "source": "ab_initio_results.json:NF at_min"},
        "NF_BO": {"value": round_val(nf3["mayer_total"], 3), "source": "ab_initio_results.json:NF at_min"},
        "MoF5plus_BO_pbe0": {"value": mof5["pbe0"]["mayer_total"], "source": "extreme_bo_results.json"},
        "WF5plus_BO_pbe0": {"value": wf5["pbe0"]["mayer_total"], "source": "extreme_bo_results.json"},
        "OF3plus_re_ang": {"value": of3["r_e_ang"], "source": "extreme_bo_results.json"},
        "OF3plus_BO_pbe0": {"value": of3["pbe0"]["mayer_total"], "source": "extreme_bo_results.json"},
        "OF3plus_BO_hf": {"value": of3["hf"]["mayer_total"], "source": "extreme_bo_results.json"},
        "OF3plus_pi_pbe0": {"value": of3["pbe0"]["mayer_pi"], "source": "extreme_bo_results.json"},
        "forensic_bs_delta_eV": {"value": singlet_row["delta_bs_ev"], "source": "forensic_mnf_reproduction.json"},
        "forensic_bs_energy_ha": {"value": singlet_row["broken_sym_uhf_ha"], "source": "forensic_mnf_reproduction.json"},
        "forensic_bs_S2": {"value": singlet_row["s2_broken_sym"], "source": "forensic_mnf_reproduction.json"},
        "forensic_mean_uhf_offset_mha": {
            "value": forensic.get(
                "uhf_preprint_offset_mha_mult_7_5_mean",
                forensic.get("mean_abs_uhf_offset_mha_excluding_singlet"),
            ),
            "source": "forensic_mnf_reproduction.json",
        },
    }

    comp_path = RESULTS_DIR / "complementary_bonding_checks.json"
    if comp_path.exists():
        comp = load_json("complementary_bonding_checks.json")
        if "of3plus_basis" in comp:
            for basis, block in comp["of3plus_basis"].items():
                key = f"OF3_{basis}_BO_pbe0"
                entries[key] = {
                    "value": block["pbe0_bo"],
                    "source": f"complementary_bonding_checks.json:of3plus_basis/{basis}",
                }
    return entries


def macros_tex(entries: dict) -> str:
    lines = [
        "% Auto-generated by scripts/emit_manuscript_macros.py — do not edit by hand.",
        f"% Regenerate: python scripts/emit_manuscript_macros.py",
        "",
    ]
    for key, meta in sorted(entries.items()):
        v = meta["value"]
        if isinstance(v, float) and not math.isfinite(v):
            continue
        if isinstance(v, float):
            if "ha" in key.lower():
                fmt = f"{v:.7f}".rstrip("0").rstrip(".")
            elif abs(v) >= 100 or (abs(v) < 0.01 and v != 0):
                fmt = f"{v:.6g}"
            else:
                fmt = f"{v:.4f}".rstrip("0").rstrip(".")
        else:
            fmt = str(v)
        tex_key = "".join(c for c in key if c.isalnum())
        lines.append(f"\\newcommand{{\\Data{tex_key}}}{{{fmt}}}")
    lines.append("")
    return "\n".join(lines)


def main():
    entries = manifest_from_results()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSON, "w") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
    with open(MACROS_TEX, "w") as f:
        f.write(macros_tex(entries))
    print(f"Wrote {MANIFEST_JSON} ({len(entries)} numbers)")
    print(f"Wrote {MACROS_TEX}")


if __name__ == "__main__":
    main()