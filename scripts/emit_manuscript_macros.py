#!/usr/bin/env python3
"""Generate LaTeX macros and manifests for headline numbers and Table 4 from saved JSON."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from repo_paths import GENERATED_DIR, PD_MATCHED_V3_DIR, RESULTS_DIR
from pd_matched_v3_table import (
    load_summary_rows,
    manifest_from_summary,
    pd_bond_at_minimum_from_curves,
    pd_ecp_table_tex,
)

HA_TO_EV = 27.211386245988
MANIFEST_JSON = RESULTS_DIR / "manuscript_number_manifest.json"
TABLE4_MANIFEST_JSON = RESULTS_DIR / "table4_manifest.json"
MACROS_TEX = GENERATED_DIR / "macros.tex"
PD_TABLE_TEX = GENERATED_DIR / "pd_ecp_table.tex"
PD_MANIFEST_JSON = RESULTS_DIR / "pd_matched_v3_manifest.json"

# Fixed manuscript Table~\ref{tab:design} order: (macro slug, pool, name substring)
TABLE4_ROWS = [
    ("DesignOF3plus", "extreme", "[OF]3+"),
    ("DesignMoF5plus", "extreme", "[MoF]5+"),
    ("DesignVF4plus", "extreme", "[VF]4+"),
    ("DesignWF5plus", "extreme", "[WF]5+"),
    ("DesignNF2plus", "extreme", "[NF]2+"),
    ("DesignNbF4plus", "extreme", "[NbF]4+"),
    ("DesignTaF4plus", "extreme", "[TaF]4+"),
    ("DesignFCF2plus", "design", "[F-C-F]2+"),
    ("DesignTiF3plus", "extreme", "[TiF]3+"),
    ("DesignCFplus", "design", "CF+ (1S+, benchmark)"),
    ("DesignZrF3plus", "extreme", "[ZrF]3+"),
    ("DesignCrF5plus", "extreme", "[CrF]5+"),
]


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def tex_macro_name(key: str) -> str:
    """Control sequence names must not embed bare digits (LaTeX macro names stop at digits)."""
    if key.startswith("PdAudit_"):
        body = key[len("PdAudit_") :]
        out: list[str] = []
        for c in body:
            if c.isdigit():
                out.append(_DIGIT_WORDS.get(c, c))
            elif c.isalnum():
                out.append(c)
        return "Data" + "".join(out)
    parts = key.split("_")
    species = parts[0]
    field = "".join(parts[1:])
    sp = "".join(_DIGIT_WORDS.get(c, c) for c in species if c.isalnum())
    fd = "".join(c for c in field if c.isalpha())
    return "Data" + sp + fd


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


def lookup_candidate(pool: list, substr: str) -> dict:
    return find_extreme(pool, substr)


def fmt_display_bo(x: float) -> str:
    v = round(float(x), 3)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def fmt_qf(q: float) -> str:
    s = f"{q:+.2f}"
    if s.endswith("0") and "." in s:
        s = f"{q:+.2f}"  # keep two decimals for table
    return s


def table4_from_json() -> dict:
    extreme = load_json("extreme_bo_results.json")["candidates"]
    design = load_json("triple_bond_design_screen.json")["candidates"]
    out = {}
    for slug, pool_name, substr in TABLE4_ROWS:
        pool = extreme if pool_name == "extreme" else design
        row = lookup_candidate(pool, substr)
        p, h = row["pbe0"], row["hf"]
        src = (
            f"extreme_bo_results.json:{row['name']}"
            if pool_name == "extreme"
            else f"triple_bond_design_screen.json:{row['name']}"
        )
        out[f"{slug}_re"] = {"value": row["r_e_ang"], "source": src}
        out[f"{slug}_BOpbe0"] = {"value": p["mayer_total"], "source": src}
        out[f"{slug}_SigPi"] = {
            "value": f"{p['mayer_sigma']:.2f}/{p['mayer_pi']:.2f}",
            "source": src,
        }
        out[f"{slug}_HFbo"] = {"value": h["mayer_total"], "source": src}
        out[f"{slug}_HFpi"] = {"value": h["mayer_pi"], "source": src}
        out[f"{slug}_qF"] = {"value": p["q_F"], "source": src}
        out[f"{slug}_bound"] = {"value": row.get("bound", True), "source": src}
    return out


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
    mof5 = find_extreme(ex, "[MoF]5+")
    wf5 = find_extreme(ex, "[WF]5+")

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

    entries.update(table4_from_json())
    entries.update(manifest_from_summary(load_summary_rows()))

    bar = pd_bond_at_minimum_from_curves()
    entries["PdF2plus4_AEccDK_mayer_sigma"] = {
        "value": bar["PdF2plus4_AEccDK_mayer_sigma"],
        "source": bar["source"],
    }
    entries["PdF2plus4_AEccDK_mayer_pi"] = {
        "value": bar["PdF2plus4_AEccDK_mayer_pi"],
        "source": bar["source"],
    }

    def _slug_id(text: str) -> str:
        return (
            text.replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "p")
            .replace("-", "")
            .replace(",", "")
        )

    org_path = RESULTS_DIR / "organometallic_fluorine_screen.json"
    if org_path.exists():
        org = load_json("organometallic_fluorine_screen.json")
        for c in org.get("candidates", []):
            slug = _slug_id(c.get("id", "row"))
            src = f"organometallic_fluorine_screen.json:{c.get('id')}"
            entries[f"OrgScreen_{slug}_re"] = {"value": c["r_e_ang"], "source": src}
            entries[f"OrgScreen_{slug}_BO"] = {"value": c["mayer_total"], "source": src}
            entries[f"OrgScreen_{slug}_SigPi"] = {
                "value": f"{c['mayer_sigma']:.2f}/{c['mayer_pi']:.2f}",
                "source": src,
            }
            entries[f"OrgScreen_{slug}_qF"] = {"value": c["q_F"], "source": src}

    cage_path = RESULTS_DIR / "superacid_and_cage_results.json"
    if cage_path.exists():
        cage = load_json("superacid_and_cage_results.json")
        for c in cage.get("monocations", []):
            slug = _slug_id(c.get("id", "row"))
            src = f"superacid_and_cage_results.json:{c.get('id')}"
            entries[f"CageScreen_{slug}_re"] = {"value": c["r_e_ang"], "source": src}
            entries[f"CageScreen_{slug}_BO"] = {"value": c["mayer_total"], "source": src}
        for block in cage.get("compression_scans", []):
            for pt in block.get("points", []):
                if abs(pt["R_ang"] - 1.62) < 1e-3:
                    entries["CageScreen_FVH3plus_eq_BO"] = {
                        "value": pt["mayer_total"],
                        "source": "superacid_and_cage_results.json:compression/F-VH3 eq",
                    }
                if abs(pt["R_ang"] - 1.32) < 1e-3:
                    entries["CageScreen_FVH3plus_R132_BO"] = {
                        "value": pt["mayer_total"],
                        "source": "superacid_and_cage_results.json:compression/F-VH3 R=1.32",
                    }
                    entries["CageScreen_FVH3plus_R132_dEeV"] = {
                        "value": pt["delta_E_eV"],
                        "source": "superacid_and_cage_results.json:compression/F-VH3 R=1.32",
                    }

    super_path = RESULTS_DIR / "superacid_complexes_results.json"
    if super_path.exists():
        sa = load_json("superacid_complexes_results.json")
        for block in sa.get("complexes", []):
            name = block["name"]
            slug = (
                name.replace("[", "")
                .replace("]", "")
                .replace(" ", "")
                .replace("...", "")
                .replace("+", "plus")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )
            pbe = block["pbe0"]
            src = f"superacid_complexes_results.json:{name}"
            entries[f"Superacid_{slug}_re"] = {"value": block["r_e"], "source": src}
            entries[f"Superacid_{slug}_BO"] = {"value": pbe["mayer_total"], "source": src}
            entries[f"Superacid_{slug}_SigPi"] = {
                "value": f"{pbe['mayer_sigma']:.2f}/{pbe['mayer_pi']:.2f}",
                "source": src,
            }
            entries[f"Superacid_{slug}_qF"] = {"value": pbe["q_F"], "source": src}

    return entries


def macros_tex(entries: dict) -> str:
    lines = [
        "% Auto-generated by scripts/emit_manuscript_macros.py — do not edit by hand.",
        "% Regenerate: python scripts/emit_manuscript_macros.py",
        "",
    ]
    seen: dict[str, str] = {}
    for key, meta in sorted(entries.items()):
        v = meta["value"]
        if key.endswith("_bound"):
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        if isinstance(v, str) and key.endswith("_SigPi"):
            fmt = v
        elif isinstance(v, float) and key.endswith("_qF"):
            fmt = fmt_qf(v)
        elif isinstance(v, float) and key.endswith("_re"):
            fmt = f"{v:.4f}".rstrip("0").rstrip(".")
        elif isinstance(v, float) and (
            "BOpbe0" in key or key.endswith("_HFbo") or key.endswith("_HFpi")
        ):
            fmt = fmt_display_bo(v)
        elif isinstance(v, float):
            if "ha" in key.lower():
                fmt = f"{v:.7f}".rstrip("0").rstrip(".")
            elif abs(v) >= 100 or (abs(v) < 0.01 and v != 0):
                fmt = f"{v:.6g}"
            else:
                fmt = f"{v:.4f}".rstrip("0").rstrip(".")
        else:
            fmt = str(v)
        macro = tex_macro_name(key)
        if macro in seen:
            raise ValueError(
                f"Duplicate LaTeX macro \\{macro} from keys {seen[macro]!r} and {key!r}"
            )
        seen[macro] = key
        lines.append(f"\\newcommand{{\\{macro}}}{{{fmt}}}")
    lines.append("")
    return "\n".join(lines)


def main():
    entries = manifest_from_results()
    table4_only = {k: v for k, v in entries.items() if k.startswith("Design")}
    pd_only = {k: v for k, v in entries.items() if k.startswith("PdAudit_")}
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSON, "w") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
    with open(PD_MANIFEST_JSON, "w") as f:
        json.dump(pd_only, f, indent=2, sort_keys=True)
    with open(PD_TABLE_TEX, "w") as f:
        f.write(pd_ecp_table_tex(load_summary_rows()))
    with open(TABLE4_MANIFEST_JSON, "w") as f:
        json.dump(table4_only, f, indent=2, sort_keys=True)
    with open(MACROS_TEX, "w") as f:
        f.write(macros_tex(entries))
    print(f"Wrote {MANIFEST_JSON} ({len(entries)} numbers)")
    print(f"Wrote {TABLE4_MANIFEST_JSON} ({len(table4_only)} Table-4 fields)")
    print(f"Wrote {MACROS_TEX}")
    print(f"Wrote {PD_TABLE_TEX}")
    print(f"Wrote {PD_MANIFEST_JSON} ({len(pd_only)} Pd fields)")


if __name__ == "__main__":
    main()