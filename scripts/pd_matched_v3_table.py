"""Build Pd matched-audit LaTeX table and manifest entries from corrected_summary.csv."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from repo_paths import PD_MATCHED_V3_DIR

SUMMARY_CSV = PD_MATCHED_V3_DIR / "corrected_summary.csv"

METHOD_TEX = {
    "ECP_ccPP": r"ECP (cc-pVTZ-PP)",
    "AE_X2C_ccDK": r"AE--X2C (cc-pVTZ-DK)",
    "AE_X2C_x2cTZ": r"AE--X2C (x2c-TZVPall)",
    "ECP_def2": r"ECP (def2-TZVP)",
}

METHOD_ORDER = ("ECP_ccPP", "AE_X2C_ccDK", "AE_X2C_x2cTZ", "ECP_def2")


def _macro_slug(method: str) -> str:
    return method.replace("_", "")


def load_summary_rows() -> list[dict]:
    with open(SUMMARY_CSV, newline="") as f:
        return list(csv.DictReader(f))


def rows_by_species(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["species"], row["2S+1"])
        out.setdefault(key, []).append(row)
    for key in out:
        out[key].sort(key=lambda r: METHOD_ORDER.index(r["method"]))
    return out


def fmt_re(re_val: str | float | None) -> str:
    if re_val is None or re_val == "":
        return r"$>3.20$"
    r = float(re_val)
    if r >= 3.19:
        return r"$>3.20$"
    return f"{r:.3f}"


def fmt_omega(om: str | float | None) -> str:
    if om is None or om == "":
        return "--"
    o = float(om)
    if o <= 0 or o > 1500:
        return "--"
    return f"{int(round(o))}"


def fmt_sigpi(sig: str, pi: str) -> str:
    if not sig or not pi:
        return "--"
    return f"{float(sig):.2f}/{float(pi):.2f}"


def fmt_qf(q: str) -> str:
    if not q:
        return "--"
    return f"{float(q):+.2f}"


def is_repulsive(row: dict) -> bool:
    cls = row.get("classification", "")
    if "no interior minimum" in cls:
        return True
    if row["species"].startswith("PdF3+"):
        return True
    re_env = row.get("r_e_env_A") or row.get("r_e_NEVPT2_lowest_root_A") or ""
    if re_env and float(re_env) >= 3.19:
        return True
    return False


def bound_row_tex(species_key: str, spin_label: str, row: dict) -> str:
    method = METHOD_TEX[row["method"]]
    slug = _macro_slug(row["method"])
    re_s = fmt_re(row.get("r_e_NEVPT2_lowest_root_A") or row.get("r_e_env_A"))
    om_s = fmt_omega(row.get("omega_e_NEVPT2_lowest_root_cm1") or row.get("omega_e_env_cm1"))
    sigpi = fmt_sigpi(row.get("Mayer_sigma", ""), row.get("Mayer_pi", ""))
    qf = fmt_qf(row.get("q_F_lowdin", ""))
    bo_tot = float(row.get("Mayer_total") or 0)
    return (
        f"{species_key} ({spin_label}) & {method} & Yes & {re_s} & {om_s} & "
        f"{sigpi} & {qf} % {row['species']}:{slug}\n"
    )


MACRO_TABLE_SPECIES = frozenset({"PdF_2", "PdF2+_4"})


def manifest_from_summary(rows: list[dict]) -> dict:
    entries = {}
    for row in rows:
        sp = row["species"]
        if sp not in MACRO_TABLE_SPECIES:
            continue
        meth = row["method"]
        spin = row["2S+1"]
        slug = _macro_slug(meth)
        sp_key = sp.replace("+", "p")
        base = f"PdAudit_{sp_key}_M{spin}_{slug}"
        src = f"data/pd_matched_v3/corrected_summary.csv:{sp},{meth},2S+1={spin}"
        for field, col, nd in (
            ("re", "r_e_NEVPT2_lowest_root_A", 4),
            ("omega", "omega_e_NEVPT2_lowest_root_cm1", 1),
            ("BO", "Mayer_total", 3),
            ("qF", "q_F_lowdin", 3),
            ("sigma", "Mayer_sigma", 3),
            ("pi", "Mayer_pi", 3),
        ):
            val = row.get(col)
            if val is None or val == "":
                continue
            fv = float(val)
            if not math.isfinite(fv):
                continue
            if field == "omega" and (fv <= 0 or fv > 1500):
                continue
            if field == "re" and fv >= 3.19:
                continue
            key = f"{base}_{field}"
            entries[key] = {
                "value": round(fv, nd) if field != "omega" else int(round(fv)),
                "source": src,
            }
    return entries


def pd_ecp_table_tex(rows: list[dict]) -> str:
    grouped = rows_by_species(rows)
    lines = [
        "% Auto-generated from data/pd_matched_v3/corrected_summary.csv",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Species ($2S+1$) & Method & Bound? & $r_e$ (\AA) & $\omega_e$ (cm$^{-1}$) & $\sigma/\pi$ BO & $q(\mathrm{F})$ \\",
        r"\midrule",
    ]
    for species_lbl, species, spin in (
        (r"$\mathrm{PdF}$ ($2\,(^2\Sigma^+)$)", "PdF_2", "2"),
        (r"$[\mathrm{PdF}]^{2+}$ ($4\,(^4\Sigma)$)", "PdF2+_4", "4"),
    ):
        first = True
        for row in grouped.get((species, spin), []):
            sp_col = species_lbl if first else ""
            method = METHOD_TEX[row["method"]]
            re_s = fmt_re(row.get("r_e_NEVPT2_lowest_root_A") or row.get("r_e_env_A"))
            om_s = fmt_omega(row.get("omega_e_NEVPT2_lowest_root_cm1") or row.get("omega_e_env_cm1"))
            sigpi = fmt_sigpi(row.get("Mayer_sigma", ""), row.get("Mayer_pi", ""))
            qf = fmt_qf(row.get("q_F_lowdin", ""))
            lines.append(
                f"{sp_col} & {method} & Yes & {re_s} & {om_s} & {sigpi} & {qf} \\\\"
            )
            first = False
        lines.append(r"\midrule")

    lines.append(
        r"$[\mathrm{PdF}]^{2+}$ ($2\,(^2\Sigma)$) & \multicolumn{6}{@{}l}{No well-defined minimum ($0.09$--$0.20\,\mathrm{eV}$ above quartet; multi-state crossings)} \\"
    )
    lines.append(r"\midrule")

    for species_lbl, species, spin in (
        (r"$[\mathrm{PdF}]^{3+}$ ($1\,(^1\Sigma)$)", "PdF3+_1", "1"),
        (r"$[\mathrm{PdF}]^{3+}$ ($3\,(^3\Sigma)$)", "PdF3+_3", "3"),
    ):
        first = True
        for row in grouped.get((species, spin), []):
            sp_col = species_lbl if first else ""
            method = METHOD_TEX[row["method"]]
            rep = (
                r"\multicolumn{4}{@{}l}{Repulsive on $1.55$--$3.20\,\AA$ grid; "
                r"no $D_e$ reported$^a$}"
            )
            lines.append(f"{sp_col} & {method} & No & {rep} \\\\")
            first = False
        if spin == "1":
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{@{}l}{\footnotesize $^a$Separated-ion asymptotes differ between AE ($\mathrm{Pd}^{3+}+\mathrm{F}$) and ECP ($\mathrm{Pd}^{2+}+\mathrm{F}^{+}$) protocols; metastable ``bound'' rows with $\omega_e \gg 10^3\,\mathrm{cm}^{-1}$ in \texttt{data/pd\_matched\_v3/corrected\_summary.csv} are envelope-fit artefacts.}",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def pd_bond_at_minimum_from_curves() -> dict:
    """Mayer indices at envelope minimum for bar chart (AE cc-DK quartet [PdF]2+)."""
    import json

    path = PD_MATCHED_V3_DIR / "corrected_curves.json"
    with open(path) as f:
        data = json.load(f)
    env = data["PdF2+_4"]["AE_X2C_ccDK"]["envelope"]
    re_best = min(
        (p for p in env if p.get("e_nevpt2") is not None),
        key=lambda p: p["e_nevpt2"],
    )
    bc = re_best["bond_cas"]
    return {
        "PdF2plus4_AEccDK_re": round(re_best["R"], 3),
        "PdF2plus4_AEccDK_mayer_sigma": round(bc["mayer_sigma"], 3),
        "PdF2plus4_AEccDK_mayer_pi": round(bc["mayer_pi"], 3),
        "source": "data/pd_matched_v3/corrected_curves.json:PdF2+_4/AE_X2C_ccDK/envelope",
    }