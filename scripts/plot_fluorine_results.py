#!/usr/bin/env python3
"""Generate publication-grade vector figures for the Fluorine Triple Bonds paper."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from repo_paths import FIGURES_DIR, PD_MATCHED_V3_DIR, RESULTS_DIR

RES_JSON = RESULTS_DIR / "ab_initio_results.json"
PD_CURVES = PD_MATCHED_V3_DIR / "corrected_curves.json"
FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
HA_TO_EV = 27.211386245988

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
PD_METHOD_STYLE = [
    ("ECP_ccPP", "ECP cc-pVTZ-PP", COLORS[3], "s--"),
    ("AE_X2C_ccDK", "AE X2C cc-pVTZ-DK", COLORS[0], "o-"),
    ("AE_X2C_x2cTZ", "AE X2C x2c-TZVPall", COLORS[2], "^-."),
    ("ECP_def2", "ECP def2-TZVP", COLORS[1], "d:"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_pd_curves():
    with open(PD_CURVES, encoding="utf-8") as f:
        return json.load(f)


def envelope_rs_es(species: str, method: str, pd_data: dict) -> tuple[np.ndarray, np.ndarray]:
    env = pd_data[species][method]["envelope"]
    pts = [p for p in env if p.get("e_nevpt2") is not None]
    pts.sort(key=lambda p: p["R"])
    rs = np.array([p["R"] for p in pts], dtype=float)
    es = np.array([p["e_nevpt2"] for p in pts], dtype=float)
    return rs, es


with open(RES_JSON, encoding="utf-8") as f:
    d = json.load(f)

pd_data = load_pd_curves()

# =========================================================================
# Figure 1: (a) MnF spin-state PES, (b) sigma/pi Mayer bond-order comparison
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1), dpi=300)

e_ref = min(p["e_nevpt2"] for p in d["MnF_neutral"]["7Sigma+"]["points"] if "e_nevpt2" in p)
styles = [
    ("7Sigma+", r"$^7\Sigma^+$", COLORS[0], "o-"),
    ("5Sigma+", r"$^5\Sigma^+$", COLORS[1], "s--"),
    ("3Sigma+", r"$^3\Sigma^+$", COLORS[2], "^-."),
    ("1Sigma+", r"$^1\Sigma^+$", COLORS[3], "d:"),
]
for key, lbl, col, fmt in styles:
    pts = [p for p in d["MnF_neutral"][key]["points"] if "e_nevpt2" in p]
    rs = [p["R"] for p in pts]
    es = [(p["e_nevpt2"] - e_ref) * HA_TO_EV for p in pts]
    ax1.plot(rs, es, fmt, color=col, lw=1.5, ms=4, label=lbl)

ax1.axvline(1.838, color="gray", ls="--", lw=1.0, label=r"Expt. $r_e = 1.838\,\mathrm{\AA}$")
ax1.set_xlabel(r"Internuclear distance $R$ ($\mathrm{\AA}$)")
ax1.set_ylabel(r"Relative NEVPT2 energy $\Delta E$ (eV)")
ax1.set_xlim(1.55, 2.85)
ax1.set_ylim(-0.6, 8.5)
ax1.set_title(r"(a) Neutral MnF Spin-State PES (X2C-NEVPT2)")
ax1.legend(frameon=False, loc="upper right", ncol=2)

# [PdF]2+ quartet at matched minimum (AE cc-DK from v3 audit)
pd2_env = pd_data["PdF2+_4"]["AE_X2C_ccDK"]["envelope"]
pd2_min = min(pd2_env, key=lambda p: p["e_nevpt2"])
pd2_bo = pd2_min["bond_cas"]

species = [
    (r"$\mathrm{CF}^+$" + "\n" + r"$^1\Sigma^+$", d["main_group_benchmarks"]["CF+_1Sigma+"]["at_min"]["bond_cas"]),
    (r"$\mathrm{NF}$" + "\n" + r"$^3\Sigma^-$", d["main_group_benchmarks"]["NF_3Sigma-"]["at_min"]["bond_cas"]),
    (r"$\mathrm{MnF}$" + "\n" + r"$^7\Sigma^+$", d["MnF_neutral"]["7Sigma+"]["at_min"]["bond_cas"]),
    (r"$\mathrm{MnF}$" + "\n" + r"$^1\Sigma^+$", d["MnF_neutral"]["1Sigma+"]["at_min"]["bond_cas"]),
    (r"$[\mathrm{MnF}]^{3+}$" + "\n" + r"$^6\Sigma$", d["high_valent_3d"]["MnF3+_6"]["at_min"]["bond_cas"]),
    (r"$\mathrm{NiF}$" + "\n" + r"$^2\Pi/\Sigma$", d["high_valent_3d"]["NiF_2"]["at_min"]["bond_cas"]),
    (r"$[\mathrm{NiF}]^{2+}$" + "\n" + r"$^4\Sigma$", d["high_valent_3d"]["NiF2+_4"]["at_min"]["bond_cas"]),
    (
        r"$[\mathrm{PdF}]^{2+}$" + "\n" + r"$^4\Sigma$ (v3)",
        {
            "mayer_sigma": pd2_bo["mayer_sigma"],
            "mayer_pi": pd2_bo["mayer_pi"],
        },
    ),
]
x = np.arange(len(species))
sigs = [s[1]["mayer_sigma"] for s in species]
pis = [s[1]["mayer_pi"] for s in species]
lbls = [s[0] for s in species]

ax2.bar(x, sigs, width=0.62, color=COLORS[0], label=r"$\sigma$-bond order")
ax2.bar(x, pis, width=0.62, bottom=sigs, color=COLORS[1], label=r"$\pi$-bond order")
ax2.axhline(1.0, color="gray", ls=":", lw=1.0)
ax2.axhline(2.0, color="gray", ls="--", lw=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(lbls, fontsize=6.8)
ax2.set_ylabel("Mayer bond order (CASSCF)")
ax2.set_ylim(0, 2.2)
ax2.set_title(r"(b) $\sigma/\pi$ Bond-Order Decomposition")
ax2.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_mnf_pes_and_bo.pdf", bbox_inches="tight")
plt.close()

# =========================================================================
# Figure 2: matched 4-protocol NEVPT2 on PdF (2) and [PdF]2+ (4)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=300)

for ax, species, title in [
    (ax1, "PdF_2", r"(a) $\mathrm{PdF}$ ($^2\Sigma^+$): matched 2$\times$2 audit"),
    (ax2, "PdF2+_4", r"(b) $[\mathrm{PdF}]^{2+}$ ($^4\Sigma$): shallow wells"),
]:
    for method, lbl, col, fmt in PD_METHOD_STYLE:
        rs, es = envelope_rs_es(species, method, pd_data)
        e5 = float(es[-1])
        ax.plot(rs, (es - e5) * HA_TO_EV, fmt, color=col, lw=1.4, ms=4, label=lbl)
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"Internuclear distance $R$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$E(R) - E(5.00\,\mathrm{\AA})$ (eV)")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, loc="best", fontsize=6.5)
    ax.set_xlim(1.55, 3.25)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_ae_vs_ecp_pdf.pdf", bbox_inches="tight")
plt.close()
print("Saved Fluorine figures to", FIG_DIR)