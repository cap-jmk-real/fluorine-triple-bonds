#!/usr/bin/env python3
"""
Generate Figure 4 for Paper 1 (fig4_ligand_supramolecular_engineering.{png,pdf}):
  (a) Organic & Inorganic Ligand Ladder on d0 [F-V(L)3]+ (equatorial + trans push-pull)
  (b) Unengineered vs. Engineered Carbon & Supramolecular Immobilization Hosts
  (c) Oriented Local Electric Field (LEF) on [F-V(CF3)3]+ breaking pi-BO > 1.0
"""
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
from repo_paths import FIGURES_DIR, RESULTS_DIR

RES_DIR = RESULTS_DIR
FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(RES_DIR / "organic_ligand_screen.json") as f:
        org = json.load(f)
    with open(RES_DIR / "f_pi_donor_ligand_study.json") as f:
        fpi = json.load(f)
    with open(RES_DIR / "carbon_supramolecular_results.json") as f:
        csup = json.load(f)
    with open(RES_DIR / "engineered_supramolecular_results.json") as f:
        eng = json.load(f)

    COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.7), dpi=240)

    # --- Panel (a): Organic & Inorganic Ligand Ladder on [F-V(L)3]+ ---
    ladder = [
        ("NH$_2^-$",      fpi["axis_A_equatorial"]["[F-V(NH2)3]+"]["pi"],  fpi["axis_A_equatorial"]["[F-V(NH2)3]+"]["BO"]),
        ("OMe$^-$",       org["equatorial_organic"]["OMe"]["pi"],           org["equatorial_organic"]["OMe"]["BO"]),
        ("CH=CH$_2$",     org["equatorial_organic"]["CH=CH2"]["pi"],        org["equatorial_organic"]["CH=CH2"]["BO"]),
        ("Ph$^-$",        org["equatorial_organic"]["Ph"]["pi"],            org["equatorial_organic"]["Ph"]["BO"]),
        ("C$\\equiv$CH",  org["equatorial_organic"]["C#CH"]["pi"],          org["equatorial_organic"]["C#CH"]["BO"]),
        ("CH$_3^-$",      org["equatorial_organic"]["CH3"]["pi"],           org["equatorial_organic"]["CH3"]["BO"]),
        ("CH$_2$F$^-$",   org["equatorial_organic"]["CH2F"]["pi"],          org["equatorial_organic"]["CH2F"]["BO"]),
        ("CHF$_2^-$",     org["equatorial_organic"]["CHF2"]["pi"],          org["equatorial_organic"]["CHF2"]["BO"]),
        ("CF$_3^-$",      org["equatorial_organic"]["CF3"]["pi"],           org["equatorial_organic"]["CF3"]["BO"]),
        ("t-CNCF$_3$",    org["trans_organic"]["CN-CF3"]["pi"],             org["trans_organic"]["CN-CF3"]["BO"]),
        ("t-BF",          fpi["axis_A_trans"]["[BF-V(CF3)3-F]+"]["pi"],     fpi["axis_A_trans"]["[BF-V(CF3)3-F]+"]["BO"]),
    ]
    y = np.arange(len(ladder))
    lbls = [x[0] for x in ladder]
    pis = [x[1] for x in ladder]
    bos = [x[2] for x in ladder]
    ax1.barh(y - 0.18, pis, height=0.35, color=COLORS[0], alpha=0.90, label=r"$\pi$-BO")
    ax1.barh(y + 0.18, bos, height=0.35, color=COLORS[4], alpha=0.80, label="Total BO")
    ax1.axvline(0.867, color=COLORS[0], ls=":", lw=0.9, alpha=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(lbls, fontsize=7.2)
    ax1.set_xlim(0.0, 1.58)
    ax1.set_xlabel("Mayer Bond Order (V–F)")
    ax1.set_title(r"(a) Organic & Trans Ligand Ladder ($d^0$ V)")
    ax1.legend(frameon=True, facecolor="white", framealpha=0.96, edgecolor="#cccccc", fontsize=7.0, loc="lower right")

    # --- Panel (b): Unengineered vs. Engineered Carbon / Supramolecular Hosts ---
    pairs = [
        ("CF$^+$@Cage\n(C$_{20}$ vs C$_{20}$H$_{20}$)",
         csup["Endohedral CF+@C20"]["BO"], eng["Insulating sp3 CF+@C20H20"]["BO"]),
        ("TiF$^{3+}$ Host\n(C$_{20}$ vs (C$_6$F$_6$)$_2$)",
         csup["Endohedral [TiF]3+@C20"]["BO"], eng["Engineered (C6F6)2...[TiF]3+"]["BO"]),
        ("peri-Naphth W\n(H vs F/CN)",
         csup["Intramol. peri-Naphth-WH2F+"]["BO"], eng["Engineered peri-C10F6-W(CN)2F+"]["BO"]),
        ("W(VI) Anchors\n(CH$_3$ vs CN)",
         csup["Carbon-Cage [F-W(CH3)4]+"]["BO"], eng["Engineered [F-W(CN)4]+"]["BO"]),
        ("V(V) Anchors\n(CH$_3$ vs CF$_3$)",
         eng["Baseline [F-V(CH3)3]+"]["BO"], eng["Engineered [F-V(CF3)3]+"]["BO"]),
    ]
    yb = np.arange(len(pairs))
    b_un = [p[1] for p in pairs]
    b_en = [p[2] for p in pairs]
    ax2.barh(yb - 0.18, b_un, height=0.34, color="#999999", alpha=0.85, label="Unengineered host")
    ax2.barh(yb + 0.18, b_en, height=0.34, color=COLORS[2], alpha=0.90, label="Engineered host")
    for yi, u, e in zip(yb, b_un, b_en):
        ax2.text(e + 0.025, yi + 0.18, f"{e:.2f}", va="center", fontsize=6.8, color="#00664B", fontweight="bold")
    ax2.set_yticks(yb)
    ax2.set_yticklabels([p[0] for p in pairs], fontsize=7.1)
    ax2.set_xlim(0.0, 1.65)
    ax2.set_ylim(-0.6, 5.25)
    ax2.set_xlabel("Mayer Total Bond Order (X–F)")
    ax2.set_title("(b) Carbon & Supramolecular Engineering")
    ax2.legend(frameon=True, facecolor="white", framealpha=0.96, edgecolor="#cccccc", fontsize=6.6, loc="upper right", ncol=2)

    # --- Panel (c): Oriented Local Electric Field (LEF) on [F-V(CF3)3]+ ---
    ez_keys = ["Ez=-0.030", "Ez=-0.015", "Ez=+0.000", "Ez=+0.015", "Ez=+0.030", "Ez=+0.050"]
    ez_vals = np.array([-0.030, -0.015, 0.0, 0.015, 0.030, 0.050])
    ef_bo = [fpi["axis_B_electric_field"][k]["BO"] for k in ez_keys]
    ef_pi = [fpi["axis_B_electric_field"][k]["pi"] for k in ez_keys]
    ef_sig = [fpi["axis_B_electric_field"][k]["sigma"] for k in ez_keys]

    ax3.plot(ez_vals, ef_bo, "o-", color=COLORS[1], lw=1.9, ms=5, label="Total BO")
    ax3.plot(ez_vals, ef_pi, "s-", color=COLORS[0], lw=1.8, ms=4.8, label=r"$\pi$-BO")
    ax3.plot(ez_vals, ef_sig, "^--", color=COLORS[2], lw=1.5, ms=4.5, label=r"$\sigma$-BO")
    ax3.axhline(1.0, color="gray", ls=":", lw=1.0, label=r"$\pi$-BO $= 1.0$ threshold")
    ax3.axvline(0.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax3.set_xlim(-0.035, 0.055)
    ax3.set_ylim(0.25, 1.80)
    ax3.set_xlabel(r"Oriented Field $E_z$ along $\mathrm{F}\!\to\!\mathrm{V}$ (a.u.)")
    ax3.set_ylabel("Mayer Bond Order (V–F)")
    ax3.set_title(r"(c) Supramolecular Electric Field on $[\mathrm{F\text{-}V(CF_3)_3}]^+$")
    ax3.legend(frameon=True, facecolor="white", framealpha=0.96, edgecolor="#cccccc", fontsize=6.8, loc="upper left")

    plt.tight_layout()
    out_png = FIG_DIR / "fig4_ligand_supramolecular_engineering.png"
    out_pdf = FIG_DIR / "fig4_ligand_supramolecular_engineering.pdf"
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"DONE -> {out_png}")


if __name__ == "__main__":
    main()
