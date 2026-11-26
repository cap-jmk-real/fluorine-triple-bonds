#!/usr/bin/env python3
"""
High-resolution Potential Energy Surface (PES) scans across wide internuclear
distances (0.85 A to 4.50 A, 26-32 closely spaced points per curve) to resolve:
  (a) Main-group triple bonds & Coulomb-explosion barriers: CF+, [NF]2+, [OF]3+
  (b) d^0 transition-metal triple bonds & 3d/4d/5d cliff: [TiF]3+, [MoF]5+, [WF]5+, [CrF]5+
  (c) Neutral MnF (7Sigma+) full dissociation curve (1.30 - 4.50 A)
  (d) Matched spin-pure NEVPT2 on triplet [PdF]3+ (repulsive; replaces legacy DFT singlet panel)

Uses X2C-PBE0 / cc-pVTZ-DK (plus def2-TZVP ECP for PdF3+ comparison),
saves full JSON curves, and generates fig3_highres_pes_barriers.{png,pdf}.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from repo_paths import FIGURES_DIR, PD_MATCHED_V3_DIR, RESULTS_DIR

HA_TO_EV = 27.211386245988
OUT_DIR = RESULTS_DIR
FIG_DIR = FIGURES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def scan_curve(el_M, charge, spin, r_grid, basis="cc-pVTZ-DK", ecp=None, use_x2c=True):
    from pyscf import gto, dft
    """Smoothly scan PES along r_grid using previous DM as initial guess for clean continuity."""
    energies = []
    dm_prev = None
    for r in r_grid:
        kw = dict(atom=f"{el_M} 0 0 0; F 0 0 {r:.5f}", basis=basis, charge=charge, spin=spin, verbose=0)
        if ecp:
            kw["ecp"] = ecp
        mol = gto.M(**kw)
        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
        mf.xc = "pbe0"
        if use_x2c and not ecp:
            mf = mf.x2c()
        mf.max_cycle = 160
        mf.level_shift = 0.12
        e = float(mf.kernel(dm0=dm_prev))
        if not mf.converged:
            e = float(mf.kernel())
        if mf.converged:
            dm_prev = mf.make_rdm1()
        energies.append(e)
    return np.array(r_grid, dtype=float), np.array(energies, dtype=float)


def barrier_stats(rs, es):
    """Find local minimum (well) and subsequent Coulomb-explosion barrier height (in eV)."""
    # Find local minima: e[i] < e[i-1] and e[i] < e[i+1]
    mins = [i for i in range(1, len(es) - 1) if es[i] < es[i - 1] and es[i] < es[i + 1]]
    if not mins:
        return {"bound": False}
    imin = mins[0]
    r_min, e_min = float(rs[imin]), float(es[imin])
    # Look for local maximum after imin (Coulomb barrier)
    maxs = [i for i in range(imin + 1, len(es) - 1) if es[i] > es[i - 1] and es[i] > es[i + 1]]
    if maxs:
        imax = maxs[0]
        return {
            "bound": True, "metastable_barrier": True,
            "r_min": round(r_min, 3), "e_min": round(e_min, 6),
            "r_ts": round(float(rs[imax]), 3),
            "barrier_eV": round(float((es[imax] - e_min) * HA_TO_EV), 3),
        }
    return {
        "bound": True, "metastable_barrier": False,
        "r_min": round(r_min, 3), "e_min": round(e_min, 6),
        "De_eV": round(float((es[-1] - e_min) * HA_TO_EV), 3),
    }


def main():
    t0 = time.time()
    data = {}
    print("=== HIGH-RESOLUTION PES SCANS (0.85 A to 4.50 A) ===", flush=True)

    # Dense grids: fine step (0.04-0.06 A) near equilibrium & barrier, wider tail to 4.2-4.5 A
    r_mg = np.concatenate([np.linspace(0.85, 1.85, 21), np.linspace(2.00, 3.80, 10)])
    r_tm = np.concatenate([np.linspace(1.25, 2.35, 21), np.linspace(2.55, 4.20, 9)])
    r_mn = np.concatenate([np.linspace(1.35, 2.45, 20), np.linspace(2.65, 4.50, 9)])
    r_pd = np.concatenate([np.linspace(1.40, 2.40, 18), np.linspace(2.60, 3.60, 6)])

    specs = [
        ("CF+",         "C",  1, 0, r_mg, "cc-pVTZ-DK", None, True),
        ("NF2+",        "N",  2, 0, r_mg, "cc-pVTZ-DK", None, True),
        ("OF3+",        "O",  3, 0, r_mg, "cc-pVTZ-DK", None, True),
        ("TiF3+",       "Ti", 3, 0, r_tm, "cc-pVTZ-DK", None, True),
        ("MoF5+",       "Mo", 5, 0, r_tm, "cc-pVTZ-DK", None, True),
        ("WF5+",        "W",  5, 0, r_tm, "cc-pVTZ-DK", None, True),
        ("CrF5+",       "Cr", 5, 0, r_tm, "cc-pVTZ-DK", None, True),
        ("MnF_7Sigma+", "Mn", 0, 6, r_mn, "cc-pVTZ-DK", None, True),
        ("PdF3+_AE",    "Pd", 3, 0, r_pd, "cc-pVTZ-DK", None, True),
        ("PdF3+_ECP",   "Pd", 3, 0, r_pd, "def2-TZVP",  "def2-TZVP", False),
    ]

    for key, el, q, sp, grid, bas, ecp, x2c in specs:
        rs, es = scan_curve(el, q, sp, grid, bas, ecp, x2c)
        st = barrier_stats(rs, es)
        data[key] = {"R": rs.tolist(), "E_ha": es.tolist(), "stats": st}
        print(f"  {key:13s} ({len(rs):2d} pts, {rs[0]:.2f}-{rs[-1]:.2f} A) | {st}", flush=True)

    with open(os.path.join(OUT_DIR, "highres_pes_curves.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Scan DONE in {time.time()-t0:.1f}s -> highres_pes_curves.json", flush=True)


def plot_pd_triplet_v3(ax, colors):
    """Panel (d): triplet [PdF]3+ from spin-pure matched NEVPT2 envelope (v3 audit)."""
    path = PD_MATCHED_V3_DIR / "corrected_curves.json"
    with open(path) as f:
        pd = json.load(f)
    styles = [
        ("AE_X2C_ccDK", "AE X2C cc-pVTZ-DK", colors[0], "-"),
        ("ECP_def2", "ECP def2-TZVP", colors[1], "--"),
    ]
    for method, lbl, col, ls in styles:
        env = pd["PdF3+_3"][method]["envelope"]
        pts = sorted((p for p in env if p.get("e_nevpt2") is not None), key=lambda p: p["R"])
        rs = np.array([p["R"] for p in pts])
        es = np.array([p["e_nevpt2"] for p in pts])
        e5 = es[-1]
        ax.plot(rs, (es - e5) * HA_TO_EV, "o" + ls, color=col, lw=1.6, ms=3.5, label=lbl)
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_xlim(1.55, 3.25)
    ax.set_xlabel(r"Internuclear distance $R$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$E(R) - E(5.00\,\mathrm{\AA})$ (eV)")
    ax.set_title(r"(d) Triplet $[\mathrm{PdF}]^{3+}$: matched NEVPT2 (repulsive)")
    ax.legend(frameon=False, fontsize=7.0, loc="lower left")


def plot_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(os.path.join(OUT_DIR, "highres_pes_curves.json")) as f:
        data = json.load(f)
    COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), dpi=240)

    # (a) Main-Group Triple Bonds & Coulomb Barriers
    ax = axes[0, 0]
    for idx, (k, lbl) in enumerate([
        ("CF+",  r"$\mathrm{CF}^+$ (bound, $D_e$)"),
        ("NF2+", r"$[\mathrm{NF}]^{2+}$ (barrier)"),
        ("OF3+", r"$[\mathrm{OF}]^{3+}$ (barrier)"),
    ]):
        rs = np.array(data[k]["R"])
        es = np.array(data[k]["E_ha"])
        e0 = data[k]["stats"].get("e_min", es[-1])
        ax.plot(rs, (es - e0) * HA_TO_EV, "o-", color=COLORS[idx], lw=1.6, ms=3.0, label=lbl)
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_xlim(0.85, 3.80)
    ax.set_ylim(-5.5, 13.0)
    ax.set_xlabel(r"Internuclear distance $R$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$E(R) - E(r_e)$ (eV)")
    ax.set_title("(a) Main-Group Triple Bonds & Coulomb Barriers")
    ax.legend(frameon=True, facecolor="white", framealpha=0.96, edgecolor="#cccccc", fontsize=7.2, loc="lower right")

    # (b) d^0 transition-metal wells vs. shallow [CrF]5+
    ax = axes[0, 1]
    for idx, (k, lbl) in enumerate([
        ("TiF3+", r"$[\mathrm{TiF}]^{3+}$ ($3d^0$)"),
        ("MoF5+", r"$[\mathrm{MoF}]^{5+}$ ($4d^0$)"),
        ("WF5+",  r"$[\mathrm{WF}]^{5+}$ ($5d^0$)"),
        ("CrF5+", r"$[\mathrm{CrF}]^{5+}$ ($3d^0$, shallow)"),
    ]):
        rs = np.array(data[k]["R"])
        es = np.array(data[k]["E_ha"])
        e0 = data[k]["stats"].get("e_min", es[np.argmin(np.abs(rs - 1.65))])
        ls = "--" if k == "CrF5+" else "-"
        ax.plot(rs, (es - e0) * HA_TO_EV, "o" + ls, color=COLORS[idx], lw=1.5, ms=2.8, label=lbl)
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_xlim(1.25, 3.05)
    ax.set_ylim(-6.0, 9.5)
    ax.set_xlabel(r"Internuclear distance $R$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Relative energy $\Delta E$ (eV)")
    ax.set_title(r"(b) $d^0$ Transition-Metal Triple-Bond Wells & Cliff")
    ax.legend(frameon=True, facecolor="white", framealpha=0.96, edgecolor="#cccccc", fontsize=7.2, loc="lower left")

    # (c) Neutral MnF (7Sigma+) Full Dissociation Curve (1.35 - 4.50 A)
    ax = axes[1, 0]
    rs = np.array(data["MnF_7Sigma+"]["R"])
    es = np.array(data["MnF_7Sigma+"]["E_ha"])
    e0 = data["MnF_7Sigma+"]["stats"]["e_min"]
    ax.plot(rs, (es - e0) * HA_TO_EV, "o-", color=COLORS[0], lw=1.7, ms=3.5, label=r"$\mathrm{MnF}$ ($X\,^7\Sigma^+$, X2C-PBE0)")
    ax.axvline(1.838, color=COLORS[1], ls="--", lw=1.2, label=r"Expt. $r_e = 1.838\,\mathrm{\AA}$")
    ax.set_xlim(1.35, 4.50)
    ax.set_ylim(-0.3, 6.0)
    ax.set_xlabel(r"Internuclear distance $R$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$E(R) - E(r_e)$ (eV)")
    ax.set_title(r"(c) Neutral $\mathrm{MnF}$ ($X\,^7\Sigma^+$): Full Dissociation Curve")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    plot_pd_triplet_v3(axes[1, 1], COLORS)

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "fig3_highres_pes_barriers.png")
    out_pdf = os.path.join(FIG_DIR, "fig3_highres_pes_barriers.pdf")
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot DONE -> {out_png}", flush=True)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot_curves()
    else:
        main()
