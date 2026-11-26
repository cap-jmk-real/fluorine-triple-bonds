#!/usr/bin/env python3
"""
Screen organometallic / coordination complexes for elevated F->M pi-donation (PBE0 / def2-TZVP).
Writes results/organometallic_fluorine_screen.json with r_e and Mayer sigma/pi/total.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto
from scipy import optimize

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from design_fluorine_triple_bonds import mayer_xf_bond_order
from repo_paths import RESULTS_DIR

DEFAULT_OUT = RESULTS_DIR / "organometallic_fluorine_screen.json"
METHOD = "PBE0"
BASIS = "def2-TZVP"


def make_c3v_ligands(lig_atoms_fn, r_ml, z_tilt=-0.35):
    lines = []
    rho = np.sqrt(1.0 - z_tilt ** 2)
    for k in range(3):
        ang = 2.0 * np.pi * k / 3.0
        ux, uy, uz = rho * np.cos(ang), rho * np.sin(ang), z_tilt
        lines.append(lig_atoms_fn(ux, uy, uz, r_ml))
    return "; ".join(lines)


def make_c4v_ligands(lig_atoms_fn, r_ml, z_tilt=-0.20):
    lines = []
    rho = np.sqrt(1.0 - z_tilt ** 2)
    for k in range(4):
        ang = 2.0 * np.pi * k / 4.0
        ux, uy, uz = rho * np.cos(ang), rho * np.sin(ang), z_tilt
        lines.append(lig_atoms_fn(ux, uy, uz, r_ml))
    return "; ".join(lines)


def h_lig(ux, uy, uz, r):
    return f"H {ux*r:.4f} {uy*r:.4f} {uz*r:.4f}"


def cn_lig(ux, uy, uz, r):
    rN = r + 1.16
    return f"C {ux*r:.4f} {uy*r:.4f} {uz*r:.4f}; N {ux*rN:.4f} {uy*rN:.4f} {uz*rN:.4f}"


def eval_complex(name, build_fn, charge, spin=0, r_bounds=(1.45, 2.05)):
    cache = {}

    def e_at(r):
        mol = gto.M(atom=build_fn(r), basis=BASIS, ecp=BASIS, charge=charge, spin=spin, verbose=0)
        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
        mf.xc = "pbe0"
        mf.max_cycle = 150
        mf.level_shift = 0.15
        e = float(mf.kernel())
        cache[round(r, 6)] = (e, mol, mf)
        return e

    res = optimize.minimize_scalar(
        e_at, bounds=r_bounds, method="bounded", options={"xatol": 6e-3, "maxiter": 11}
    )
    r_opt = float(res.x)
    e_opt, mol, mf = cache.get(round(r_opt, 6)) or (e_at(r_opt), *cache[round(r_opt, 6)][1:])
    bo = mayer_xf_bond_order(mol, mf.make_rdm1(), idx_X=0, idx_F=1)
    out = {
        "id": name,
        "charge": charge,
        "method": METHOD,
        "basis": BASIS,
        "r_e_ang": round(r_opt, 4),
        "mayer_total": round(bo["mayer_total"], 4),
        "mayer_sigma": round(bo["mayer_sigma"], 4),
        "mayer_pi": round(bo["mayer_pi"], 4),
        "q_F": round(bo["q_F"], 4),
    }
    print(
        f"  {name:26s} (q={charge:+d}) | r_e={r_opt:.4f} A | BO={bo['mayer_total']:.3f} "
        f"(sig={bo['mayer_sigma']:.3f}, pi={bo['mayer_pi']:.3f}) | qF={bo['q_F']:+.3f}",
        flush=True,
    )
    return out


def run_screen() -> dict:
    candidates = [
        ("F-TiH3 (C3v, d0, 0)", lambda r: f"Ti 0 0 0; F 0 0 {r}; " + make_c3v_ligands(h_lig, 1.70), 0, (1.50, 1.95)),
        ("F-VH4 (C4v, d0, 0)", lambda r: f"V 0 0 0; F 0 0 {r}; " + make_c4v_ligands(h_lig, 1.63), 0, (1.45, 1.95)),
        ("F-MoH5 (C4v+ax, d0, 0)", lambda r: f"Mo 0 0 0; F 0 0 {r}; H 0 0 -1.70; " + make_c4v_ligands(h_lig, 1.70, 0.0), 0, (1.55, 2.05)),
        ("[F-TiH2]+ (C2v, d0, +1)", lambda r: f"Ti 0 0 0; F 0 0 {r}; H 1.45 0 -0.85; H -1.45 0 -0.85", 1, (1.45, 1.90)),
        ("[F-VH3]+ (C3v, d0, +1)", lambda r: f"V 0 0 0; F 0 0 {r}; " + make_c3v_ligands(h_lig, 1.62), 1, (1.40, 1.90)),
        ("[F-MoH4]+ (C4v, d0, +1)", lambda r: f"Mo 0 0 0; F 0 0 {r}; " + make_c4v_ligands(h_lig, 1.68), 1, (1.50, 1.95)),
        ("[F-WH4]+ (C4v, d0, +1)", lambda r: f"W 0 0 0; F 0 0 {r}; " + make_c4v_ligands(h_lig, 1.70), 1, (1.50, 1.95)),
        ("[F-B-CN]+ (linear, +1)", lambda r: f"B 0 0 0; F 0 0 {r}; C 0 0 -1.42; N 0 0 -2.58", 1, (1.10, 1.45)),
    ]
    results = []
    for name, fn, q, bounds in candidates:
        try:
            results.append(eval_complex(name, fn, q, 0, bounds))
        except Exception as e:
            print(f"  {name:26s} | ERROR: {e}", flush=True)
    return {"method": METHOD, "basis": BASIS, "candidates": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="Write schema only, no QC")
    args = ap.parse_args()
    t0 = time.time()
    if args.dry_run:
        payload = {"method": METHOD, "basis": BASIS, "candidates": [], "note": "dry-run schema"}
    else:
        print("=== ORGANOMETALLIC FLUORINE SCREEN (PBE0 / def2-TZVP) ===", flush=True)
        payload = run_screen()
    payload["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()