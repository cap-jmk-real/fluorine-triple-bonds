#!/usr/bin/env python3
"""
Superacid-abstracted d0 fluorocations and mechanical [F-VH3]+ compression scans.
Writes results/superacid_and_cage_results.json (PBE0 / def2-TZVP).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pyscf import dft, gto
from scipy import optimize

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from design_fluorine_triple_bonds import mayer_xf_bond_order
from repo_paths import RESULTS_DIR

DEFAULT_OUT = RESULTS_DIR / "superacid_and_cage_results.json"
METHOD = "PBE0"
BASIS = "def2-TZVP"
HA_TO_EV = 27.211386245988


def run_sp(atom_str, charge, basis=BASIS):
    mol = gto.M(atom=atom_str, basis=basis, ecp=basis, charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.max_cycle = 150
    mf.level_shift = 0.1
    e = float(mf.kernel())
    bo = mayer_xf_bond_order(mol, mf.make_rdm1(), idx_X=0, idx_F=1)
    return e, bo, mol


def fvh3_str(r):
    return f"V 0 0 0; F 0 0 {r}; H 1.518 0 -0.567; H -0.759 1.315 -0.567; H -0.759 -1.315 -0.567"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    payload = {"method": METHOD, "basis": BASIS, "monocations": [], "compression_scans": []}

    if not args.dry_run:
        print("=== SUPERACID-GENERATED d0 FLUOROCATIONS ===", flush=True)
        cases = [
            ("[WF5]+ (C4v, +1)", lambda r: f"W 0 0 0; F 0 0 {r}; F 1.78 0 -0.25; F -1.78 0 -0.25; F 0 1.78 -0.25; F 0 -1.78 -0.25", 1, (1.55, 1.95)),
        ]
        for name, fn, q, bnd in cases:
            res = optimize.minimize_scalar(
                lambda r: run_sp(fn(r), q)[0], bounds=bnd, method="bounded", options={"xatol": 8e-3, "maxiter": 9}
            )
            e_opt, bo, _ = run_sp(fn(res.x), q)
            payload["monocations"].append(
                {
                    "id": name,
                    "charge": q,
                    "method": METHOD,
                    "basis": BASIS,
                    "r_e_ang": round(float(res.x), 4),
                    "mayer_total": round(bo["mayer_total"], 4),
                    "mayer_sigma": round(bo["mayer_sigma"], 4),
                    "mayer_pi": round(bo["mayer_pi"], 4),
                    "q_F": round(bo["q_F"], 4),
                }
            )
            print(f"  {name} r_e={res.x:.4f}", flush=True)

        print("=== CAGE COMPRESSION [F-VH3]+ ===", flush=True)
        e_eq, bo_eq, _ = run_sp(fvh3_str(1.62), 1)
        scan = []
        for r in [1.70, 1.62, 1.50, 1.40, 1.32]:
            e, bo, _ = run_sp(fvh3_str(r), 1)
            scan.append(
                {
                    "R_ang": r,
                    "delta_E_eV": round((e - e_eq) * HA_TO_EV, 2),
                    "mayer_total": round(bo["mayer_total"], 4),
                    "mayer_sigma": round(bo["mayer_sigma"], 4),
                    "mayer_pi": round(bo["mayer_pi"], 4),
                    "q_F": round(bo["q_F"], 4),
                }
            )
        payload["compression_scans"].append(
            {
                "id": "[F-VH3]+ compression",
                "reference_R_ang": 1.62,
                "points": scan,
            }
        )

    payload["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()