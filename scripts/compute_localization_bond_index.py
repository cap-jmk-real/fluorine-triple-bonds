#!/usr/bin/env python3
"""
Compute localization-based bond indices (PySCF IBO) at DFT minima for extreme cation candidates.

Default: write a stub JSON with geometries from extreme_bo_results.json (no heavy QC).
  python compute_localization_bond_index.py --stub

Smoke test (fast):
  python compute_localization_bond_index.py --smoke

Full run (slow; not for CI):
  python compute_localization_bond_index.py --run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from repo_paths import RESULTS_DIR

OUT = RESULTS_DIR / "localization_bond_indices.json"
TARGET_SUBSTR = ("[OF]3+", "[MoF]5+", "[WF]5+")


def ibo_pair_index(mol, mf) -> dict:
    from pyscf import lo

    ibos = lo.ibo.ibo(mf)
    # IBO 0 / 1 are on atoms 0 (metal) and 1 (F) in diatomic layout
    s = mol.intor("int1e_ovlp")
    c = ibos @ s @ ibos.T
    raw = float(c[0, 1])
    return {"ibo_raw_01": round(raw, 4), "n_ibo": int(ibos.shape[0])}


def smoke_h2() -> dict:
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    idx = ibo_pair_index(mol, mf)
    idx["smoke"] = True
    return idx


def stub_from_extreme() -> dict:
    extreme = json.loads((RESULTS_DIR / "extreme_bo_results.json").read_text())
    rows = []
    for sub in TARGET_SUBSTR:
        c = next(x for x in extreme["candidates"] if sub in x["name"])
        rows.append(
            {
                "species": c["name"],
                "r_e_ang": c["r_e_ang"],
                "mayer_pbe0": c["pbe0"]["mayer_total"],
                "status": "pending",
                "method": None,
                "ibo": None,
            }
        )
    return {"note": "stub from extreme_bo_results; run with --run on compute host", "entries": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        payload = {"smoke_test": smoke_h2()}
    elif args.run:
        raise SystemExit("Full IBO runs at cation minima are not enabled in CI; use compute host.")
    else:
        payload = stub_from_extreme()
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()