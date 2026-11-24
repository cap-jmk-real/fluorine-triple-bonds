#!/usr/bin/env python3
"""Cheap complementary checks: basis convergence and Boys/IBO localization for [OF]3+."""
import json
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from design_fluorine_triple_bonds import collect_runtime_meta, eval_candidate, mayer_xf_bond_order
from repo_paths import RESULTS_DIR

OUT_JSON = RESULTS_DIR / "complementary_bonding_checks.json"


def localization_summary(mol, mf):
    """Boys and IBO Mayer indices after re-building the density in the localized MO basis."""
    from pyscf import lo

    dm = mf.make_rdm1()
    mo = mf.mo_coeff
    occ = mf.mo_occ

    def mayer_from_localized(localized_mo):
        dm_loc = localized_mo @ np.diag(occ) @ localized_mo.T
        dm_loc = dm_loc + dm_loc.T
        bo = mayer_xf_bond_order(mol, dm_loc)
        return bo["mayer_total"]

    boys_mo = lo.Boys(mol, mo).kernel()
    ibo_mo = lo.ibo.ibo(mol, mo)

    return {
        "mayer_at_opt": mayer_xf_bond_order(mol, dm),
        "boys_mayer_equiv": round(mayer_from_localized(boys_mo), 4),
        "ibo_mayer_equiv": round(mayer_from_localized(ibo_mo), 4),
    }


def main():
    basis_sets = ["cc-pVDZ-DK", "cc-pVTZ-DK"]
    basis_block = {}
    for basis in basis_sets:
        entry = eval_candidate(
            f"[OF]3+ ({basis})",
            lambda r: f"O 0 0 0; F 0 0 {r}",
            3,
            0,
            (0.95, 1.80),
            basis=basis,
        )
        basis_block[basis] = {
            "r_e_ang": entry["r_e_ang"],
            "pbe0_bo": entry["pbe0"]["mayer_total"],
            "hf_bo": entry["hf"]["mayer_total"],
        }

    # Re-run VXZ at equilibrium for localization (same geometry as VTZ optimum)
    r_vtz = basis_block["cc-pVTZ-DK"]["r_e_ang"]
    from pyscf import dft, gto

    mol = gto.M(
        atom=f"O 0 0 0; F 0 0 {r_vtz}",
        basis="cc-pVTZ-DK",
        charge=3,
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol).x2c()
    mf.xc = "pbe0"
    mf.max_cycle = 150
    mf.kernel()
    loc = localization_summary(mol, mf)

    payload = {
        "meta": collect_runtime_meta(),
        "of3plus_basis": basis_block,
        "of3plus_localization_at_vtz_re": loc,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()