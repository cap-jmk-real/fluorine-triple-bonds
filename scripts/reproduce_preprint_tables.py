#!/usr/bin/env python3
"""Reproduce ChemRxiv v3 Table 1 (non-relativistic cc-pVTZ UHF) and write forensic JSON."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, mp, scf

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from repo_paths import RESULTS_DIR

HA_TO_EV = 27.211386245988
OUT_JSON = RESULTS_DIR / "forensic_mnf_reproduction.json"

print("=== EXACT REPRODUCTION OF PREPRINT TABLE 1 (Non-rel UHF / cc-pVTZ) ===")
table1 = [
    (7, 1.86563091, -1249.370195, -1249.362607, -1249.617938),
    (5, 1.76660973, -1249.217022, -1249.355452, -1249.585352),
    (3, 1.80542986, -1249.128642, None, None),
    (1, 1.85863186, -1249.028371, -1249.392305, -1249.412418),
]
rows = []
for mult, R, e_hf_ref, e_cas_ref, e_mrci_ref in table1:
    spin = mult - 1
    mol = gto.M(atom=f"Mn 0 0 0; F 0 0 {R:.8f}", basis="cc-pVTZ", spin=spin, symmetry=False, verbose=0)
    mf = scf.UHF(mol)
    mf.max_cycle = 200
    e_uhf = mf.kernel()
    ss, _mult_eff = mf.spin_square()
    e_uhf_bs = e_uhf
    ss_bs = ss
    stable = True
    if mult == 1:
        mo_int, _, stable, _ = mf.stability(external=False, return_status=True)
        if not stable:
            mf2 = scf.UHF(mol)
            mf2.max_cycle = 200
            e_uhf_bs = mf2.kernel(mf.make_rdm1(mo_int, mf.mo_occ))
            ss_bs, _ = mf2.spin_square()
    e_mp2 = mp.UMP2(mf).kernel()[0] + e_uhf
    diff_mha = (e_uhf - e_hf_ref) * 1000.0
    row = {
        "mult": mult,
        "R_ang": R,
        "preprint_hf_ha": e_hf_ref,
        "pyscf_uhf_ha": round(e_uhf, 7),
        "diff_uhf_mha": round(diff_mha, 2),
        "s2_uhf": round(float(ss), 3),
        "ump2_ha": round(e_mp2, 7),
        "uhf_stable": bool(stable),
    }
    if mult == 1:
        row.update(
            {
                "broken_sym_uhf_ha": round(e_uhf_bs, 7),
                "delta_bs_ev": round((e_uhf_bs - e_uhf) * HA_TO_EV, 3),
                "s2_broken_sym": round(float(ss_bs), 3),
            }
        )
    rows.append(row)
    print(
        f"Mult={mult} R={R:.4f} | Preprint HF={e_hf_ref:.6f} | PySCF UHF={e_uhf:.6f} "
        f"(diff={diff_mha:+.2f} mHa) | <S^2>={ss:.3f} | UMP2={e_mp2:.6f}"
    )
    if mult == 1:
        print(
            f"  -> Singlet UHF stability: stable={stable} | Broken-sym UHF={e_uhf_bs:.6f} "
            f"(dE={(e_uhf_bs - e_uhf) * HA_TO_EV:+.2f} eV) | <S^2>={ss_bs:.3f}"
        )

# Hypothesized constant basis-contraction offset (septet/quintet UHF vs preprint; triplet is an outlier)
mha_7_5 = [r["diff_uhf_mha"] for r in rows if r["mult"] in (7, 5)]
mean_shift_mha = float(np.mean(mha_7_5)) if mha_7_5 else 0.0

payload = {
    "meta": {
        "pyscf_version": pyscf.__version__,
        "basis": "cc-pVTZ",
        "method": "UHF",
        "scf_max_cycle": 200,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    },
    "table1_rows": rows,
    "uhf_preprint_offset_mha_mult_7_5_mean": round(mean_shift_mha, 2),
    "notes": {
        "singlet_rhf_instability_ev": rows[-1].get("delta_bs_ev"),
        "broken_sym_energy_ha": rows[-1].get("broken_sym_uhf_ha"),
        "hypothesis_basis_contraction_offset_mha": -32,
    },
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nWrote {OUT_JSON}")

print("\n=== EXACT REPRODUCTION OF PREPRINT TABLE 2 (Non-rel ROHF/RHF + CAS(12,9) / cc-pVTZ) ===")
table2 = [
    (7, 1.86828998, -1249.368852, -1249.392614),
    (5, 1.88376701, -1249.214203, -1249.358136),
    (1, 1.85863061, -1249.028371, -1249.129454),
]
for mult, R, e_hf_ref, e_cas_ref in table2:
    spin = mult - 1
    mol = gto.M(atom=f"Mn 0 0 0; F 0 0 {R:.8f}", basis="cc-pVTZ", spin=spin, symmetry=False, verbose=0)
    from pyscf import mcscf

    mf = scf.ROHF(mol)
    mf.max_cycle = 200
    e_rohf = mf.kernel()
    mc = mcscf.CASSCF(mf, 9, 12)
    mc.max_cycle_macro = 60
    e_cas = mc.kernel()[0]
    diff_hf = (e_rohf - e_hf_ref) * 1000.0
    diff_cas = (e_cas - e_cas_ref) * 1000.0
    print(
        f"Mult={mult} R={R:.4f} | Preprint ROHF={e_hf_ref:.6f} vs PySCF={e_rohf:.6f} ({diff_hf:+.2f} mHa) | "
        f"Preprint CAS={e_cas_ref:.6f} vs PySCF CAS(12,9)={e_cas:.6f} ({diff_cas:+.2f} mHa)"
    )