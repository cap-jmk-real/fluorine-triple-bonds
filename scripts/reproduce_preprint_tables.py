#!/usr/bin/env python3
"""Exact 1-to-1 reproduction of Tables 1 & 2 from ChemRxiv v3 (non-relativistic cc-pVTZ)."""
import numpy as np
from pyscf import gto, scf, mp, mcscf

print("=== EXACT REPRODUCTION OF PREPRINT TABLE 1 (Non-rel UHF / cc-pVTZ) ===")
table1 = [
    (7, 1.86563091, -1249.370195, -1249.362607, -1249.617938),
    (5, 1.76660973, -1249.217022, -1249.355452, -1249.585352),
    (3, 1.80542986, -1249.128642, None,         None),
    (1, 1.85863186, -1249.028371, -1249.392305, -1249.412418),
]
for mult, R, e_hf_ref, e_cas_ref, e_mrci_ref in table1:
    spin = mult - 1
    mol = gto.M(atom=f"Mn 0 0 0; F 0 0 {R:.8f}", basis="cc-pVTZ", spin=spin, symmetry=False, verbose=0)
    mf = scf.UHF(mol)
    mf.max_cycle = 200
    e_uhf = mf.kernel()
    ss, mult_eff = mf.spin_square()
    # Check stability (broken-symmetry UHF for singlet!)
    if mult == 1:
        mo_int, _, stable, _ = mf.stability(external=False, return_status=True)
        e_uhf_bs = e_uhf
        ss_bs = ss
        if not stable:
            mf2 = scf.UHF(mol)
            mf2.max_cycle = 200
            e_uhf_bs = mf2.kernel(mf.make_rdm1(mo_int, mf.mo_occ))
            ss_bs, _ = mf2.spin_square()
    else:
        stable, e_uhf_bs, ss_bs = True, e_uhf, ss

    # Also UMP2 check (preprint §3)
    e_mp2 = mp.UMP2(mf).kernel()[0] + e_uhf
    diff_mha = (e_uhf - e_hf_ref) * 1000.0
    print(f"Mult={mult} R={R:.4f} | Preprint HF={e_hf_ref:.6f} | PySCF UHF={e_uhf:.6f} (diff={diff_mha:+.2f} mHa) | <S^2>={ss:.3f} | UMP2={e_mp2:.6f}")
    if mult == 1:
        print(f"  -> Singlet UHF stability check: stable={stable} | Broken-sym UHF={e_uhf_bs:.6f} (dE={(e_uhf_bs-e_uhf)*27.2114:+.2f} eV!) | <S^2>={ss_bs:.3f}")

print("\n=== EXACT REPRODUCTION OF PREPRINT TABLE 2 (Non-rel ROHF/RHF + CAS(12,9) / cc-pVTZ) ===")
table2 = [
    (7, 1.86828998, -1249.368852, -1249.392614),
    (5, 1.88376701, -1249.214203, -1249.358136),
    (1, 1.85863061, -1249.028371, -1249.129454),
]
for mult, R, e_hf_ref, e_cas_ref in table2:
    spin = mult - 1
    mol = gto.M(atom=f"Mn 0 0 0; F 0 0 {R:.8f}", basis="cc-pVTZ", spin=spin, symmetry=False, verbose=0)
    mf = scf.ROHF(mol)
    mf.max_cycle = 200
    e_rohf = mf.kernel()
    mc = mcscf.CASSCF(mf, 9, 12)
    mc.max_cycle_macro = 60
    e_cas = mc.kernel()[0]
    diff_hf = (e_rohf - e_hf_ref) * 1000.0
    diff_cas = (e_cas - e_cas_ref) * 1000.0
    print(f"Mult={mult} R={R:.4f} | Preprint ROHF={e_hf_ref:.6f} vs PySCF={e_rohf:.6f} ({diff_hf:+.2f} mHa) | Preprint CAS={e_cas_ref:.6f} vs PySCF CAS(12,9)={e_cas:.6f} ({diff_cas:+.2f} mHa)")
