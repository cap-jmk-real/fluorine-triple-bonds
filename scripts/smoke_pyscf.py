import time
import numpy as np
from pyscf import gto, scf, mcscf, mrpt

t0 = time.time()
# Quick test: MnF at R = 1.84 A, spin = 6 (septet) vs spin = 0 (singlet), cc-pVTZ
for spin, label in [(6, "7Sigma+"), (4, "5Sigma+"), (0, "1Sigma+")]:
    mol = gto.M(
        atom="Mn 0 0 0; F 0 0 1.84",
        basis="cc-pVTZ",
        spin=spin,
        charge=0,
        symmetry=True,
        verbose=0,
    )
    mf = scf.ROHF(mol).x2c()
    mf.max_cycle = 200
    mf.level_shift = 0.2
    e_hf = mf.kernel()
    # Active space: Mn 3d, 4s (6 orb) + F 2p (3 orb) = 9 orb; electrons = 7 (Mn 3d5 4s2) + 5 (F 2p5) = 12e -> CAS(12,9)
    mc = mcscf.CASSCF(mf, 9, 12)
    mc.max_cycle_macro = 60
    e_cas = mc.kernel()[0]
    e_corr = mrpt.NEVPT(mc).kernel()
    e_nevpt2 = e_cas + e_corr
    # Natural orbital occupancies -> effective bond order estimate
    occ = mc.mo_occ
    print(f"MnF {label:8s} | conv={mf.converged}/{mc.converged} | ROHF={e_hf:.6f} | CAS(12,9)={e_cas:.6f} | NEVPT2={e_nevpt2:.6f}")

print(f"Elapsed: {time.time() - t0:.1f}s")
