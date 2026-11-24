#!/usr/bin/env python3
"""
Test Julian's two stabilization strategies:
  (A) Superacid-abstracted d^0 fluorocations (MF_n + SbF5 -> [MF_{n-1}]+ [SbF6]-):
      [TiF3]+ (D3h, +1), [VF4]+ (Td, +1), [MoF5]+ (C4v, +1), [WF5]+ (C4v, +1)
  (B) Mechanical Cage / Endohedral Vise Compression:
      Compressing R(M-F) in neutral F-TiH3 and monocationic [F-VH3]+ from r_e down to 1.35 A
      to quantify how much caging boosts sigma/pi Mayer bond order vs. compression energy cost.
"""
import sys, time
import numpy as np
from pyscf import gto, dft
from scipy import optimize
sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from design_fluorine_triple_bonds import mayer_xf_bond_order

def run_sp(atom_str, charge, basis="def2-TZVP"):
    mol = gto.M(atom=atom_str, basis=basis, ecp=basis, charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.max_cycle = 150
    mf.level_shift = 0.1
    e = float(mf.kernel())
    bo = mayer_xf_bond_order(mol, mf.make_rdm1(), idx_X=0, idx_F=1)
    return e, bo

print("=== PART A: SUPERACID-GENERATED d^0 FLUOROCATIONS (MF_n + SbF5 -> [MF_{n-1}]+) ===", flush=True)
superacid_cases = [
    ("[TiF3]+ (D3h, +1)", lambda r: f"Ti 0 0 0; F 0 0 {r}; F {r*0.866:.4f} 0 {-r*0.5:.4f}; F {-r*0.866:.4f} 0 {-r*0.5:.4f}", 1, (1.50, 1.85)),
    ("[VF4]+ (Td, +1)",   lambda r: f"V 0 0 0; F 0 0 {r}; F {r*0.9428:.4f} 0 {-r*0.3333:.4f}; F {-r*0.4714:.4f} {r*0.8165:.4f} {-r*0.3333:.4f}; F {-r*0.4714:.4f} {-r*0.8165:.4f} {-r*0.3333:.4f}", 1, (1.45, 1.85)),
    ("[MoF5]+ (C4v, +1)", lambda r: f"Mo 0 0 0; F 0 0 {r}; F 1.75 0 -0.25; F -1.75 0 -0.25; F 0 1.75 -0.25; F 0 -1.75 -0.25", 1, (1.55, 1.95)),
    ("[WF5]+ (C4v, +1)",  lambda r: f"W 0 0 0; F 0 0 {r}; F 1.78 0 -0.25; F -1.78 0 -0.25; F 0 1.78 -0.25; F 0 -1.78 -0.25", 1, (1.55, 1.95)),
]
for name, fn, q, bnd in superacid_cases:
    res = optimize.minimize_scalar(lambda r: run_sp(fn(r), q)[0], bounds=bnd, method="bounded", options={"xatol": 8e-3, "maxiter": 9})
    e_opt, bo = run_sp(fn(res.x), q)
    print(f"  {name:20s} | r_e={res.x:.4f} A | BO={bo['mayer_total']:.3f} (sig={bo['mayer_sigma']:.3f}, pi={bo['mayer_pi']:.3f}) | qF={bo['q_F']:+.3f}", flush=True)

print("\n=== PART B: MECHANICAL CAGE / VISE COMPRESSION ON [F-VH3]+ (q=+1, d0) ===", flush=True)
def fvh3_str(r):
    return f"V 0 0 0; F 0 0 {r}; H 1.518 0 -0.567; H -0.759 1.315 -0.567; H -0.759 -1.315 -0.567"

e_eq, _ = run_sp(fvh3_str(1.62), 1)
for r in [1.70, 1.62, 1.50, 1.40, 1.32]:
    e, bo = run_sp(fvh3_str(r), 1)
    de_ev = (e - e_eq) * 27.2114
    print(f"  R={r:.2f} A | dE={de_ev:+6.2f} eV | BO={bo['mayer_total']:.3f} (sig={bo['mayer_sigma']:.3f}, pi={bo['mayer_pi']:.3f}) | qF={bo['q_F']:+.3f}", flush=True)
