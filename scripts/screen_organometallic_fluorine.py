#!/usr/bin/env python3
"""
Test wild neutral & monocationic organometallic / coordination complexes engineered
to stabilize F -> M double-pi donation (M === F) without bare-cation Coulomb explosion!

Strategy: d^0 early/mid transition metals with ZERO pi-donor competition in the equatorial belt:
  - Pure sigma-donor hydrides: F-TiH3 (0), F-VH4 (0), F-MoH5 (0), [F-TiH2]+ (+1), [F-VH3]+ (+1), [F-MoH4]+ (+1)
  - Strong inductive / pi-acceptor cyanides: F-Ti(CN)3 (0), F-V(CN)4 (0), [F-Ti(CN)2]+ (+1), [F-V(CN)3]+ (+1), [F-Mo(CN)4]+ (+1)
  - Organometallic carbonyl / borane / main-group: [F-B-CN]+ (+1), [F-C(H)]2+ vs F-Ti(CH3)3 (0)
"""
import json
import sys
import time
import numpy as np
from pyscf import gto, dft
from scipy import optimize

sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from design_fluorine_triple_bonds import mayer_xf_bond_order

OUT_JSON = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results/organometallic_fluorine_screen.json"


def make_c3v_ligands(lig_atoms_fn, r_ml, z_tilt=-0.35):
    """Build 3 equatorial ligands in C3v symmetry around z-axis."""
    lines = []
    rho = np.sqrt(1.0 - z_tilt ** 2)
    for k in range(3):
        ang = 2.0 * np.pi * k / 3.0
        ux, uy, uz = rho * np.cos(ang), rho * np.sin(ang), z_tilt
        lines.append(lig_atoms_fn(ux, uy, uz, r_ml))
    return "; ".join(lines)


def make_c4v_ligands(lig_atoms_fn, r_ml, z_tilt=-0.20):
    """Build 4 equatorial ligands in C4v symmetry around z-axis."""
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


def eval_complex(name, build_fn, charge, spin=0, r_bounds=(1.45, 2.05), basis="def2-TZVP"):
    cache = {}
    def e_at(r):
        mol = gto.M(atom=build_fn(r), basis=basis, ecp=basis, charge=charge, spin=spin, verbose=0)
        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
        mf.xc = "pbe0"
        mf.max_cycle = 150
        mf.level_shift = 0.15
        e = float(mf.kernel())
        cache[round(r, 6)] = (e, mol, mf)
        return e

    res = optimize.minimize_scalar(e_at, bounds=r_bounds, method="bounded", options={"xatol": 6e-3, "maxiter": 11})
    r_opt = float(res.x)
    e_opt, mol, mf = cache.get(round(r_opt, 6)) or (e_at(r_opt), *cache[round(r_opt, 6)][1:])
    bo = mayer_xf_bond_order(mol, mf.make_rdm1(), idx_X=0, idx_F=1)
    out = {"name": name, "charge": charge, "r_e": round(r_opt, 4), "pbe0": bo}
    print(f"  {name:26s} (q={charge:+d}) | r_e={r_opt:.4f} A | BO={bo['mayer_total']:.3f} "
          f"(sig={bo['mayer_sigma']:.3f}, pi={bo['mayer_pi']:.3f}) | qF={bo['q_F']:+.3f}", flush=True)
    return out


def main():
    t0 = time.time()
    print("=== WILD NEUTRAL & MONOCATIONIC ORGANOMETALLIC / COORDINATION COMPLEXES (PBE0 / def2-TZVP) ===", flush=True)
    candidates = [
        # Neutral d^0 pure sigma-donor hydrides (zero pi-competition!)
        ("F-TiH3 (C3v, d0, 0)",       lambda r: f"Ti 0 0 0; F 0 0 {r}; " + make_c3v_ligands(h_lig, 1.70), 0, (1.50, 1.95)),
        ("F-VH4 (C4v, d0, 0)",        lambda r: f"V 0 0 0; F 0 0 {r}; "  + make_c4v_ligands(h_lig, 1.63), 0, (1.45, 1.95)),
        ("F-MoH5 (C4v+ax, d0, 0)",    lambda r: f"Mo 0 0 0; F 0 0 {r}; H 0 0 -1.70; " + make_c4v_ligands(h_lig, 1.70, 0.0), 0, (1.55, 2.05)),

        # Monocationic d^0 hydrides (+1, bottleable with WCA!)
        ("[F-TiH2]+ (C2v, d0, +1)",   lambda r: f"Ti 0 0 0; F 0 0 {r}; H 1.45 0 -0.85; H -1.45 0 -0.85", 1, (1.45, 1.90)),
        ("[F-VH3]+ (C3v, d0, +1)",    lambda r: f"V 0 0 0; F 0 0 {r}; " + make_c3v_ligands(h_lig, 1.62), 1, (1.40, 1.90)),
        ("[F-MoH4]+ (C4v, d0, +1)",   lambda r: f"Mo 0 0 0; F 0 0 {r}; " + make_c4v_ligands(h_lig, 1.68), 1, (1.50, 1.95)),
        ("[F-WH4]+ (C4v, d0, +1)",    lambda r: f"W 0 0 0; F 0 0 {r}; " + make_c4v_ligands(h_lig, 1.70), 1, (1.50, 1.95)),

        # Neutral & Monocationic d^0 Cyanide / Pi-Acceptor Belt Complexes!
        ("F-Ti(CN)3 (C3v, d0, 0)",    lambda r: f"Ti 0 0 0; F 0 0 {r}; " + make_c3v_ligands(cn_lig, 2.08), 0, (1.50, 1.95)),
        ("[F-Ti(CN)2]+ (C2v, d0,+1)", lambda r: f"Ti 0 0 0; F 0 0 {r}; C 1.80 0 -1.00; N 2.80 0 -1.56; C -1.80 0 -1.00; N -2.80 0 -1.56", 1, (1.45, 1.90)),
        ("[F-V(CN)3]+ (C3v, d0, +1)", lambda r: f"V 0 0 0; F 0 0 {r}; " + make_c3v_ligands(cn_lig, 2.02), 1, (1.40, 1.90)),
        ("[F-Mo(CN)4]+ (C4v,d0,+1)",  lambda r: f"Mo 0 0 0; F 0 0 {r}; " + make_c4v_ligands(cn_lig, 2.10), 1, (1.50, 1.95)),

        # Main-group linear pi-acceptor monocation
        ("[F-B-CN]+ (linear, +1)",    lambda r: f"B 0 0 0; F 0 0 {r}; C 0 0 -1.42; N 0 0 -2.58", 1, (1.10, 1.45)),
    ]

    results = []
    for name, fn, q, bounds in candidates:
        try:
            results.append(eval_complex(name, fn, q, 0, bounds))
        except Exception as e:
            print(f"  {name:26s} | ERROR: {e}", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({"candidates": results, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.1f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
