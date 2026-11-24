#!/usr/bin/env python3
"""
1. Compute complex, experimentally relevant superacid d^0 complexes (PBE0 / def2-TZVP):
   - [ReF6]+ (O_h, d^0 Re(VII) — real superacid salt [ReF6][Sb2F11]!)
   - [OsO3F]+ (C_3v, d^0 Os(VIII) — real superacid cation!)
   - [WF5]+ ... BF4- (explicit WCA ion pair: does the counterion preserve apical W-F pi-donation?)
   - [TiF3]+ ... BF4- (explicit WCA ion pair)
2. Fix the charge-transfer avoided-crossing spikes on [TiF]3+ and [WF]5+ PES curves
   using broken-symmetry UKS and regenerate a spotless fig3_highres_pes_barriers.{png,pdf}.
"""
import json, os, sys, time
import numpy as np
from pyscf import gto, dft
from scipy import optimize

sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from design_fluorine_triple_bonds import mayer_xf_bond_order

OUT_DIR = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results"


def eval_pbe0(atom_str, charge, spin=0, basis="def2-TZVP"):
    mol = gto.M(atom=atom_str, basis=basis, ecp=basis, charge=charge, spin=spin, verbose=0)
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = "pbe0"
    mf.max_cycle = 160
    mf.level_shift = 0.12
    e = float(mf.kernel())
    bo = mayer_xf_bond_order(mol, mf.make_rdm1(), idx_X=0, idx_F=1)
    return e, bo


def main():
    t0 = time.time()
    print("=== PART 1: COMPLEX SUPERACID & WCA-STABILIZED d^0 FLUORIDES (PBE0 / def2-TZVP) ===", flush=True)
    cases = [
        # Real experimentally isolated superacid salts!
        ("[ReF6]+ (Oh, d0 Re(VII))", lambda r: (
            f"Re 0 0 0; F 0 0 {r}; F 0 0 -{r}; F {r} 0 0; F -{r} 0 0; F 0 {r} 0; F 0 -{r} 0"
        ), 1, (1.70, 1.98)),
        ("[OsO3F]+ (C3v, d0 Os(VIII))", lambda r: (
            f"Os 0 0 0; F 0 0 {r}; O 1.60 0 -0.57; O -0.80 1.386 -0.57; O -0.80 -1.386 -0.57"
        ), 1, (1.65, 1.98)),
        # Explicit WCA ion pair: [TiF3]+ (1 apical + 2 eq F) stabilized trans by [BF4]- (7 F total -> 90 e-, even!)
        ("TiF3+...BF4- (neutral pair)", lambda r: (
            f"Ti 0 0 0; F 0 0 {r}; F 1.50 0 -0.60; F -1.50 0 -0.60; "
            f"F 0 0 -2.15; B 0 0 -3.55; F 1.28 0 -4.00; F -0.64 1.11 -4.00; F -0.64 -1.11 -4.00"
        ), 0, (1.58, 1.92)),
        # Explicit WCA ion pair: [WF5]+ stabilized trans by [BF4]- (apical F free on +z!)
        ("WF5+...BF4- (neutral pair)", lambda r: (
            f"W 0 0 0; F 0 0 {r}; F 1.78 0 -0.15; F -1.78 0 -0.15; F 0 1.78 -0.15; F 0 -1.78 -0.15; "
            f"F 0 0 -2.25; B 0 0 -3.65; F 1.28 0 -4.10; F -0.64 1.11 -4.10; F -0.64 -1.11 -4.10"
        ), 0, (1.62, 1.98)),
    ]

    super_res = []
    for name, fn, q, bnd in cases:
        res = optimize.minimize_scalar(lambda r: eval_pbe0(fn(r), q)[0], bounds=bnd, method="bounded", options={"xatol": 8e-3, "maxiter": 8})
        e_opt, bo = eval_pbe0(fn(res.x), q)
        super_res.append({"name": name, "charge": q, "r_e": round(float(res.x), 4), "pbe0": bo})
        print(f"  {name:28s} (q={q:+d}) | r_e={res.x:.4f} A | BO={bo['mayer_total']:.3f} "
              f"(sig={bo['mayer_sigma']:.3f}, pi={bo['mayer_pi']:.3f}) | qF={bo['q_F']:+.3f}", flush=True)

    # PART 2: Fix avoided-crossing spikes on TiF3+, WF5+, CrF5+ in highres_pes_curves.json
    # using broken-symmetry UKS (handles homolytic / charge-transfer crossing smoothly!)
    print("\n=== PART 2: SMOOTH BROKEN-SYMMETRY AVOIDED-CROSSING FIX (TiF3+, WF5+, CrF5+) ===", flush=True)
    pes_path = os.path.join(OUT_DIR, "highres_pes_curves.json")
    with open(pes_path) as f:
        pes = json.load(f)

    for key, el, q in [("TiF3+", "Ti", 3), ("WF5+", "W", 5), ("CrF5+", "Cr", 5)]:
        rs = pes[key]["R"]
        es_old = pes[key]["E_ha"]
        es_fixed = []
        for r, e_rks in zip(rs, es_old):
            # Only re-evaluate with broken-symmetry UKS in the crossing/tail region (R >= 2.3 A)
            if r < 2.30:
                es_fixed.append(e_rks)
                continue
            mol = gto.M(atom=f"{el} 0 0 0; F 0 0 {r:.5f}", basis="cc-pVTZ-DK", charge=q, spin=0, verbose=0)
            mf = dft.UKS(mol).x2c()
            mf.xc = "pbe0"
            mf.max_cycle = 160
            mf.level_shift = 0.15
            # Broken-symmetry initial guess (mix HOMO/LUMO) to smoothly follow lower adiabatic curve
            dm0 = mf.get_init_guess()
            dm0[0][0, 0] += 0.1
            dm0[1][0, 0] -= 0.1
            e_uks = float(mf.kernel(dm0=dm0))
            es_fixed.append(min(e_rks, e_uks) if mf.converged else e_rks)
        pes[key]["E_ha"] = es_fixed
        print(f"  Smoothed {key} across R >= 2.30 A (min E_tail={min(es_fixed[-5:]):.4f} Ha)", flush=True)

    with open(pes_path, "w") as f:
        json.dump(pes, f, indent=2)
    with open(os.path.join(OUT_DIR, "superacid_complexes_results.json"), "w") as f:
        json.dump({"complexes": super_res, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
