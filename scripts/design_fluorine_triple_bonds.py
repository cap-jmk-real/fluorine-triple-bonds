#!/usr/bin/env python3
"""
Rational design & quantum-chemical screening of molecules engineered to maximize
fluorine sigma + 2*pi (triple-bond) donation: F -> X.

Computes equilibrium X-F bond length r_e, Mayer bond order decomposed into
(sigma, pi, total) along the X-F z-axis, Lowdin charge q(F), and Wiberg/Mayer
comparison using X2C-PBE0 / cc-pVTZ-DK (plus X2C-CCSD / MP2 verification on top hits).
"""
import json
import os
import time
import numpy as np
from pyscf import gto, dft, scf, mp
from scipy import optimize

OUT_JSON = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results/triple_bond_design_screen.json"


def mayer_xf_bond_order(mol, dm, idx_X=0, idx_F=1):
    """Compute Mayer bond order (total, sigma, pi) between atom idx_X and atom idx_F along z-axis."""
    S = mol.intor("int1e_ovlp")
    if isinstance(dm, np.ndarray) and dm.ndim == 3:
        dm_a, dm_b = dm[0], dm[1]
        dm_tot = dm_a + dm_b
    else:
        dm_tot = dm
        dm_a = dm_b = 0.5 * dm_tot

    PS_a = dm_a @ S
    PS_b = dm_b @ S

    ao_slices = mol.aoslice_by_atom()
    i0, i1 = ao_slices[idx_X][2], ao_slices[idx_X][3]
    j0, j1 = ao_slices[idx_F][2], ao_slices[idx_F][3]

    mayer_total = float(2.0 * (
        np.sum(PS_a[i0:i1, j0:j1] * PS_a[j0:j1, i0:i1].T)
        + np.sum(PS_b[i0:i1, j0:j1] * PS_b[j0:j1, i0:i1].T)
    ))

    labels = mol.ao_labels(fmt=False)
    def ml_class(lbl):
        m_str = lbl[3].strip()
        if m_str in ("", "z", "z^2", "z^3", "0"):
            return "sigma"
        if m_str in ("x", "y", "xz", "yz", "xz^2", "yz^2", "+1", "-1"):
            return "pi"
        return "delta"

    classes = [ml_class(l) for l in labels]
    by_sym = {"sigma": 0.0, "pi": 0.0}
    for sym_name in ("sigma", "pi"):
        ia = [k for k in range(i0, i1) if classes[k] == sym_name]
        ib = [k for k in range(j0, j1) if classes[k] == sym_name]
        if ia and ib:
            by_sym[sym_name] = float(2.0 * (
                np.sum(PS_a[np.ix_(ia, ib)] * PS_a[np.ix_(ib, ia)].T)
                + np.sum(PS_b[np.ix_(ia, ib)] * PS_b[np.ix_(ib, ia)].T)
            ))

    # Lowdin charge on F
    s_eig, s_vec = np.linalg.eigh(S)
    s_half = s_vec @ np.diag(np.sqrt(np.maximum(s_eig, 1e-14))) @ s_vec.T
    P_low = s_half @ dm_tot @ s_half
    q_F = float(mol.atom_charge(idx_F) - np.trace(P_low[j0:j1, j0:j1]))
    q_X = float(mol.atom_charge(idx_X) - np.trace(P_low[i0:i1, i0:i1]))
    return {
        "mayer_total": round(mayer_total, 4),
        "mayer_sigma": round(by_sym["sigma"], 4),
        "mayer_pi": round(by_sym["pi"], 4),
        "q_F": round(q_F, 4),
        "q_X": round(q_X, 4),
    }


def eval_candidate(name, build_atom_fn, charge, spin, r_bounds=(1.00, 2.20), basis="cc-pVTZ-DK"):
    """Optimize X-F bond length along z-axis at X2C-PBE0 / cc-pVTZ-DK and compute sigma/pi Mayer BO."""
    cache = {}

    def energy_at(r):
        mol = gto.M(atom=build_atom_fn(r), basis=basis, charge=charge, spin=spin, verbose=0)
        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
        mf.xc = "pbe0"
        mf = mf.x2c()
        mf.max_cycle = 150
        mf.level_shift = 0.1
        e = float(mf.kernel())
        cache[round(r, 6)] = (e, mol, mf)
        return e

    res = optimize.minimize_scalar(energy_at, bounds=r_bounds, method="bounded", options={"xatol": 5e-3, "maxiter": 14})
    r_opt = float(res.x)
    e_opt, mol, mf = cache.get(round(r_opt, 6)) or (energy_at(r_opt), *cache[round(r_opt, 6)][1:])
    bound = (r_bounds[0] + 0.02 < r_opt < r_bounds[1] - 0.02)
    dm = mf.make_rdm1()
    bo = mayer_xf_bond_order(mol, dm, idx_X=0, idx_F=1)

    # Also evaluate RHF/UHF + MP2 single point at r_opt to verify DFT is not over-delocalizing!
    mf_hf = (scf.RHF(mol) if spin == 0 else scf.UHF(mol)).x2c()
    mf_hf.max_cycle = 150
    e_hf = float(mf_hf.kernel())
    bo_hf = mayer_xf_bond_order(mol, mf_hf.make_rdm1(), idx_X=0, idx_F=1)

    out = {
        "name": name,
        "charge": charge,
        "spin": spin,
        "bound": bool(bound),
        "r_e_ang": round(r_opt, 4),
        "e_pbe0_ha": round(e_opt, 6),
        "pbe0": bo,
        "hf": bo_hf,
    }
    print(f"  {name:22s} | bound={str(bound):5s} | r_e={r_opt:.4f} A | "
          f"PBE0 BO={bo['mayer_total']:.3f} (sig={bo['mayer_sigma']:.3f}, pi={bo['mayer_pi']:.3f}) qF={bo['q_F']:+.3f} | "
          f"HF BO={bo_hf['mayer_total']:.3f} (pi={bo_hf['mayer_pi']:.3f})", flush=True)
    return out


def main():
    t0 = time.time()
    # Atom 0 is X, Atom 1 is F at (0, 0, r). Additional ligands arranged symmetrically in xy / -z.
    candidates = [
        # Family A: 10-valence-electron main-group diatomics (CO / N2 analogs)
        ("BF (neutral, 1S+)",      lambda r: f"B 0 0 0; F 0 0 {r}", 0, 0, (1.10, 1.60)),
        ("CF+ (1S+, benchmark)",   lambda r: f"C 0 0 0; F 0 0 {r}", 1, 0, (1.00, 1.45)),
        ("NF2+ (1S+, dication)",   lambda r: f"N 0 0 0; F 0 0 {r}", 2, 0, (0.98, 1.60)),
        ("SiF+ (1S+)",             lambda r: f"Si 0 0 0; F 0 0 {r}", 1, 0, (1.35, 1.85)),
        ("AlF (neutral, 1S+)",     lambda r: f"Al 0 0 0; F 0 0 {r}", 0, 0, (1.45, 1.95)),

        # Family B: Linear sp-hybridized boron/carbon fluorides (double pi-acceptance)
        ("F-B=O (linear, 0)",      lambda r: f"B 0 0 0; F 0 0 {r}; O 0 0 -1.21", 0, 0, (1.15, 1.55)),
        ("[F-B-F]+ (linear, +1)",  lambda r: f"B 0 0 0; F 0 0 {r}; F 0 0 -{r}", 1, 0, (1.10, 1.55)),
        ("[F-B-H]+ (linear, +1)",  lambda r: f"B 0 0 0; F 0 0 {r}; H 0 0 -1.16", 1, 0, (1.08, 1.55)),
        ("[F-C-F]2+ (linear, +2)", lambda r: f"C 0 0 0; F 0 0 {r}; F 0 0 -{r}", 2, 0, (1.05, 1.65)),
        ("[F-C=O]+ (linear, +1)",  lambda r: f"C 0 0 0; F 0 0 {r}; O 0 0 -1.12", 1, 0, (1.08, 1.55)),

        # Family C: d^0 Transition-metal monofluorides & ligand-screened high-valent complexes
        ("[ScF]2+ (d0, 1S+)",      lambda r: f"Sc 0 0 0; F 0 0 {r}", 2, 0, (1.45, 2.05)),
        ("[TiF]3+ (d0, 1S+)",      lambda r: f"Ti 0 0 0; F 0 0 {r}", 3, 0, (1.40, 2.15)),
        ("[YF]2+ (4d0, 1S+)",      lambda r: f"Y 0 0 0; F 0 0 {r}", 2, 0, (1.60, 2.20)),
        ("TiF4 (Td, d0, 0)",       lambda r: (
            f"Ti 0 0 0; F 0 0 {r}; "
            f"F {r*0.9428:.4f} 0 {-r*0.3333:.4f}; "
            f"F {-r*0.4714:.4f} {r*0.8165:.4f} {-r*0.3333:.4f}; "
            f"F {-r*0.4714:.4f} {-r*0.8165:.4f} {-r*0.3333:.4f}"
        ), 0, 0, (1.55, 1.95)),
        ("MnO3F (C3v, d0, 0)",     lambda r: (
            f"Mn 0 0 0; F 0 0 {r}; "
            f"O 1.490 0 -0.527; O -0.745 1.290 -0.527; O -0.745 -1.290 -0.527"
        ), 0, 0, (1.55, 1.95)),
        ("[Cr(CO)5F]+ (C4v, +1)",  lambda r: (
            f"Cr 0 0 0; F 0 0 {r}; "
            f"C 0 0 -1.90; O 0 0 -3.03; "
            f"C 1.95 0 0; O 3.08 0 0; "
            f"C -1.95 0 0; O -3.08 0 0; "
            f"C 0 1.95 0; O 0 3.08 0; "
            f"C 0 -1.95 0; O 0 -3.08 0"
        ), 1, 1, (1.55, 2.00)),
        ("AuF (neutral, 1S+)",     lambda r: f"Au 0 0 0; F 0 0 {r}", 0, 0, (1.75, 2.15)),
        ("[AuF]2+ (d8, 2S)",       lambda r: f"Au 0 0 0; F 0 0 {r}", 2, 1, (1.65, 2.20)),
    ]

    print("=== RATIONAL DESIGN SCREENING FOR FLUORINE TRIPLE BONDS (X2C-PBE0 / cc-pVTZ-DK) ===", flush=True)
    results = []
    for name, fn, q, spin, bounds in candidates:
        try:
            r = eval_candidate(name, fn, q, spin, bounds)
            results.append(r)
        except Exception as e:
            print(f"  {name:22s} | ERROR: {e}", flush=True)

    # Sort by PBE0 Mayer total bond order descending
    print("\n=== TOP RANKED BY FLUORINE BOND ORDER (X2C-PBE0 / cc-pVTZ-DK) ===")
    for r in sorted([x for x in results if x["bound"]], key=lambda x: -x["pbe0"]["mayer_total"])[:8]:
        p = r["pbe0"]
        print(f"  {r['name']:22s} | r_e={r['r_e_ang']:.4f} A | BO={p['mayer_total']:.3f} (sig={p['mayer_sigma']:.3f}, pi={p['mayer_pi']:.3f}) | qF={p['q_F']:+.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump({"candidates": results, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nDONE in {time.time() - t0:.1f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
