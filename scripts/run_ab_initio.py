#!/usr/bin/env python3
"""
Reproducible ab initio benchmark for:
  "Can Fluorine Form Triple Bonds? Multireference Electronic Structure,
   Bond-Order Decomposition, and Relativistic AE vs. ECP Audit of
   Transition-Metal Monofluorides"

Uses PySCF (X2C scalar relativistic ROHF / CASSCF / SC-NEVPT2 with cc-pVTZ-DK,
plus def2-TZVP ECP comparison for Pd species).
Saves all numerical results to results/ab_initio_results.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf, mrpt, lo

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from repo_paths import RESULTS_DIR

# Constants
HA_TO_EV = 27.211386245988
HA_TO_KJMOL = 2625.4996394799
BOHR_TO_ANG = 0.529177210903
AMU_TO_ME = 1822.888486209
HA_TO_CM1 = 219474.63136320

ATOMIC_MASSES = {
    "C": 12.0,
    "N": 14.003074,
    "F": 18.998403,
    "Mn": 54.938044,
    "Ni": 57.935342,
    "Pd": 105.903480,
}

OUT_DIR = str(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "ab_initio_results.json")


def mayer_bond_order_and_charges(mol, dm):
    """Compute Mayer bond order (total, sigma, pi) and Lowdin/Mulliken charges for a diatomic on z-axis."""
    S = mol.intor("int1e_ovlp")
    if isinstance(dm, np.ndarray) and dm.ndim == 3:
        dm_a, dm_b = dm[0], dm[1]
        dm_tot = dm_a + dm_b
    else:
        dm_tot = dm
        dm_a = dm_b = 0.5 * dm_tot

    PS_a = dm_a @ S
    PS_b = dm_b @ S
    PS_tot = dm_tot @ S

    ao_slices = mol.aoslice_by_atom()
    i0, i1 = ao_slices[0][2], ao_slices[0][3]
    j0, j1 = ao_slices[1][2], ao_slices[1][3]

    # Mayer bond order: B_AB = 2 * sum_{u in A, v in B} [ (P^a S)_{uv}(P^a S)_{vu} + (P^b S)_{uv}(P^b S)_{vu} ]
    mayer_total = 2.0 * (
        np.sum(PS_a[i0:i1, j0:j1] * PS_a[j0:j1, i0:i1].T)
        + np.sum(PS_b[i0:i1, j0:j1] * PS_b[j0:j1, i0:i1].T)
    )

    # Decompose by AO angular momentum projection onto z-axis (m_l = 0 -> sigma, |m_l| = 1 -> pi, |m_l| >= 2 -> delta)
    labels = mol.ao_labels(fmt=False)
    def ml_class(lbl):
        m_str = lbl[3].strip()
        if m_str in ("", "z", "z^2", "z^3", "0"):
            return "sigma"
        if m_str in ("x", "y", "xz", "yz", "xz^2", "yz^2", "+1", "-1"):
            return "pi"
        return "delta"

    classes = [ml_class(l) for l in labels]
    mayer_by_sym = {"sigma": 0.0, "pi": 0.0, "delta": 0.0}
    for sym_name in ("sigma", "pi", "delta"):
        idx_a = [k for k in range(i0, i1) if classes[k] == sym_name]
        idx_b = [k for k in range(j0, j1) if classes[k] == sym_name]
        if idx_a and idx_b:
            sub_a1 = PS_a[np.ix_(idx_a, idx_b)]
            sub_a2 = PS_a[np.ix_(idx_b, idx_a)].T
            sub_b1 = PS_b[np.ix_(idx_a, idx_b)]
            sub_b2 = PS_b[np.ix_(idx_b, idx_a)].T
            mayer_by_sym[sym_name] = float(2.0 * (np.sum(sub_a1 * sub_a2) + np.sum(sub_b1 * sub_b2)))

    # Lowdin charges
    s_eig, s_vec = np.linalg.eigh(S)
    s_half = s_vec @ np.diag(np.sqrt(np.maximum(s_eig, 1e-14))) @ s_vec.T
    P_lowdin = s_half @ dm_tot @ s_half
    pop_lowdin_0 = float(np.trace(P_lowdin[i0:i1, i0:i1]))
    pop_lowdin_1 = float(np.trace(P_lowdin[j0:j1, j0:j1]))
    q_lowdin_0 = float(mol.atom_charge(0) - pop_lowdin_0)
    q_lowdin_1 = float(mol.atom_charge(1) - pop_lowdin_1)

    # Mulliken charges
    pop_mull_1 = float(np.trace(PS_tot[j0:j1, j0:j1]))
    q_mull_1 = float(mol.atom_charge(1) - pop_mull_1)

    return {
        "mayer_total": float(mayer_total),
        "mayer_sigma": mayer_by_sym["sigma"],
        "mayer_pi": mayer_by_sym["pi"],
        "mayer_delta": mayer_by_sym["delta"],
        "q_lowdin_M": q_lowdin_0,
        "q_lowdin_F": q_lowdin_1,
        "q_mulliken_F": q_mull_1,
    }


def run_single_point(el_M, R, charge, spin, basis="cc-pVTZ-DK", ecp=None, use_x2c=True, ncas=None, nelecas=None):
    """Run ROHF (+ optional CASSCF + SC-NEVPT2) single point and extract bonding descriptors."""
    kwargs = dict(
        atom=f"{el_M} 0 0 0; F 0 0 {R:.6f}",
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=False,
        verbose=0,
    )
    if ecp:
        kwargs["ecp"] = ecp
    mol = gto.M(**kwargs)
    mf = scf.ROHF(mol)
    if use_x2c and not ecp:
        mf = mf.x2c()
    mf.max_cycle = 250
    mf.level_shift = 0.25
    mf.diis_space = 12
    e_hf = float(mf.kernel())
    if not mf.converged:
        mf = scf.newton(mf)
        e_hf = float(mf.kernel())

    dm_hf = mf.make_rdm1()
    bond_hf = mayer_bond_order_and_charges(mol, dm_hf)

    res = {
        "R": float(R),
        "charge": charge,
        "spin": spin,
        "mult": spin + 1,
        "basis": basis,
        "ecp": ecp,
        "x2c": bool(use_x2c and not ecp),
        "hf_conv": bool(mf.converged),
        "e_hf": e_hf,
        "bond_hf": bond_hf,
    }

    if ncas is not None and nelecas is not None:
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.max_cycle_macro = 60
        mc.conv_tol = 1e-6
        try:
            e_cas = float(mc.kernel()[0])
            rdm1_cas = mc.fcisolver.make_rdm1s(mc.ci, ncas, mc.nelecas)
            dm_a, dm_b = mc.make_rdm1s()
            bond_cas = mayer_bond_order_and_charges(mol, np.array([dm_a, dm_b]))
            no_occ = np.sort(np.linalg.eigvalsh(rdm1_cas[0] + rdm1_cas[1]))[::-1]
            e_corr = float(mrpt.NEVPT(mc).kernel())
            e_nevpt2 = e_cas + e_corr
            res.update({
                "ncas": ncas,
                "nelecas": nelecas,
                "cas_conv": bool(mc.converged),
                "e_cas": e_cas,
                "e_nevpt2": e_nevpt2,
                "cas_no_occ": [round(float(x), 4) for x in no_occ],
                "bond_cas": bond_cas,
            })
        except Exception as exc:
            print(f"    [WARN] CAS/NEVPT2 error for {el_M}F({charge:+d}, mult={spin+1}) R={R}: {exc}", flush=True)
            res["cas_error"] = str(exc)

    return res


def fit_spectroscopic(el_M, R_list, E_list):
    """Fit quadratic/quartic around minimum of PES to obtain r_e (A) and omega_e (cm^-1)."""
    R_arr = np.array(R_list, dtype=float)
    E_arr = np.array(E_list, dtype=float)
    idx_min = int(np.argmin(E_arr))
    if idx_min == 0 or idx_min == len(R_arr) - 1:
        return {"bound": False, "r_min_grid": float(R_arr[idx_min]), "e_min_grid": float(E_arr[idx_min])}

    # Take up to 5 points around minimum
    lo_i = max(0, idx_min - 2)
    hi_i = min(len(R_arr), idx_min + 3)
    r_sub = R_arr[lo_i:hi_i]
    e_sub = E_arr[lo_i:hi_i]
    deg = min(4, len(r_sub) - 1)
    coeffs = np.polyfit(r_sub, e_sub, deg)
    p = np.poly1d(coeffs)
    dp = np.polyder(p, 1)
    d2p = np.polyder(p, 2)

    roots = np.roots(dp)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r_sub[0] <= r.real <= r_sub[-1]]
    if not real_roots:
        return {"bound": False, "r_min_grid": float(R_arr[idx_min]), "e_min_grid": float(E_arr[idx_min])}

    r_e = min(real_roots, key=lambda r: abs(r - R_arr[idx_min]))
    k_ang = float(d2p(r_e))  # Ha / A^2
    if k_ang <= 0:
        return {"bound": False, "r_min_grid": float(R_arr[idx_min]), "e_min_grid": float(E_arr[idx_min])}

    k_bohr = k_ang * (BOHR_TO_ANG ** 2)  # Ha / bohr^2
    m1 = ATOMIC_MASSES[el_M]
    m2 = ATOMIC_MASSES["F"]
    mu_au = (m1 * m2 / (m1 + m2)) * AMU_TO_ME
    omega_au = np.sqrt(k_bohr / mu_au)
    omega_cm1 = float(omega_au * HA_TO_CM1)
    return {
        "bound": True,
        "r_e_ang": round(float(r_e), 4),
        "e_min_ha": round(float(p(r_e)), 6),
        "omega_e_cm1": round(omega_cm1, 1),
    }


def main():
    t0 = time.time()
    all_results = {"meta": {"pyscf_version": "2.15.1", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}

    # =========================================================================
    # PART 1: Main-group benchmarks (CF+, NF) at X2C-CASSCF/NEVPT2 / cc-pVTZ-DK
    # =========================================================================
    print("=== PART 1: Main-group benchmarks (CF+, NF) ===", flush=True)
    mg = {}
    for name, el, q, spin, ncas, nelecas, r_grid in [
        ("CF+_1Sigma+", "C", 1, 0, 8, 8, [1.05, 1.10, 1.15, 1.20, 1.25, 1.35]),
        ("NF_3Sigma-", "N", 0, 2, 8, 10, [1.20, 1.26, 1.32, 1.38, 1.44, 1.55]),
        ("NF_1Delta", "N", 0, 0, 8, 10, [1.20, 1.26, 1.32, 1.38, 1.44, 1.55]),
    ]:
        pts = [run_single_point(el, r, q, spin, "cc-pVTZ-DK", None, True, ncas, nelecas) for r in r_grid]
        valid = [p for p in pts if "e_nevpt2" in p]
        spec = fit_spectroscopic(el, [p["R"] for p in valid], [p["e_nevpt2"] for p in valid]) if len(valid) >= 3 else {"bound": False}
        best_pt = min(valid, key=lambda p: p["e_nevpt2"]) if valid else pts[0]
        bc = best_pt.get("bond_cas", best_pt["bond_hf"])
        mg[name] = {"points": pts, "spectroscopic_nevpt2": spec, "at_min": best_pt}
        print(f"  {name:14s} | r_e={spec.get('r_e_ang')} A | w_e={spec.get('omega_e_cm1')} cm-1 | "
              f"Mayer={bc['mayer_total']:.3f} (sig={bc['mayer_sigma']:.3f}, pi={bc['mayer_pi']:.3f}) | "
              f"q(F)={bc['q_lowdin_F']:+.3f}", flush=True)
    all_results["main_group_benchmarks"] = mg

    # =========================================================================
    # PART 2: Neutral MnF spin-state PES (7Sigma+, 5Sigma+, 3Sigma+, 1Sigma+)
    # =========================================================================
    print("\n=== PART 2: Neutral MnF Spin-State PES (X2C-CAS(12,9)-NEVPT2 / cc-pVTZ-DK) ===", flush=True)
    mnf_grid = [1.60, 1.70, 1.78, 1.84, 1.90, 2.00, 2.15, 2.40, 2.80]
    mnf = {}
    for spin, label in [(6, "7Sigma+"), (4, "5Sigma+"), (2, "3Sigma+"), (0, "1Sigma+")]:
        pts = []
        for r in mnf_grid:
            pt = run_single_point("Mn", r, 0, spin, "cc-pVTZ-DK", None, True, 9, 12)
            pts.append(pt)
        valid = [p for p in pts if "e_nevpt2" in p]
        spec_nev = fit_spectroscopic("Mn", [p["R"] for p in valid], [p["e_nevpt2"] for p in valid]) if len(valid) >= 3 else {"bound": False}
        spec_cas = fit_spectroscopic("Mn", [p["R"] for p in valid], [p["e_cas"] for p in valid]) if len(valid) >= 3 else {"bound": False}
        spec_hf = fit_spectroscopic("Mn", [p["R"] for p in pts], [p["e_hf"] for p in pts])
        best_pt = min(valid, key=lambda p: p["e_nevpt2"]) if valid else pts[0]
        bc = best_pt.get("bond_cas", best_pt["bond_hf"])
        mnf[label] = {
            "points": pts,
            "spec_nevpt2": spec_nev,
            "spec_cas": spec_cas,
            "spec_hf": spec_hf,
            "at_min": best_pt,
        }
        print(f"  MnF {label:8s} | r_e(NEVPT2)={spec_nev.get('r_e_ang')} A | w_e={spec_nev.get('omega_e_cm1')} cm-1 | "
              f"E_min={spec_nev.get('e_min_ha')} Ha | Mayer={bc['mayer_total']:.3f} "
              f"(sig={bc['mayer_sigma']:.3f}, pi={bc['mayer_pi']:.3f}) | "
              f"q(F)={bc['q_lowdin_F']:+.3f}", flush=True)
    all_results["MnF_neutral"] = mnf

    # =========================================================================
    # PART 3: High-Valent Metal Fluoride Cations (testing F pi-donation)
    # =========================================================================
    print("\n=== PART 3: High-Valent Cations (MnF3+, NiF, NiF2+, NiF3+, PdF, PdF2+, PdF3+) ===", flush=True)
    cations = {}
    cation_specs = [
        # (key, element, charge, spin, ncas, nelecas, r_grid)
        # Mn(IV)F = [MnF]3+ (31 e-, odd): doublet, quartet, sextet
        ("MnF3+_2", "Mn", 3, 1, 9, 9, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        ("MnF3+_4", "Mn", 3, 3, 9, 9, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        ("MnF3+_6", "Mn", 3, 5, 9, 9, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        # Ni(I)F = NiF (37 e-, odd): doublet, quartet
        ("NiF_2", "Ni", 0, 1, 9, 15, [1.60, 1.70, 1.75, 1.82, 1.92, 2.10]),
        ("NiF_4", "Ni", 0, 3, 9, 15, [1.60, 1.70, 1.75, 1.82, 1.92, 2.10]),
        # Ni(III)F = [NiF]2+ (35 e-, odd!): doublet, quartet (cannot be singlet!)
        ("NiF2+_2", "Ni", 2, 1, 9, 13, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        ("NiF2+_4", "Ni", 2, 3, 9, 13, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        # Ni(IV)F = [NiF]3+ (34 e-, even): singlet, triplet, quintet
        ("NiF3+_1", "Ni", 3, 0, 9, 12, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        ("NiF3+_3", "Ni", 3, 2, 9, 12, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
        ("NiF3+_5", "Ni", 3, 4, 9, 12, [1.45, 1.55, 1.65, 1.75, 1.90, 2.20]),
    ]
    for key, el, q, spin, ncas, nelecas, r_grid in cation_specs:
        pts = [run_single_point(el, r, q, spin, "cc-pVTZ-DK", None, True, ncas, nelecas) for r in r_grid]
        valid = [p for p in pts if "e_nevpt2" in p]
        spec = fit_spectroscopic(el, [p["R"] for p in valid], [p["e_nevpt2"] for p in valid]) if len(valid) >= 3 else {"bound": False}
        best_pt = min(valid, key=lambda p: p["e_nevpt2"]) if valid else pts[0]
        cations[key] = {"points": pts, "spec_nevpt2": spec, "at_min": best_pt}
        bc = best_pt.get("bond_cas", best_pt["bond_hf"])
        print(f"  {key:10s} | bound={spec.get('bound')} | r_e={spec.get('r_e_ang', spec.get('r_min_grid'))} A | "
              f"w_e={spec.get('omega_e_cm1')} | Mayer={bc['mayer_total']:.3f} (sig={bc['mayer_sigma']:.3f}, pi={bc['mayer_pi']:.3f}) | "
              f"q(F)={bc['q_lowdin_F']:+.3f}", flush=True)
    all_results["high_valent_3d"] = cations

    # =========================================================================
    # PART 4: AE (X2C / cc-pVTZ-DK) vs. ECP (def2-TZVP + ECP) Audit on PdF species
    # =========================================================================
    print("\n=== PART 4: AE (X2C/cc-pVTZ-DK) vs. ECP (def2-TZVP) Audit on PdF / PdF2+ / PdF3+ ===", flush=True)
    pd_audit = {}
    pd_specs = [
        ("PdF_2", 0, 1, 9, 15, [1.75, 1.85, 1.92, 2.00, 2.15]),
        ("PdF2+_2", 2, 1, 9, 13, [1.65, 1.75, 1.85, 1.95, 2.15]),
        ("PdF2+_4", 2, 3, 9, 13, [1.65, 1.75, 1.85, 1.95, 2.15]),
        ("PdF3+_1", 3, 0, 9, 12, [1.60, 1.70, 1.80, 1.95, 2.15]),
        ("PdF3+_3", 3, 2, 9, 12, [1.60, 1.70, 1.80, 1.95, 2.15]),
    ]
    for key, q, spin, ncas, nelecas, r_grid in pd_specs:
        pd_audit[key] = {}
        for mode, basis, ecp, use_x2c in [
            ("AE_X2C", "cc-pVTZ-DK", None, True),
            ("ECP_def2", "def2-TZVP", "def2-TZVP", False),
        ]:
            pts = [run_single_point("Pd", r, q, spin, basis, ecp, use_x2c, ncas, nelecas) for r in r_grid]
            valid = [p for p in pts if "e_nevpt2" in p]
            spec = fit_spectroscopic("Pd", [p["R"] for p in valid], [p["e_nevpt2"] for p in valid]) if len(valid) >= 3 else {"bound": False}
            best_pt = min(valid, key=lambda p: p["e_nevpt2"]) if valid else pts[0]
            pd_audit[key][mode] = {"points": pts, "spec_nevpt2": spec, "at_min": best_pt}
            bc = best_pt.get("bond_cas", best_pt["bond_hf"])
            print(f"  {key:8s} [{mode:8s}] | bound={spec.get('bound')} | r_e={spec.get('r_e_ang', spec.get('r_min_grid'))} A | "
                  f"w_e={spec.get('omega_e_cm1')} | Mayer={bc['mayer_total']:.3f} (sig={bc['mayer_sigma']:.3f}, pi={bc['mayer_pi']:.3f}) | "
                  f"q(F)={bc['q_lowdin_F']:+.3f}", flush=True)
    all_results["PdF_AE_vs_ECP"] = pd_audit

    all_results["meta"]["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDONE in {all_results['meta']['elapsed_seconds']}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
