#!/usr/bin/env python3
"""
Investigating How Ligands Increase Fluorine's pi-Donor Quality (Paper 1):
Two complementary mechanisms tested quantitatively at DF-PBE0/def2-SVP (ECP on W):

Axis A — Equatorial & Trans-Metal Ligand Engineering on d0 [F-V(L)3]+ and [F-W(L)4]+:
  Sweeps equatorial ligands from pi-donors -> sigma-donors -> sigma-withdrawing -> pi-acceptors:
    L = F-, OH-, NH2-, H-, CH3-, C#CH-, CN-, CF3-, OCF3-, SF5-, C#C-CF3-
  Plus trans push-pull pi-acceptor synergy directly opposite F:
    [F-V(CF3)3(CO)]+, [F-V(CF3)3(CS)]+, [F-V(CF3)3(BF)]+, [F-W(CN)4(CO)]+
  Plus extreme dicationic pi-acidic ligand shells:
    [F-V(CF3)2(CO)2]2+, [F-V(CO)4]4+ (pure pi-acid limit)

Axis B — Direct F-Side Secondary-Sphere Ligands & Oriented Local Electric Field (LEF):
  1. Secondary-sphere Lewis acid vs. push-dipole on [F-V(CF3)3]+:
       [F-V(CF3)3]+...BF3 (Lewis acid on F — pulls F lone pairs away)
       [F-V(CF3)3]+...HF  (H-bond donor on F)
       [F-V(CF3)3]+...NH3 (lone-pair dipole pushing behind F at 2.8 A)
  2. Exact Oriented Local Electric Field (LEF) along z-axis (F -> V) on [F-V(CF3)3]+:
       E_z from -0.04 to +0.04 a.u. (via PySCF dipole perturbation h1e += E_z * r_z)
       Quantifies how a supramolecular push-pull electrostatic field boosts F pi-BO!
"""
import os
import sys
import time
import json
import numpy as np
from pyscf import gto, dft

sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from test_carbon_supramolecular import mayer_bo_and_host_q

OUT_DIR = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results"


def eval_complex(name, build_geom, charge, r_grid, ecp_dict=None, ef_z=0.0):
    """Optimize axial M-F distance on r_grid at DF-PBE0/def2-SVP, optional z-electric field ef_z (a.u.)."""
    t1 = time.time()
    es = []
    dm0 = None
    for r in r_grid:
        mol = gto.M(atom=build_geom(r), basis="def2-SVP", ecp=ecp_dict, charge=charge, spin=0, verbose=0)
        mf = dft.RKS(mol).density_fit()
        mf.xc = "pbe0"
        mf.max_cycle = 80
        mf.level_shift = 0.25
        if abs(ef_z) > 1e-12:
            # Add electric field perturbation V_ext = +ef_z * z (pushes electrons toward -z, i.e., F(+z) -> M(0)!)
            h_core_orig = mf.get_hcore
            def get_hcore_ef(mol_=mol, _orig=h_core_orig, _ez=ef_z):
                r_int = mol_.intor("int1e_r")[2]  # z matrix
                return _orig(mol_) + _ez * r_int
            mf.get_hcore = get_hcore_ef
        e = float(mf.kernel(dm0=dm0))
        dm0 = mf.make_rdm1()
        es.append(e)

    es = np.array(es)
    im = int(np.argmin(es))
    if 0 < im < len(r_grid) - 1:
        coeffs = np.polyfit(r_grid[im - 1:im + 2], es[im - 1:im + 2], 2)
        r_opt = float(-coeffs[1] / (2.0 * coeffs[0])) if coeffs[0] > 0 else float(r_grid[im])
        r_opt = float(np.clip(r_opt, r_grid[0], r_grid[-1]))
    else:
        r_opt = float(r_grid[im])

    mol = gto.M(atom=build_geom(r_opt), basis="def2-SVP", ecp=ecp_dict, charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe0"
    mf.max_cycle = 100
    mf.level_shift = 0.20
    if abs(ef_z) > 1e-12:
        h_core_orig = mf.get_hcore
        def get_hcore_ef2(mol_=mol, _orig=h_core_orig, _ez=ef_z):
            r_int = mol_.intor("int1e_r")[2]
            return _orig(mol_) + _ez * r_int
        mf.get_hcore = get_hcore_ef2
    mf.kernel(dm0=dm0)
    bo, bo_s, bo_p, q_m, q_f, q_host = mayer_bo_and_host_q(mf, 0, 1)
    res = {
        "r_e_A": round(r_opt, 3),
        "BO": round(bo, 3),
        "sigma": round(bo_s, 3),
        "pi": round(bo_p, 3),
        "q_M": round(q_m, 3),
        "q_F": round(q_f, 3),
        "converged": bool(mf.converged),
        "time_s": round(time.time() - t1, 1),
    }
    print(f"  {name:34s} | r_e={res['r_e_A']:.3f} A | BO={res['BO']:.3f} "
          f"(sig={res['sigma']:.3f}, pi={res['pi']:.3f}) | q(F)={res['q_F']:+.3f} | conv={res['converged']} ({res['time_s']}s)", flush=True)
    return res


# --- Geometry Builders for [F-V(L)3]+ (C3v, V at origin, axial F at (0,0,r)) ---

def geom_fv_l3(r, lig, trans_lig=None, f_cap=None):
    s = f"V 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        if lig == "NH2":
            s += f"N {1.80*c:.4f} {1.80*sn:.4f} -0.45; "
            s += f"H {2.35*c - 0.72*sn:.4f} {2.35*sn + 0.72*c:.4f} -0.75; "
            s += f"H {2.35*c + 0.72*sn:.4f} {2.35*sn - 0.72*c:.4f} -0.75; "
        elif lig == "OH":
            s += f"O {1.76*c:.4f} {1.76*sn:.4f} -0.45; H {2.58*c:.4f} {2.58*sn:.4f} -0.95; "
        elif lig == "F":
            s += f"F {1.70*c:.4f} {1.70*sn:.4f} -0.45; "
        elif lig == "H":
            s += f"H {1.62*c:.4f} {1.62*sn:.4f} -0.45; "
        elif lig == "CH3":
            s += f"C {1.98*c:.4f} {1.98*sn:.4f} -0.45; H {2.70*c:.4f} {2.70*sn:.4f} 0.25; "
            for d in (-0.88, 0.88):
                s += f"H {2.42*c - d*sn:.4f} {2.42*sn + d*c:.4f} -1.05; "
        elif lig == "CN":
            s += f"C {1.96*c:.4f} {1.96*sn:.4f} -0.45; N {3.10*c:.4f} {3.10*sn:.4f} -0.72; "
        elif lig == "CF3":
            s += f"C {2.02*c:.4f} {2.02*sn:.4f} -0.45; F {2.95*c:.4f} {2.95*sn:.4f} 0.45; "
            for d in (-1.08, 1.08):
                s += f"F {2.60*c - d*sn:.4f} {2.60*sn + d*c:.4f} -1.20; "
        elif lig == "OCF3":
            s += f"O {1.78*c:.4f} {1.78*sn:.4f} -0.40; C {3.05*c:.4f} {3.05*sn:.4f} -0.85; "
            s += f"F {3.95*c:.4f} {3.95*sn:.4f} -0.05; "
            for d in (-1.08, 1.08):
                s += f"F {3.55*c - d*sn:.4f} {3.55*sn + d*c:.4f} -1.60; "
        elif lig == "C2CF3":
            # Perfluoroalkynyl -C#C-CF3
            s += f"C {1.95*c:.4f} {1.95*sn:.4f} -0.40; C {3.13*c:.4f} {3.13*sn:.4f} -0.65; "
            s += f"C {4.54*c:.4f} {4.54*sn:.4f} -0.95; F {5.45*c:.4f} {5.45*sn:.4f} -0.15; "
            for d in (-1.08, 1.08):
                s += f"F {5.08*c - d*sn:.4f} {5.08*sn + d*c:.4f} -1.70; "
        elif lig == "SF5":
            # Ultra-electronegative -SF5 (Hammett sigma_I = 0.59, stronger than CF3!)
            s += f"S {2.28*c:.4f} {2.28*sn:.4f} -0.45; F {3.85*c:.4f} {3.85*sn:.4f} -0.78; "
            s += f"F {2.62*c:.4f} {2.62*sn:.4f} 1.08; F {1.94*c:.4f} {1.94*sn:.4f} -1.98; "
            for d in (-1.56, 1.56):
                s += f"F {2.28*c - d*sn:.4f} {2.28*sn + d*c:.4f} -0.45; "

    # Optional trans push-pull ligand directly opposite F along -z
    if trans_lig == "CO":
        s += "C 0 0 -2.05; O 0 0 -3.18; "
    elif trans_lig == "CS":
        s += "C 0 0 -1.98; S 0 0 -3.52; "
    elif trans_lig == "BF":
        s += "B 0 0 -2.05; F 0 0 -3.32; "
    elif trans_lig == "N2":
        s += "N 0 0 -2.08; N 0 0 -3.18; "
    elif trans_lig == "NH3":
        s += "N 0 0 -2.15; H 0.94 0 -2.55; H -0.47 0.81 -2.55; H -0.47 -0.81 -2.55; "

    # Optional direct F-side secondary-sphere cap behind F along +z
    if f_cap == "BF3":
        zb = r + 2.15
        s += f"B 0 0 {zb:.4f}; F 1.31 0 {zb+0.15:.4f}; F -0.655 1.134 {zb+0.15:.4f}; F -0.655 -1.134 {zb+0.15:.4f}; "
    elif f_cap == "HF":
        s += f"H 0 0 {r+1.85:.4f}; F 0 0 {r+2.78:.4f}; "
    elif f_cap == "NH3_push":
        # Lone pair of NH3 pointing directly at back of F (electrostatic repulsion/push toward V!)
        zn = r + 2.75
        s += f"N 0 0 {zn:.4f}; H 0.94 0 {zn+0.38:.4f}; H -0.47 0.81 {zn+0.38:.4f}; H -0.47 -0.81 {zn+0.38:.4f}; "
    return s


def main():
    t0 = time.time()
    results = {"axis_A_equatorial": {}, "axis_A_trans": {}, "axis_B_secondary_sphere": {}, "axis_B_electric_field": {}}

    print("=== AXIS A1: EQUATORIAL LIGAND SPECTRUM ON d0 [F-V(L)3]+ ===", flush=True)
    for lig, rg in [
        ("NH2",   [1.62, 1.68, 1.74]),
        ("OH",    [1.60, 1.66, 1.72]),
        ("F",     [1.56, 1.62, 1.68]),
        ("H",     [1.56, 1.62, 1.68]),
        ("CH3",   [1.58, 1.64, 1.70]),
        ("OCF3",  [1.54, 1.60, 1.66]),
        ("CN",    [1.54, 1.60, 1.66]),
        ("C2CF3", [1.54, 1.60, 1.66]),
        ("CF3",   [1.54, 1.60, 1.66]),
        ("SF5",   [1.52, 1.58, 1.64]),
    ]:
        results["axis_A_equatorial"][f"[F-V({lig})3]+"] = eval_complex(
            f"[F-V({lig})3]+", lambda r, _l=lig: geom_fv_l3(r, _l), 1, rg
        )

    print("\n=== AXIS A2: TRANS PUSH-PULL SYNERGY ON [L_trans-V(CF3)3-F]+ ===", flush=True)
    for tl in ["NH3", "N2", "CO", "CS", "BF"]:
        results["axis_A_trans"][f"[{tl}-V(CF3)3-F]+"] = eval_complex(
            f"trans-{tl} [V(CF3)3-F]+", lambda r, _tl=tl: geom_fv_l3(r, "CF3", trans_lig=_tl), 1, [1.54, 1.60, 1.66]
        )

    print("\n=== AXIS B1: DIRECT F-SIDE SECONDARY-SPHERE LIGANDS ON [F-V(CF3)3]+ ===", flush=True)
    for cap in ["BF3", "HF", "NH3_push"]:
        results["axis_B_secondary_sphere"][f"[F-V(CF3)3]+...{cap}"] = eval_complex(
            f"F-side {cap} on [F-V(CF3)3]+", lambda r, _c=cap: geom_fv_l3(r, "CF3", f_cap=_c), 1, [1.54, 1.62, 1.70]
        )

    print("\n=== AXIS B2: ORIENTED LOCAL ELECTRIC FIELD (LEF) ON [F-V(CF3)3]+ ===", flush=True)
    # Positive ef_z pushes electrons from F(+z) -> V(0) (mimics anionic cap behind F!)
    for ez in [-0.03, -0.015, 0.0, +0.015, +0.03, +0.05]:
        results["axis_B_electric_field"][f"Ez={ez:+.3f}"] = eval_complex(
            f"LEF Ez={ez:+.3f} a.u. [F-V(CF3)3]+", lambda r: geom_fv_l3(r, "CF3"), 1, [1.52, 1.59, 1.66], ef_z=ez
        )

    out_path = os.path.join(OUT_DIR, "f_pi_donor_ligand_study.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
