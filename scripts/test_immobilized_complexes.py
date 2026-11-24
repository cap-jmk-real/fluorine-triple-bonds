#!/usr/bin/env python3
"""
Complex Immobilized High-Valent Fluorido Complexes (Paper 1):
Tests four realistic chemical immobilization architectures for preserving F -> M pi-bonding:
  1. Real Superacid Salts: [WF5]+[SbF6]-, [OsO3F]+[SbF6]-, [ReF6]+[SbF6]-
  2. Silica-Surface Organometallic Chemistry (SOMC) Immobilization:
       [(H3SiO)4WF]+, [(H3SiO)3WF2]+, [(H3SiO)3MoF2]+
  3. pi-Innocent vs. pi-Competing Equatorial Ligand Cages:
       [F-W(CO)4]+ (pi-acidic equatorial cage), [F-W(NH3)4]5+ (sigma-only cage),
       [F-W(PH3)4]5+ (phosphine cage), vs. [F-W(NH2)3]2+ (pi-competing amido cage)
  4. Noble-Gas Matrix Immobilization:
       [TiF]3+@He12 (icosahedral He matrix cage) vs. bare [TiF]3+
Computes equilibrium terminal M-F distance r_e, Mayer total BO, sigma/pi decomposition,
and Lowdin charge q(F) at PBE0/def2-SVP (relax) -> PBE0/def2-TZVP (single-point BO).
"""
import os
import time
import json
import numpy as np
from pyscf import gto, dft

OUT_DIR = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results"


def mayer_bo_sigma_pi(mf, a1=0, a2=1):
    """Mayer BO between atom a1 (metal at origin) and atom a2 (axial F on +z axis)."""
    mol = mf.mol
    S = mol.intor("int1e_ovlp")
    dm = mf.make_rdm1()
    if dm.ndim == 3:
        P = dm[0] + dm[1]
        PS = P @ S
        P_s = dm[0] - dm[1]
        PS_s = P_s @ S
    else:
        PS = dm @ S
        PS_s = None

    idx1 = [i for i, l in enumerate(mol.ao_labels(fmt=False)) if l[0] == a1]
    idx2 = [i for i, l in enumerate(mol.ao_labels(fmt=False)) if l[0] == a2]
    labels = mol.ao_labels(fmt=False)

    def sym_class(m_str):
        if m_str in ("", "z", "z^2", "z^3"):
            return "sigma"
        if m_str in ("x", "y", "xz", "yz", "xz^2", "yz^2"):
            return "pi"
        return "delta"

    bo_tot = bo_sig = bo_pi = 0.0
    for u in idx1:
        for v in idx2:
            val = float(PS[u, v] * PS[v, u])
            if PS_s is not None:
                val += float(PS_s[u, v] * PS_s[v, u])
            bo_tot += val
            c1, c2 = sym_class(labels[u][3]), sym_class(labels[v][3])
            if c1 == "sigma" and c2 == "sigma":
                bo_sig += val
            elif c1 == "pi" and c2 == "pi":
                bo_pi += val

    # Lowdin charge on atom a2 (axial F)
    eig, vec = np.linalg.eigh(S)
    S_half = vec @ np.diag(np.sqrt(np.maximum(eig, 1e-12))) @ vec.T
    P_tot = dm[0] + dm[1] if dm.ndim == 3 else dm
    P_low = S_half @ P_tot @ S_half
    pop_f = float(np.sum(np.diag(P_low)[idx2]))
    q_f = float(mol.atom_charge(a2) - pop_f)
    return bo_tot, bo_sig, bo_pi, q_f


def optimize_axial_mf(build_geom, charge, spin=0, r_grid=None, basis_opt="def2-SVP", basis_sp="def2-TZVP"):
    """1D parabolic optimization of axial M-F distance (atom 0 = M at origin, atom 1 = F at (0,0,r))."""
    if r_grid is None:
        r_grid = [1.65, 1.75, 1.85, 1.95]
    es = []
    dm0 = None
    for r in r_grid:
        mol = gto.M(atom=build_geom(r), basis=basis_opt, charge=charge, spin=spin, verbose=0)
        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
        mf.xc = "pbe0"
        mf.max_cycle = 80
        mf.level_shift = 0.2
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

    # Single-point at r_opt with larger basis_sp
    mol_sp = gto.M(atom=build_geom(r_opt), basis=basis_sp, charge=charge, spin=spin, verbose=0)
    mf_sp = dft.RKS(mol_sp) if spin == 0 else dft.UKS(mol_sp)
    mf_sp.xc = "pbe0"
    mf_sp.max_cycle = 100
    mf_sp.level_shift = 0.2
    e_sp = float(mf_sp.kernel())
    bo, bo_s, bo_p, q_f = mayer_bo_sigma_pi(mf_sp, 0, 1)
    return {
        "r_e_A": round(r_opt, 3),
        "E_ha": round(e_sp, 5),
        "converged": bool(mf_sp.converged),
        "BO": round(bo, 3),
        "sigma": round(bo_s, 3),
        "pi": round(bo_p, 3),
        "q_F": round(q_f, 3),
        "n_atoms": int(mol_sp.natm),
    }


# --- Geometry Builders (Atom 0 = Metal at (0,0,0), Atom 1 = Axial F at (0,0,r)) ---

def geom_wf5_sbf6(r):
    """Real superacid salt [WF5]+[SbF6]- (C4v W paired via trans mu-F to SbF6-)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        s += f"F {1.80*np.cos(rad):.4f} {1.80*np.sin(rad):.4f} -0.15; "
    # Bridging F at z = -2.15, Sb at z = -4.15, 4 eq F on Sb, 1 trans F on Sb
    s += "F 0 0 -2.15; Sb 0 0 -4.15; F 0 0 -6.03; "
    for ang in [45, 135, 225, 315]:
        rad = np.radians(ang)
        s += f"F {1.88*np.cos(rad):.4f} {1.88*np.sin(rad):.4f} -4.15; "
    return s


def geom_oso3f_sbf6(r):
    """Real isolated superacid salt [OsO3F]+[SbF6]- (Seppelt/Schrobilgen)."""
    s = f"Os 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        s += f"O {1.67*np.cos(rad):.4f} {1.67*np.sin(rad):.4f} -0.35; "
    s += "F 0 0 -2.30; Sb 0 0 -4.30; F 0 0 -6.18; "
    for ang in [45, 135, 225, 315]:
        rad = np.radians(ang)
        s += f"F {1.88*np.cos(rad):.4f} {1.88*np.sin(rad):.4f} -4.30; "
    return s


def geom_ref6_sbf6(r):
    """Real isolated superacid salt [ReF6]+[SbF6]-."""
    s = f"Re 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 72, 144, 216, 288]:
        rad = np.radians(ang)
        s += f"F {1.78*np.cos(rad):.4f} {1.78*np.sin(rad):.4f} -0.25; "
    s += "F 0 0 -2.25; Sb 0 0 -4.25; F 0 0 -6.13; "
    for ang in [45, 135, 225, 315]:
        rad = np.radians(ang)
        s += f"F {1.88*np.cos(rad):.4f} {1.88*np.sin(rad):.4f} -4.25; "
    return s


def geom_somc_w_sio4(r):
    """Silica-surface-immobilized (SOMC) [(H3SiO)4 W-F]+ (d0 W(VI), 4 siloxide anchors)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        # O at 1.88 A, Si at 3.45 A (tilted down -z)
        s += f"O {1.85*c:.4f} {1.85*sn:.4f} -0.35; "
        s += f"Si {3.25*c:.4f} {3.25*sn:.4f} -1.15; "
        # 3 H on Si
        for h_ang in [-40, 40]:
            hr = np.radians(ang + h_ang)
            s += f"H {3.25*c + 1.25*np.cos(hr):.4f} {3.25*sn + 1.25*np.sin(hr):.4f} -1.90; "
        s += f"H {3.25*c:.4f} {3.25*sn:.4f} -2.62; "
    return s


def geom_somc_m_sio3_f2(r, metal="W"):
    """Silica-surface-immobilized (SOMC) [(H3SiO)3 M-F2]+ (d0 M(VI), 3 siloxide anchors + 1 trans F)."""
    s = f"{metal} 0 0 0; F 0 0 {r:.4f}; F 0 0 -1.82; "
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"O {1.86*c:.4f} {1.86*sn:.4f} 0.0; "
        s += f"Si {3.30*c:.4f} {3.30*sn:.4f} -0.65; "
        for h_ang in [-40, 40]:
            hr = np.radians(ang + h_ang)
            s += f"H {3.30*c + 1.25*np.cos(hr):.4f} {3.30*sn + 1.25*np.sin(hr):.4f} -1.40; "
        s += f"H {3.30*c:.4f} {3.30*sn:.4f} -2.12; "
    return s


def geom_fw_co4(r):
    """pi-acidic equatorial cage [F-W(CO)4]+ (C4v, d4/d0 test; wait W(II) d4 vs W(VI): use d0 [F-W(NH3)4]5+ & [F-W(CO)4]+)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"C {2.05*c:.4f} {2.05*sn:.4f} -0.15; O {3.18*c:.4f} {3.18*sn:.4f} -0.25; "
    return s


def geom_fw_nh3_4(r):
    """sigma-only equatorial amine cage [F-W(NH3)4]5+ (d0 W(VI), zero ligand pi-competition!)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"N {2.15*c:.4f} {2.15*sn:.4f} -0.25; "
        s += f"H {2.80*c:.4f} {2.80*sn:.4f} 0.45; "
        for d in [-0.75, 0.75]:
            s += f"H {2.65*c - d*sn:.4f} {2.65*sn + d*c:.4f} -0.80; "
    return s


def geom_fw_nh2_3(r):
    """pi-competing equatorial amido cage [F-W(NH2)3]2+ (d0 W(VI), strong N->W pi competition)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"N {1.92*c:.4f} {1.92*sn:.4f} -0.35; "
        s += f"H {2.55*c - 0.72*sn:.4f} {2.55*sn + 0.72*c:.4f} -0.65; "
        s += f"H {2.55*c + 0.72*sn:.4f} {2.55*sn - 0.72*c:.4f} -0.65; "
    return s


def geom_tif3_he12(r):
    """Noble-gas matrix immobilization [TiF]3+@He12 (icosahedral He cage at R_cage = 2.6 A)."""
    s = f"Ti 0 0 0; F 0 0 {r:.4f}; "
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = []
    for x in (-1, 1):
        for y in (-1, 1):
            verts.append((0, x, y * phi))
            verts.append((x, y * phi, 0))
            verts.append((y * phi, 0, x))
    R_cage = 2.60
    for vx, vy, vz in verts:
        norm = np.sqrt(vx**2 + vy**2 + vz**2)
        cx, cy, cz = R_cage * vx / norm, R_cage * vy / norm, R_cage * vz / norm + 0.5 * r
        s += f"He {cx:.4f} {cy:.4f} {cz:.4f}; "
    return s


def main():
    t0 = time.time()
    print("=== COMPLEX IMMOBILIZED FLUORIDO COMPLEXES (PBE0/def2-SVP -> def2-TZVP) ===", flush=True)
    cases = [
        ("Superacid [WF5]+[SbF6]-", geom_wf5_sbf6, 0, [1.70, 1.78, 1.86]),
        ("Superacid [OsO3F]+[SbF6]-", geom_oso3f_sbf6, 0, [1.75, 1.85, 1.95]),
        ("Superacid [ReF6]+[SbF6]-", geom_ref6_sbf6, 0, [1.70, 1.78, 1.86]),
        ("SOMC [(H3SiO)4W-F]+", geom_somc_w_sio4, 1, [1.72, 1.80, 1.88]),
        ("SOMC [(H3SiO)3W-F2]+", lambda r: geom_somc_m_sio3_f2(r, "W"), 1, [1.72, 1.80, 1.88]),
        ("SOMC [(H3SiO)3Mo-F2]+", lambda r: geom_somc_m_sio3_f2(r, "Mo"), 1, [1.70, 1.78, 1.86]),
        ("sigma-Cage [F-W(NH3)4]5+", geom_fw_nh3_4, 5, [1.58, 1.66, 1.74]),
        ("pi-Cage [F-W(NH2)3]2+", geom_fw_nh2_3, 2, [1.74, 1.82, 1.90]),
        ("He-Matrix [TiF]3+@He12", geom_tif3_he12, 3, [1.52, 1.60, 1.68]),
    ]

    results = {}
    for name, fn, q, rg in cases:
        t1 = time.time()
        res = optimize_axial_mf(fn, charge=q, spin=0, r_grid=rg, basis_opt="def2-SVP", basis_sp="def2-TZVP")
        results[name] = res
        print(f"  {name:28s} ({res['n_atoms']:2d} at) | r_e={res['r_e_A']:.3f} A | "
              f"BO={res['BO']:.3f} (sig={res['sigma']:.3f}, pi={res['pi']:.3f}) | "
              f"q(F)={res['q_F']:+.3f} | conv={res['converged']} ({time.time()-t1:.1f}s)", flush=True)

    out_path = os.path.join(OUT_DIR, "immobilized_complexes_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
