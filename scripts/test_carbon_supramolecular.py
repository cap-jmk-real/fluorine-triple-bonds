#!/usr/bin/env python3
"""
Carbon, Supramolecular & Intramolecular Immobilization Architectures (Paper 1):
Tests realistic carbon-based & supramolecular scaffolds at DF-PBE0/def2-SVP (with ECP on 4d/5d):
  1. Endohedral Fullerene C20 Cage:       CF+@C20, [TiF]3+@C20, TiF@C20
  2. Supramolecular Cyclo[18]carbon Ring: CF+@C18, [TiF]3+@C18, [F-WH4]+@C18
  3. Organometallic Carbon Frameworks:    CpW(F)3 (neutral), [F-W(CH3)4]+ (monocation)
  4. Intramolecular peri-Naphthalene Clamp: [C10H6-W(H)2-F]+ (rigid 1,8-carbon chelate)
  5. Supramolecular Benzene Sandwich:     (C6H6)2...CF+ (pi-stacked aromatic pocket)
Computes equilibrium r_e(X-F), Mayer BO (total, sigma, pi), Lowdin q(F), and host charge transfer.
"""
import os
import time
import json
import numpy as np
from pyscf import gto, dft

OUT_DIR = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results"


def mayer_bo_and_host_q(mf, a1=0, a2=1):
    """Mayer BO (total, sigma, pi) between atoms a1 (at origin) and a2 (axial F on +z), plus host charge."""
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

    labels = mol.ao_labels(fmt=False)
    idx1 = [i for i, l in enumerate(labels) if l[0] == a1]
    idx2 = [i for i, l in enumerate(labels) if l[0] == a2]

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

    eig, vec = np.linalg.eigh(S)
    S_half = vec @ np.diag(np.sqrt(np.maximum(eig, 1e-12))) @ vec.T
    P_tot = dm[0] + dm[1] if dm.ndim == 3 else dm
    P_low = S_half @ P_tot @ S_half
    diag_low = np.diag(P_low)

    def atom_q(aid):
        idxs = [i for i, l in enumerate(labels) if l[0] == aid]
        return float(mol.atom_charge(aid) - np.sum(diag_low[idxs]))

    q_a1 = atom_q(a1)
    q_f = atom_q(a2)
    q_host = float(sum(atom_q(i) for i in range(2, mol.natm)))
    return bo_tot, bo_sig, bo_pi, q_a1, q_f, q_host


def run_complex(name, build_geom, charge, spin, r_grid, ecp_dict=None):
    """Optimize axial X-F distance on r_grid using DF-PBE0/def2-SVP and compute Mayer sigma/pi BO."""
    t1 = time.time()
    es = []
    dm0 = None
    for r in r_grid:
        mol = gto.M(atom=build_geom(r), basis="def2-SVP", ecp=ecp_dict, charge=charge, spin=spin, verbose=0)
        mf = (dft.RKS(mol) if spin == 0 else dft.UKS(mol)).density_fit()
        mf.xc = "pbe0"
        mf.max_cycle = 80
        mf.level_shift = 0.25
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

    mol = gto.M(atom=build_geom(r_opt), basis="def2-SVP", ecp=ecp_dict, charge=charge, spin=spin, verbose=0)
    mf = (dft.RKS(mol) if spin == 0 else dft.UKS(mol)).density_fit()
    mf.xc = "pbe0"
    mf.max_cycle = 100
    mf.level_shift = 0.20
    e_opt = float(mf.kernel(dm0=dm0))
    bo, bo_s, bo_p, q_x, q_f, q_host = mayer_bo_and_host_q(mf, 0, 1)
    res = {
        "n_atoms": int(mol.natm),
        "r_e_A": round(r_opt, 3),
        "BO": round(bo, 3),
        "sigma": round(bo_s, 3),
        "pi": round(bo_p, 3),
        "q_X": round(q_x, 3),
        "q_F": round(q_f, 3),
        "q_host": round(q_host, 3),
        "converged": bool(mf.converged),
        "time_s": round(time.time() - t1, 1),
    }
    print(f"  {name:30s} ({res['n_atoms']:2d} at) | r_e={res['r_e_A']:.3f} A | "
          f"BO={res['BO']:.3f} (sig={res['sigma']:.3f}, pi={res['pi']:.3f}) | "
          f"q(F)={res['q_F']:+.3f} q(host)={res['q_host']:+.2f} | conv={res['converged']} ({res['time_s']}s)", flush=True)
    return res


# --- Geometry Builders (Atom 0 = X at origin, Atom 1 = axial F at (0,0,r)) ---

def c20_vertices(radius=2.03):
    """Regular dodecahedron (20 vertices of C20 fullerene, C-C ~ 1.45 A at R = 2.03 A)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                verts.append((x, y, z))
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts.append((0, s1 / phi, s2 * phi))
            verts.append((s1 / phi, s2 * phi, 0))
            verts.append((s2 * phi, 0, s1 / phi))
    verts = np.array(verts, dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return verts


def geom_endo_c20(el, r):
    """Endohedral X-F centered inside C20 carbon fullerene (R = 2.03 A)."""
    s = f"{el} 0 0 {-0.5*r:.4f}; F 0 0 {0.5*r:.4f}; "
    for vx, vy, vz in c20_vertices(2.03):
        s += f"C {vx:.4f} {vy:.4f} {vz:.4f}; "
    return s


def geom_c18_ring(el, r, radius=3.65, extra_H=False):
    """Supramolecular cyclo[18]carbon ring (R = 3.65 A, C-C ~ 1.27 A) in xy-plane around axial X-F."""
    s = f"{el} 0 0 0; F 0 0 {r:.4f}; "
    if extra_H:
        for ang in [0, 90, 180, 270]:
            rad = np.radians(ang)
            s += f"H {1.72*np.cos(rad):.4f} {1.72*np.sin(rad):.4f} -0.25; "
    for k in range(18):
        ang = 2.0 * np.pi * k / 18.0
        s += f"C {radius*np.cos(ang):.4f} {radius*np.sin(ang):.4f} {0.5*r:.4f}; "
    return s


def geom_benzene_sandwich_cf(r):
    """Supramolecular (C6H6)2...CF+ pi-stacked aromatic pocket (two benzenes at z = +/-2.8 A)."""
    s = f"C 0 0 {-0.5*r:.4f}; F 0 0 {0.5*r:.4f}; "
    for z_plane in (-2.70, 2.70):
        for k in range(6):
            ang = 2.0 * np.pi * k / 6.0
            s += f"C {1.40*np.cos(ang):.4f} {1.40*np.sin(ang):.4f} {z_plane:.4f}; "
            s += f"H {2.48*np.cos(ang):.4f} {2.48*np.sin(ang):.4f} {z_plane:.4f}; "
    return s


def geom_fw_me4(r):
    """Organometallic carbon-coordinated [F-W(CH3)4]+ (d0 W(VI), C4v)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"C {2.12*c:.4f} {2.12*sn:.4f} -0.35; "
        s += f"H {2.85*c:.4f} {2.85*sn:.4f} 0.35; "
        for d in (-0.88, 0.88):
            s += f"H {2.55*c - d*sn:.4f} {2.55*sn + d*c:.4f} -0.95; "
    return s


def geom_cp_wf3(r):
    """Neutral organometallic half-sandwich Cp-W(F)3 (eta5-C5H5 carbon anchor, d0 W(VI))."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    # 2 equatorial/tilted F atoms on -y/+y side, Cp ring centered at (0, 0, -2.05)
    s += "F 1.72 0.0 -0.20; F -1.72 0.0 -0.20; "
    for k in range(5):
        ang = 2.0 * np.pi * k / 5.0
        s += f"C {1.21*np.cos(ang):.4f} {1.21*np.sin(ang):.4f} -2.05; "
        s += f"H {2.28*np.cos(ang):.4f} {2.28*np.sin(ang):.4f} -2.10; "
    return s


def geom_peri_naphth_wf(r):
    """Intramolecular 1,8-peri-naphthalene carbon clamp [C10H6-W(H)2-F]+ (rigid bidentate C,C chelate)."""
    # W at origin, axial F at (0,0,r), 2 H ligands at (0, +/-1.68, -0.25)
    # Naphthalene in xz-plane below W: C1, C8 at (+/-1.25, 0, -1.85) chelating W!
    s = f"W 0 0 0; F 0 0 {r:.4f}; H 0.0 1.68 -0.20; H 0.0 -1.68 -0.20; "
    # Rigid naphthalene skeleton (10 C + 6 H)
    c_coords = [
        (-1.24, 0, -1.85), (1.24, 0, -1.85),   # C1, C8 (peri-chelating carbons)
        (0.00, 0, -2.55), (0.00, 0, -3.97),    # C9, C10 (central bridge)
        (-2.45, 0, -2.55), (2.45, 0, -2.55),   # C2, C7
        (-2.45, 0, -3.97), (2.45, 0, -3.97),   # C3, C6
        (-1.24, 0, -4.67), (1.24, 0, -4.67),   # C4, C5
    ]
    h_coords = [
        (-3.39, 0, -2.01), (3.39, 0, -2.01),
        (-3.39, 0, -4.51), (3.39, 0, -4.51),
        (-1.24, 0, -5.75), (1.24, 0, -5.75),
    ]
    for x, y, z in c_coords:
        s += f"C {x:.4f} {y:.4f} {z:.4f}; "
    for x, y, z in h_coords:
        s += f"H {x:.4f} {y:.4f} {z:.4f}; "
    return s


def main():
    t0 = time.time()
    print("=== CARBON, SUPRAMOLECULAR & INTRAMOLECULAR IMMOBILIZATION (DF-PBE0/def2-SVP) ===", flush=True)
    results = {}

    cases = [
        ("Bare CF+ (reference)", lambda r: f"C 0 0 0; F 0 0 {r:.4f}", 1, 0, [1.10, 1.15, 1.20], None),
        ("Endohedral CF+@C20", lambda r: geom_endo_c20("C", r), 1, 0, [1.08, 1.13, 1.18], None),
        ("Supramolecular CF+@C18", lambda r: geom_c18_ring("C", r), 1, 0, [1.10, 1.15, 1.20], None),
        ("Sandwich (C6H6)2...CF+", geom_benzene_sandwich_cf, 1, 0, [1.10, 1.15, 1.20], None),
        ("Endohedral [TiF]3+@C20", lambda r: geom_endo_c20("Ti", r), 3, 0, [1.50, 1.60, 1.70], None),
        ("Supramolecular [TiF]3+@C18", lambda r: geom_c18_ring("Ti", r), 3, 0, [1.55, 1.65, 1.75], None),
        ("Supramolecular [F-WH4]+@C18", lambda r: geom_c18_ring("W", r, extra_H=True), 1, 0, [1.70, 1.78, 1.86], {"W": "def2-svp"}),
        ("Carbon-Cage [F-W(CH3)4]+", geom_fw_me4, 1, 0, [1.72, 1.80, 1.88], {"W": "def2-svp"}),
        ("Half-Sandwich Cp-W(F)3", geom_cp_wf3, 0, 0, [1.76, 1.84, 1.92], {"W": "def2-svp"}),
        ("Intramol. peri-Naphth-WH2F+", geom_peri_naphth_wf, 1, 0, [1.72, 1.80, 1.88], {"W": "def2-svp"}),
    ]

    for name, fn, q, sp, rg, ecp in cases:
        results[name] = run_complex(name, fn, q, sp, rg, ecp)

    out_path = os.path.join(OUT_DIR, "carbon_supramolecular_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
