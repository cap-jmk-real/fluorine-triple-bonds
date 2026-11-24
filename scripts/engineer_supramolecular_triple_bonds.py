#!/usr/bin/env python3
"""
Rational Engineering of Carbon & Supramolecular Hosts for Fluorine Multiple/Triple-Bond Stabilization:
Head-to-head comparison of unengineered (electron-rich) vs. rationally engineered (high-IP,
perfluorinated / pi-acidic / sigma-withdrawing) carbon & supramolecular architectures:
  1. Aromatic Sandwich:       (C6H6)2...CF+  vs.  (C6F6)2...CF+ (flipped quadrupole, high-IP)
  2. Carbon Anchors (W(VI)):  [F-W(CH3)4]+   vs.  [F-W(CF3)4]+  vs.  [F-W(CN)4]+  (sigma-withdrawing + pi-acidic)
  3. 3d0 V(V) Carbon Cage:    [F-V(CH3)3]+   vs.  [F-V(CF3)3]+  vs.  [F-V(CN)3]+
  4. Intramolecular Clamp:    peri-C10H6-WH2F+ vs. peri-C10F6-W(CN)2F+ (perfluorinated + pi-acidic clamp)
  5. Insulating sp3 Cage:     CF+@C20 (sp2)  vs.  CF+@C20H20 (saturated dodecahedrane) vs. CF+@C20F20 (perfluorododecahedrane)
  6. Supramolecular [TiF]3+:  [TiF]3+@C20    vs.  (C6F6)2...[TiF]3+ (high-IP aromatic host)
All at DF-PBE0/def2-SVP (with ECP on W), optimizing axial r_e(X-F) and computing Mayer sigma/pi BO & q(host).
"""
import os
import sys
import time
import json
import numpy as np
from pyscf import gto, dft

sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from test_carbon_supramolecular import run_complex, c20_vertices

OUT_DIR = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results"


def geom_c6f6_sandwich(el, r, z_plane=2.85):
    """Engineered perfluorobenzene sandwich (C6F6)2...X-F (positive quadrupole, high IP)."""
    s = f"{el} 0 0 {-0.5*r:.4f}; F 0 0 {0.5*r:.4f}; "
    for zp in (-z_plane, z_plane):
        for k in range(6):
            ang = 2.0 * np.pi * k / 6.0
            s += f"C {1.39*np.cos(ang):.4f} {1.39*np.sin(ang):.4f} {zp:.4f}; "
            s += f"F {2.73*np.cos(ang):.4f} {2.73*np.sin(ang):.4f} {zp:.4f}; "
    return s


def geom_fw_cf3_4(r):
    """Engineered sigma-withdrawing carbon anchor [F-W(CF3)4]+ (d0 W(VI), C4v)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"C {2.14*c:.4f} {2.14*sn:.4f} -0.35; "
        s += f"F {3.05*c:.4f} {3.05*sn:.4f} 0.55; "
        for d in (-1.08, 1.08):
            s += f"F {2.72*c - d*sn:.4f} {2.72*sn + d*c:.4f} -1.10; "
    return s


def geom_fw_cn_4(r):
    """Engineered pi-acidic + sigma-withdrawing carbon anchor [F-W(CN)4]+ (d0 W(VI), C4v)."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        s += f"C {2.08*c:.4f} {2.08*sn:.4f} -0.25; "
        s += f"N {3.23*c:.4f} {3.23*sn:.4f} -0.40; "
    return s


def geom_fv_x3(r, lig="CF3"):
    """3d0 V(V) carbon-anchored [F-V(CH3)3]+ vs [F-V(CF3)3]+ vs [F-V(CN)3]+."""
    s = f"V 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        c, sn = np.cos(rad), np.sin(rad)
        if lig == "CH3":
            s += f"C {1.98*c:.4f} {1.98*sn:.4f} -0.45; "
            s += f"H {2.70*c:.4f} {2.70*sn:.4f} 0.25; "
            for d in (-0.88, 0.88):
                s += f"H {2.42*c - d*sn:.4f} {2.42*sn + d*c:.4f} -1.05; "
        elif lig == "CF3":
            s += f"C {2.02*c:.4f} {2.02*sn:.4f} -0.45; "
            s += f"F {2.95*c:.4f} {2.95*sn:.4f} 0.45; "
            for d in (-1.08, 1.08):
                s += f"F {2.60*c - d*sn:.4f} {2.60*sn + d*c:.4f} -1.20; "
        elif lig == "CN":
            s += f"C {1.96*c:.4f} {1.96*sn:.4f} -0.45; "
            s += f"N {3.10*c:.4f} {3.10*sn:.4f} -0.72; "
    return s


def geom_peri_c10f6_w_cn2_f(r):
    """Engineered perfluorinated intramolecular peri-naphthalene clamp [C10F6-W(CN)2-F]+."""
    s = f"W 0 0 0; F 0 0 {r:.4f}; "
    s += "C 0.0 2.08 -0.20; N 0.0 3.23 -0.32; C 0.0 -2.08 -0.20; N 0.0 -3.23 -0.32; "
    c_coords = [
        (-1.24, 0, -1.90), (1.24, 0, -1.90),
        (0.00, 0, -2.60), (0.00, 0, -4.02),
        (-2.45, 0, -2.60), (2.45, 0, -2.60),
        (-2.45, 0, -4.02), (2.45, 0, -4.02),
        (-1.24, 0, -4.72), (1.24, 0, -4.72),
    ]
    f_coords = [
        (-3.62, 0, -1.92), (3.62, 0, -1.92),
        (-3.62, 0, -4.70), (3.62, 0, -4.70),
        (-1.24, 0, -6.06), (1.24, 0, -6.06),
    ]
    for x, y, z in c_coords:
        s += f"C {x:.4f} {y:.4f} {z:.4f}; "
    for x, y, z in f_coords:
        s += f"F {x:.4f} {y:.4f} {z:.4f}; "
    return s


def geom_endo_dodecahedrane(el, r, cap="H"):
    """Insulating saturated sp3 carbon cage: C20H20 (R=2.16 A) or C20F20 (R=2.18 A)."""
    R_c = 2.18 if cap == "F" else 2.16
    R_cap = R_c + (1.35 if cap == "F" else 1.09)
    s = f"{el} 0 0 {-0.5*r:.4f}; F 0 0 {0.5*r:.4f}; "
    verts_c = c20_vertices(R_c)
    verts_cap = c20_vertices(R_cap)
    for vx, vy, vz in verts_c:
        s += f"C {vx:.4f} {vy:.4f} {vz:.4f}; "
    for vx, vy, vz in verts_cap:
        s += f"{cap} {vx:.4f} {vy:.4f} {vz:.4f}; "
    return s


def main():
    t0 = time.time()
    print("=== ENGINEERED CARBON & SUPRAMOLECULAR HOSTS FOR TRIPLE-BOND STABILIZATION ===", flush=True)
    results = {}

    cases = [
        ("Engineered (C6F6)2...CF+", lambda r: geom_c6f6_sandwich("C", r, 2.85), 1, 0, [1.11, 1.15, 1.19], None),
        ("Engineered (C6F6)2...[TiF]3+", lambda r: geom_c6f6_sandwich("Ti", r, 2.95), 3, 0, [1.55, 1.63, 1.71], None),
        ("Engineered [F-W(CN)4]+", geom_fw_cn_4, 1, 0, [1.70, 1.76, 1.82], {"W": "def2-svp"}),
        ("Engineered [F-W(CF3)4]+", geom_fw_cf3_4, 1, 0, [1.68, 1.75, 1.82], {"W": "def2-svp"}),
        ("Baseline [F-V(CH3)3]+", lambda r: geom_fv_x3(r, "CH3"), 1, 0, [1.56, 1.62, 1.68], None),
        ("Engineered [F-V(CN)3]+", lambda r: geom_fv_x3(r, "CN"), 1, 0, [1.52, 1.58, 1.64], None),
        ("Engineered [F-V(CF3)3]+", lambda r: geom_fv_x3(r, "CF3"), 1, 0, [1.52, 1.58, 1.64], None),
        ("Engineered peri-C10F6-W(CN)2F+", geom_peri_c10f6_w_cn2_f, 1, 0, [1.70, 1.77, 1.84], {"W": "def2-svp"}),
        ("Insulating sp3 CF+@C20H20", lambda r: geom_endo_dodecahedrane("C", r, "H"), 1, 0, [1.08, 1.13, 1.18], None),
    ]

    for name, fn, q, sp, rg, ecp in cases:
        results[name] = run_complex(name, fn, q, sp, rg, ecp)

    out_path = os.path.join(OUT_DIR, "engineered_supramolecular_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
