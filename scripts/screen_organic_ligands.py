#!/usr/bin/env python3
"""
Comprehensive Organic Ligand Screen for Enhancing Fluorine pi-Donation (Paper 1):
Tested on d0 [F-V(L)3]+ (C3v, equatorial anionic organic ligands) and [L_org-V(CF3)3-F]+
(trans neutral organic ligands) at DF-PBE0/def2-SVP:

Part 1 — Equatorial Anionic Organic Ligands on [F-V(L)3]+:
  - Stepwise fluoroalkyl: -CH3, -CH2F, -CHF2, -CF3
  - Cyanocarbanion:       -CH(CN)2 (malononitrile anion — extreme organic EWG!)
  - Vinyl / Perfluorovinyl: -CH=CH2, -CF=CF2
  - Alkynyl:              -C#CH, -C#C-CN (cyanoethynyl)
  - Aryl / Perfluoroaryl: -C6H5 (Ph), -C6F5 (pentafluorophenyl)
  - Organic O-ligands:    -OCH3 (OMe), -OSO2CF3 (triflate, OTf-)

Part 2 — Trans Neutral Organic Ligands on [L_org-V(CF3)3-F]+:
  - NHC (imidazol-2-ylidene)
  - Isocyanides: CN-CH3 (MeNC), CN-CF3 (perfluoro-isocyanide, strong organic pi-acid!)
  - Pyridines:   C5H5N (py), C5F5N (pentafluoropyridine)
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from investigate_f_pi_donor_ligands import eval_complex

OUT_DIR = "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results"


def _rot_z(x, y, z, rad):
    c, s = np.cos(rad), np.sin(rad)
    return x * c - y * s, x * s + y * c, z


def geom_fv_org3(r, lig):
    """Build C3v [F-V(L)3]+ with V at origin and axial F at (0,0,r)."""
    s = f"V 0 0 0; F 0 0 {r:.4f}; "
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        pts = []  # list of (el, x, y, z) in local frame along +x, tilted slightly -z
        if lig in ("CH3", "CH2F", "CHF2", "CF3"):
            pts.append(("C", 2.00, 0.0, -0.45))
            # 3 substituents on C
            subs = {
                "CH3":  ["H", "H", "H"],
                "CH2F": ["F", "H", "H"],
                "CHF2": ["F", "F", "H"],
                "CF3":  ["F", "F", "F"],
            }[lig]
            vecs = [(0.90, 0.0, 0.85), (0.55, 0.95, -0.65), (0.55, -0.95, -0.65)]
            for el, (dx, dy, dz) in zip(subs, vecs):
                bl = 1.35 if el == "F" else 1.09
                nrm = np.sqrt(dx*dx + dy*dy + dz*dz)
                pts.append((el, 2.00 + bl*dx/nrm, bl*dy/nrm, -0.45 + bl*dz/nrm))
        elif lig == "CH(CN)2":
            # Malononitrile carbanion -CH(CN)2
            pts.append(("C", 2.04, 0.0, -0.45))
            pts.append(("H", 2.75, 0.0, 0.35))
            pts.append(("C", 2.68, 1.18, -0.95))
            pts.append(("N", 3.20, 2.12, -1.35))
            pts.append(("C", 2.68, -1.18, -0.95))
            pts.append(("N", 3.20, -2.12, -1.35))
        elif lig in ("CH=CH2", "CF=CF2"):
            cap = "F" if lig == "CF=CF2" else "H"
            bl = 1.34 if cap == "F" else 1.08
            pts.append(("C", 1.98, 0.0, -0.42))
            pts.append((cap, 2.55, 0.0, -0.42 + bl * 0.92))
            pts.append(("C", 3.15, 0.0, -1.05))
            pts.append((cap, 4.10, 0.0, -0.55))
            pts.append((cap, 3.15, 0.0, -1.05 - bl))
        elif lig == "C#CH":
            pts.append(("C", 1.95, 0.0, -0.42))
            pts.append(("C", 3.13, 0.0, -0.68))
            pts.append(("H", 4.18, 0.0, -0.91))
        elif lig == "C#C-CN":
            pts.append(("C", 1.95, 0.0, -0.42))
            pts.append(("C", 3.14, 0.0, -0.68))
            pts.append(("C", 4.49, 0.0, -0.97))
            pts.append(("N", 5.63, 0.0, -1.22))
        elif lig in ("Ph", "C6F5"):
            cap = "F" if lig == "C6F5" else "H"
            bl = 1.34 if cap == "F" else 1.08
            # Phenyl ring in radial-vertical plane (xz-plane rotated by ang), tilted slightly down -z
            # C1 at (2.00, 0, -0.45), ring center at (3.39, 0, -0.75)
            cx, cz = 3.39, -0.75
            # Ring in xy-tilted plane? Put ring perpendicular to xy (in local xz? no, local x + y plane avoids clash!)
            # Actually put ring plane at 45 deg or vertical (x, z) — at 120 deg separation, vertical rings have ZERO steric clash!
            # Local ring in (x, y) tilted by -12 deg in z:
            for k in range(6):
                th = np.radians(60 * k)  # k=0 is C4 (para), k=3 is C1 (ipso)
                rx = cx + 1.39 * np.cos(th)
                ry = 1.39 * np.sin(th)
                rz = cz - 0.22 * (rx - 2.00)
                pts.append(("C", rx, ry, rz))
                if k != 3:  # skip ipso C1
                    hx = cx + (1.39 + bl) * np.cos(th)
                    hy = (1.39 + bl) * np.sin(th)
                    hz = cz - 0.22 * (hx - 2.00)
                    pts.append((cap, hx, hy, hz))
        elif lig == "OMe":
            pts.append(("O", 1.77, 0.0, -0.42))
            pts.append(("C", 3.08, 0.0, -0.95))
            pts.append(("H", 3.78, 0.0, -0.12))
            pts.append(("H", 3.22, 0.89, -1.56))
            pts.append(("H", 3.22, -0.89, -1.56))
        elif lig == "OTf":
            # Triflate -O-SO2-CF3 (monodentate via O)
            pts.append(("O", 1.85, 0.0, -0.40))
            pts.append(("S", 3.22, 0.0, -0.95))
            pts.append(("O", 3.45, 1.25, -1.60))
            pts.append(("O", 3.45, -1.25, -1.60))
            pts.append(("C", 4.35, 0.0, 0.45))
            pts.append(("F", 5.62, 0.0, 0.05))
            pts.append(("F", 4.10, 1.08, 1.22))
            pts.append(("F", 4.10, -1.08, 1.22))

        for el, x, y, z in pts:
            xr, yr, zr = _rot_z(x, y, z, rad)
            s += f"{el} {xr:.4f} {yr:.4f} {zr:.4f}; "
    return s


def geom_fv_cf3_trans_org(r, tlig):
    """[L_org-V(CF3)3-F]+ with neutral organic ligand L_org trans to F along -z."""
    s = geom_fv_org3(r, "CF3")
    if tlig == "NHC":
        # Imidazol-2-ylidene (C2 at (0,0,-2.15), ring in xz-plane going down -z)
        s += "C 0.0 0.0 -2.15; "
        s += "N 1.08 0.0 -2.95; N -1.08 0.0 -2.95; "
        s += "C 0.68 0.0 -4.25; C -0.68 0.0 -4.25; "
        s += "H 2.05 0.0 -2.65; H -2.05 0.0 -2.65; "
        s += "H 1.36 0.0 -5.09; H -1.36 0.0 -5.09; "
    elif tlig == "CN-CH3":
        s += "C 0.0 0.0 -2.10; N 0.0 0.0 -3.26; C 0.0 0.0 -4.69; "
        s += "H 1.03 0.0 -5.05; H -0.51 0.89 -5.05; H -0.51 -0.89 -5.05; "
    elif tlig == "CN-CF3":
        s += "C 0.0 0.0 -2.08; N 0.0 0.0 -3.24; C 0.0 0.0 -4.65; "
        s += "F 1.25 0.0 -5.12; F -0.625 1.08 -5.12; F -0.625 -1.08 -5.12; "
    elif tlig in ("py", "C5F5N"):
        cap = "F" if tlig == "C5F5N" else "H"
        bl = 1.34 if cap == "F" else 1.08
        # N at (0,0,-2.15), ring in xz-plane down -z
        s += "N 0.0 0.0 -2.15; "
        ring_c = [
            (1.18, -2.82), (-1.18, -2.82),
            (1.20, -4.20), (-1.20, -4.20),
            (0.00, -4.90),
        ]
        ring_h = [
            (1.18 + bl*0.87, -2.82 + bl*0.50), (-1.18 - bl*0.87, -2.82 + bl*0.50),
            (1.20 + bl*0.87, -4.20 - bl*0.50), (-1.20 - bl*0.87, -4.20 - bl*0.50),
            (0.00, -4.90 - bl),
        ]
        for x, z in ring_c:
            s += f"C {x:.4f} 0.0 {z:.4f}; "
        for x, z in ring_h:
            s += f"{cap} {x:.4f} 0.0 {z:.4f}; "
    return s


def main():
    t0 = time.time()
    results = {"equatorial_organic": {}, "trans_organic": {}}

    print("=== PART 1: EQUATORIAL ORGANIC LIGANDS ON [F-V(L)3]+ (DF-PBE0/def2-SVP) ===", flush=True)
    for lig, rg in [
        ("OMe",     [1.62, 1.68, 1.74]),
        ("CH3",     [1.58, 1.64, 1.70]),
        ("CH2F",    [1.56, 1.62, 1.68]),
        ("CHF2",    [1.55, 1.61, 1.67]),
        ("CF3",     [1.55, 1.61, 1.67]),
        ("CH=CH2",  [1.58, 1.64, 1.70]),
        ("CF=CF2",  [1.55, 1.61, 1.67]),
        ("C#CH",    [1.56, 1.62, 1.68]),
        ("C#C-CN",  [1.54, 1.60, 1.66]),
        ("Ph",      [1.58, 1.64, 1.70]),
        ("C6F5",    [1.54, 1.60, 1.66]),
        ("CH(CN)2", [1.54, 1.60, 1.66]),
        ("OTf",     [1.52, 1.58, 1.64]),
    ]:
        results["equatorial_organic"][lig] = eval_complex(
            f"[F-V({lig})3]+", lambda r, _l=lig: geom_fv_org3(r, _l), 1, rg
        )

    print("\n=== PART 2: TRANS NEUTRAL ORGANIC LIGANDS ON [L_org-V(CF3)3-F]+ ===", flush=True)
    for tl in ["NHC", "py", "C5F5N", "CN-CH3", "CN-CF3"]:
        results["trans_organic"][tl] = eval_complex(
            f"trans-{tl} [V(CF3)3-F]+", lambda r, _tl=tl: geom_fv_cf3_trans_org(r, _tl), 1, [1.56, 1.62, 1.68]
        )

    out_path = os.path.join(OUT_DIR, "organic_ligand_screen.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
