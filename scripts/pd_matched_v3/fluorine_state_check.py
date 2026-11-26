#!/usr/bin/env python3
"""Electronic-state check for fluorine_pd_matched (fluorine PR #2 ledger item 11 follow-up; brief 25.11 22:36 CET).
Per (species, method) scan:
  legacy : the 5 legacy grid points, legacy protocol (ROHF canonical guess, unconstrained CASSCF, conv 1e-6), CASSCF only,
           now recording <S^2>, natural occupations, NO symmetry/atom character -> shows which state the legacy protocol lands on.
  fwd    : extended grid ascending, ROHF from previous density + internal stability following, spin-pure CASSCF
           (fix_spin_ to the target S) from the previous point's CAS orbitals (project_init_guess) and CI vector, SC-NEVPT2.
  bwd    : same, descending from 5.0 A.
Per point: <S^2>, multiplicity, CAS natural occupations with sigma/pi/delta and Pd/F character, active-space overlap with the
neighbouring point (min singular value of C_prev^T S_cross C_cur), ROHF stability status. Deterministic. Resumable per (species, arm, sweep).
v2 (25.11 ~23:20): v1 showed the frontier-orbital active space contains Pd 4s/4p semicore and only ~2 Pd 4d orbitals, with fwd/bwd
identical E_CAS but SC-NEVPT2 differing by 15-45 mEh (active composition ambiguous).
v3 (25.11 ~23:30): v2 showed that CASSCF orbital optimisation rotates the (occupation ~2.000) Pd 4d orbitals out of the AVAS space and pulls
Pd 4s/4p semicore in, even when started from AVAS: CAS(n,9) "4d+5s+F 2p" is not a stable CASSCF space for PdF/[PdF]2+. v3 therefore uses
AVAS(Pd 4d, 5s, F 2p) orbitals of the (stability-checked, density-followed) ROHF at each R with spin-pure CASCI (3 roots) + SC-NEVPT2 on root 0:
a fixed, well-defined active space; fwd and bwd differ only by the ROHF solution followed. v2 defines the active space as AVAS(Pd 4d, Pd 5s, F 2p; ANO
reference, threshold 0.2) = the 9 orbitals the Methods describe, re-seeds from AVAS whenever the followed composition drifts, and records it."""
import json, os, sys, time
from pathlib import Path
import numpy as np
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); MODE = sys.argv[3] if len(sys.argv) > 3 else "run"
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 else None
sys.path.insert(0, str(REPO / "scripts"))
from pyscf import gto, scf, mcscf, mrpt, __version__ as PV  # noqa: E402
from pyscf.mcscf import avas  # noqa: E402
import run_ab_initio as R  # noqa: E402

ARMS = [("ECP_ccPP", {"Pd": "cc-pvtz-pp", "F": "cc-pvtz"}, {"Pd": "cc-pvtz-pp"}, False),
        ("AE_X2C_x2cTZ", "x2c-tzvpall", None, True),
        ("AE_X2C_ccDK", "cc-pVTZ-DK", None, True),
        ("ECP_def2", "def2-TZVP", "def2-TZVP", False)]
SPECS = [("PdF_2", 0, 1, 9, 15, [1.75, 1.85, 1.92, 2.00, 2.15]),
         ("PdF2+_2", 2, 1, 9, 13, [1.65, 1.75, 1.85, 1.95, 2.15]),
         ("PdF2+_4", 2, 3, 9, 13, [1.65, 1.75, 1.85, 1.95, 2.15]),
         ("PdF3+_1", 3, 0, 9, 12, [1.60, 1.70, 1.80, 1.95, 2.15]),
         ("PdF3+_3", 3, 2, 9, 12, [1.60, 1.70, 1.80, 1.95, 2.15])]
GRID = [1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.10, 2.20, 2.35, 2.50, 2.80, 3.20, 5.00]

def ml_class(lbl):
    m = lbl[3].strip()
    if m in ("", "z", "z^2", "z^3", "0"): return "sigma"
    if m in ("x", "y", "xz", "yz", "xz^2", "yz^2", "+1", "-1"): return "pi"
    return "delta"

def build(r, q, spin, basis, ecp):
    kw = dict(atom=f"Pd 0 0 0; F 0 0 {r:.6f}", basis=basis, charge=q, spin=spin, symmetry=False, verbose=0)
    if ecp: kw["ecp"] = ecp
    return gto.M(**kw)

def rohf(mol, x2c, dm0=None, follow=True):
    mf = scf.ROHF(mol)
    if x2c: mf = mf.x2c()
    mf.max_cycle = 250; mf.level_shift = 0.25; mf.diis_space = 12
    mf.kernel(dm0)
    if not mf.converged:
        mf = scf.newton(mf); mf.kernel(mf.make_rdm1())
    nfol, stable = 0, None
    if follow:
        for _ in range(4):
            mo_i, _, stable, _ = mf.stability(return_status=True)
            if stable: break
            nfol += 1; mf.kernel(mf.make_rdm1(mo_i, mf.mo_occ))
    return mf, {"e_hf": float(mf.e_tot), "hf_conv": bool(mf.converged), "rohf_internal_stable": stable, "stab_follow_steps": nfol}

def analyse(mol, mc, prev):
    ncore, ncas = mc.ncore, mc.ncas
    ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, mc.nelecas)
    da, db = mc.fcisolver.make_rdm1s(mc.ci, ncas, mc.nelecas)
    occ, U = np.linalg.eigh(da + db); idx = np.argsort(-occ); occ, U = occ[idx], U[:, idx]
    C = mc.mo_coeff[:, ncore:ncore + ncas] @ U
    S = mol.intor("int1e_ovlp"); lab = mol.ao_labels(fmt=False); cls = [ml_class(l) for l in lab]
    nos = []
    for k in range(ncas):
        w = C[:, k] * (S @ C[:, k])
        sym = {c: float(sum(w[i] for i in range(len(w)) if cls[i] == c)) for c in ("sigma", "pi", "delta")}
        pd = float(sum(w[i] for i in range(len(w)) if lab[i][0] == 0)); pdd = float(sum(w[i] for i in range(len(w)) if lab[i][0] == 0 and lab[i][2].endswith("d")))
        fp_ = float(sum(w[i] for i in range(len(w)) if lab[i][0] == 1 and lab[i][2].endswith("p")))
        pdsp = float(sum(w[i] for i in range(len(w)) if lab[i][0] == 0 and (lab[i][2].endswith("s") or lab[i][2].endswith("p"))))
        nos.append({"occ": round(float(occ[k]), 4), "sym": max(sym, key=sym.get), "sym_w": round(max(sym.values()), 3), "Pd": round(pd, 3), "Pd_d": round(pdd, 3), "F_p": round(fp_, 3), "Pd_sp": round(pdsp, 3)})
    open_ = sorted(f"{n['sym']}({'Pd' if n['Pd'] > 0.5 else 'F'})" for n in nos if 0.5 < n["occ"] < 1.5)
    fp = {"n_open": len(open_), "open_shells": open_, "n_sigma_doubly": sum(1 for n in nos if n["occ"] >= 1.5 and n["sym"] == "sigma"),
          "n_pi_doubly": sum(1 for n in nos if n["occ"] >= 1.5 and n["sym"] == "pi"), "n_delta_doubly": sum(1 for n in nos if n["occ"] >= 1.5 and n["sym"] == "delta")}
    ov = None
    if prev is not None:
        pmol, pC = prev
        Sx = gto.intor_cross("int1e_ovlp", pmol, mol)
        ov = round(float(np.linalg.svd(pC.T @ Sx @ C, compute_uv=False).min()), 4)
    dma, dmb = mc.make_rdm1s()
    comp = {k: round(sum(n[k] for n in nos), 2) for k in ("Pd_d", "F_p", "Pd_sp")}
    return {"comp": comp, "comp_ok": bool(comp["Pd_d"] >= 4.3 and comp["F_p"] >= 2.3), "ss": round(float(ss), 4), "mult": round(float(mult), 3), "no": nos, "fingerprint": fp, "act_overlap_prev": ov,
            "bond_cas": R.mayer_bond_order_and_charges(mol, np.array([dma, dmb]))}, C

def avas_mo(mf, nel):
    for minao in ("ano", "minao"):
        try:
            n, ne, mo = avas.kernel(mf, ["Pd 4d", "Pd 5s", "F 2p"], threshold=0.2, openshell_option=3, minao=minao, verbose=0)
            if n == 9 and int(round(float(ne if not isinstance(ne, tuple) else sum(ne)))) == nel: return mo, minao
        except Exception:
            pass
    return None, None

def cas_run(mf, ncas, nel, spin, mo0, ci0):
    mc = mcscf.CASSCF(mf, ncas, nel); mc.max_cycle_macro = 100; mc.conv_tol = 1e-7; mc.fix_spin_(shift=0.5, ss=0.5 * spin * (0.5 * spin + 1))
    e = float(mc.kernel(mo0, ci0=ci0)[0]); return mc, e

def point(r, q, spin, basis, ecp, x2c, ncas, nel, prev_state, legacy=False, nevpt=True):
    t0 = time.time(); mol = build(r, q, spin, basis, ecp)
    dm0 = None if (legacy or prev_state is None) else prev_state["dm_hf"]
    mf, res = rohf(mol, x2c, dm0, follow=not legacy)
    res["R"] = r
    try:
        if legacy:
            mc = mcscf.CASSCF(mf, ncas, nel); mc.max_cycle_macro = 60; mc.conv_tol = 1e-6
            e_cas = float(mc.kernel(mf.mo_coeff)[0]); res["guess"] = "legacy-frontier"
        else:
            mo0, src = avas_mo(mf, nel)
            if mo0 is None: raise RuntimeError("AVAS did not give 9 orbitals / target electrons")
            ssv = 0.5 * spin * (0.5 * spin + 1)
            mr = mcscf.CASCI(mf, ncas, nel); mr.fcisolver.nroots = 3; mr.fix_spin_(shift=float(os.environ.get("FIX_SPIN_SHIFT", "0.5")), ss=ssv); mr.kernel(mo0)
            res["casci_roots_Eh"] = [float(x) for x in mr.e_tot]
            roots = []
            for i in range(len(mr.e_tot)):
                ssi, _ = mr.fcisolver.spin_square(mr.ci[i], ncas, mr.nelecas)
                da_, db_ = mr.fcisolver.make_rdm1s(mr.ci[i], ncas, mr.nelecas); oc = sorted(np.linalg.eigvalsh(da_ + db_))[::-1]
                try: en = float(mr.e_tot[i]) + float(mrpt.NEVPT(mr, root=i).kernel())
                except Exception as exc: en = None
                roots.append({"e_casci": float(mr.e_tot[i]), "e_nevpt2": en, "ss": round(float(ssi), 4), "n_open": int(sum(1 for x in oc if 0.5 < x < 1.5)), "no_occ": [round(float(x), 4) for x in oc]})
            res["roots"] = roots
            mc = mcscf.CASCI(mf, ncas, nel); mc.fix_spin_(shift=float(os.environ.get("FIX_SPIN_SHIFT", "0.5")), ss=ssv)
            e_cas = float(mc.kernel(mo0, ci0=mr.ci[0])[0]); res["guess"] = f"avas-{src}-casci"
            res["root0_consistent"] = bool(abs(e_cas - mr.e_tot[0]) < 1e-6)
        a, C = analyse(mol, mc, None if prev_state is None else (prev_state["mol"], prev_state["C"]))
        res.update({"e_cas": e_cas, "cas_conv": bool(mc.converged)}); res.update(a)
        if nevpt: res["e_nevpt2"] = e_cas + float(mrpt.NEVPT(mc).kernel())
        st = {"mol": mol, "mo": mc.mo_coeff, "ci": mc.ci, "C": C, "dm_hf": mf.make_rdm1()}
    except Exception as exc:
        res["cas_error"] = repr(exc)[:300]; st = prev_state
    res["seconds"] = round(time.time() - t0, 1)
    return res, st

def sweep(kind, key, q, spin, ncas, nel, lgrid, name, basis, ecp, x2c):
    grid = lgrid if kind == "legacy" else (GRID if kind == "fwd" else GRID[::-1])
    pts, st = [], None
    for r in grid:
        p, st2 = point(r, q, spin, basis, ecp, x2c, ncas, nel, None if kind == "legacy" else st, legacy=(kind == "legacy"), nevpt=(kind != "legacy"))
        pts.append(p); st = st2 if kind != "legacy" else None
        print(f"{time.strftime('%F %T')} {key} {name} {kind} R={r:.2f} g={p.get('guess')} comp={p.get('comp')} S2={p.get('ss')} Enev={p.get('e_nevpt2')} Ecas={p.get('e_cas')} open={p.get('fingerprint', {}).get('open_shells')} ov={p.get('act_overlap_prev')} stab={p.get('rohf_internal_stable')}/{p.get('stab_follow_steps')} {p['seconds']}s", flush=True)
    return sorted(pts, key=lambda p: p["R"])

OUT.mkdir(parents=True, exist_ok=True)
res_p = OUT / "state_check_raw.json"
res = json.loads(res_p.read_text()) if res_p.exists() else {"meta": {"pyscf_version": PV, "grid": GRID}, "scans": {}}
if MODE == "test":
    p, _ = point(1.85, 2, 1, "cc-pVTZ-DK", None, True, 9, 13, None); print(json.dumps({k: v for k, v in p.items() if k != "no"}, default=str)); sys.exit(0)
for key, q, spin, ncas, nel, lgrid in SPECS:
    for name, basis, ecp, x2c in ARMS:
        if ONLY and f"{key}:{name}" not in ONLY: continue
        for kind in ("legacy", "fwd", "bwd"):
            slot = res["scans"].setdefault(key, {}).setdefault(name, {})
            if kind in slot: continue
            t0 = time.time(); slot[kind] = {"points": sweep(kind, key, q, spin, ncas, nel, lgrid, name, basis, ecp, x2c), "seconds": round(time.time() - t0, 1)}
            tmp = res_p.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1, default=str) + "\n"); tmp.replace(res_p)
print("DONE", flush=True)
