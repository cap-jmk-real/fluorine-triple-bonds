"""Shared spin-pure AVAS-CASCI/NEVPT2 state-check primitives (Pd and 3d Mn/Ni)."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from pyscf import gto, mcscf, mrpt, scf
from pyscf.mcscf import avas

DEFAULT_GRID = [
    1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00,
    2.10, 2.20, 2.35, 2.50, 2.80, 3.20, 5.00,
]


def ml_class(lbl: str) -> str:
    m = lbl[3].strip()
    if m in ("", "z", "z^2", "z^3", "0"):
        return "sigma"
    if m in ("x", "y", "xz", "yz", "xz^2", "yz^2", "+1", "-1"):
        return "pi"
    return "delta"


def build(
    element: str,
    r: float,
    charge: int,
    spin: int,
    basis: str | dict,
    ecp: str | dict | None,
) -> gto.Mole:
    kw = dict(
        atom=f"{element} 0 0 0; F 0 0 {r:.6f}",
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=False,
        verbose=0,
    )
    if ecp:
        kw["ecp"] = ecp
    return gto.M(**kw)


def rohf(mol: gto.Mole, x2c: bool, dm0=None, follow: bool = True):
    mf = scf.ROHF(mol)
    if x2c:
        mf = mf.x2c()
    mf.max_cycle = 250
    mf.level_shift = 0.25
    mf.diis_space = 12
    mf.kernel(dm0)
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel(mf.make_rdm1())
    nfol, stable = 0, None
    if follow:
        for _ in range(4):
            mo_i, _, stable, _ = mf.stability(return_status=True)
            if stable:
                break
            nfol += 1
            mf.kernel(mf.make_rdm1(mo_i, mf.mo_occ))
    return mf, {
        "e_hf": float(mf.e_tot),
        "hf_conv": bool(mf.converged),
        "rohf_internal_stable": stable,
        "stab_follow_steps": nfol,
    }


def analyse(
    mol: gto.Mole,
    mc,
    metal_idx: int,
    prev,
    bond_fn,
) -> tuple[dict[str, Any], np.ndarray]:
    ncore, ncas = mc.ncore, mc.ncas
    ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, mc.nelecas)
    da, db = mc.fcisolver.make_rdm1s(mc.ci, ncas, mc.nelecas)
    occ, U = np.linalg.eigh(da + db)
    idx = np.argsort(-occ)
    occ, U = occ[idx], U[:, idx]
    C = mc.mo_coeff[:, ncore : ncore + ncas] @ U
    S = mol.intor("int1e_ovlp")
    lab = mol.ao_labels(fmt=False)
    cls = [ml_class(l) for l in lab]
    nos = []
    for k in range(ncas):
        w = C[:, k] * (S @ C[:, k])
        sym = {c: float(sum(w[i] for i in range(len(w)) if cls[i] == c)) for c in ("sigma", "pi", "delta")}
        m_idx = metal_idx
        pd = float(sum(w[i] for i in range(len(w)) if lab[i][0] == m_idx))
        pdd = float(
            sum(w[i] for i in range(len(w)) if lab[i][0] == m_idx and lab[i][2].endswith("d"))
        )
        fp_ = float(sum(w[i] for i in range(len(w)) if lab[i][0] == 1 and lab[i][2].endswith("p")))
        pdsp = float(
            sum(
                w[i]
                for i in range(len(w))
                if lab[i][0] == m_idx and (lab[i][2].endswith("s") or lab[i][2].endswith("p"))
            )
        )
        nos.append(
            {
                "occ": round(float(occ[k]), 4),
                "sym": max(sym, key=sym.get),
                "sym_w": round(max(sym.values()), 3),
                "Pd": round(pd, 3),
                "Pd_d": round(pdd, 3),
                "F_p": round(fp_, 3),
                "Pd_sp": round(pdsp, 3),
            }
        )
    open_ = sorted(
        f"{n['sym']}({'M' if n['Pd'] > 0.5 else 'F'})"
        for n in nos
        if 0.5 < n["occ"] < 1.5
    )
    fp = {
        "n_open": len(open_),
        "open_shells": open_,
        "n_sigma_doubly": sum(1 for n in nos if n["occ"] >= 1.5 and n["sym"] == "sigma"),
        "n_pi_doubly": sum(1 for n in nos if n["occ"] >= 1.5 and n["sym"] == "pi"),
        "n_delta_doubly": sum(1 for n in nos if n["occ"] >= 1.5 and n["sym"] == "delta"),
    }
    ov = None
    if prev is not None:
        pmol, pC = prev
        Sx = gto.intor_cross("int1e_ovlp", pmol, mol)
        ov = round(float(np.linalg.svd(pC.T @ Sx @ C, compute_uv=False).min()), 4)
    dma, dmb = mc.make_rdm1s()
    comp = {k: round(sum(n[k] for n in nos), 2) for k in ("Pd_d", "F_p", "Pd_sp")}
    return {
        "comp": comp,
        "comp_ok": bool(comp["Pd_d"] >= 4.3 and comp["F_p"] >= 2.3),
        "ss": round(float(ss), 4),
        "mult": round(float(mult), 3),
        "no": nos,
        "fingerprint": fp,
        "act_overlap_prev": ov,
        "bond_cas": bond_fn(mol, np.array([dma, dmb])),
    }, C


def avas_mo(mf, nel: int, metal: str, ncas: int = 9):
    if ncas == 9 and metal in ("Pd",):
        labels = [f"{metal} 4d", f"{metal} 5s", "F 2p"]
    else:
        labels = [f"{metal} 3d", f"{metal} 4s", "F 2p"]
    for minao in ("ano", "minao"):
        try:
            n, ne, mo = avas.kernel(
                mf, labels, threshold=0.2, openshell_option=3, minao=minao, verbose=0
            )
            ne_int = int(round(float(ne if not isinstance(ne, tuple) else sum(ne))))
            if n == ncas and ne_int == nel:
                return mo, minao
        except Exception:
            pass
    return None, None


def fix_spin_shift() -> float:
    return float(os.environ.get("FIX_SPIN_SHIFT", "0.5"))


def point(
    element: str,
    r: float,
    charge: int,
    spin: int,
    basis: str | dict,
    ecp: str | dict | None,
    x2c: bool,
    ncas: int,
    nel: int,
    prev_state,
    legacy: bool = False,
    nevpt: bool = True,
    bond_fn=None,
    metal_idx: int = 0,
) -> tuple[dict[str, Any], dict | None]:
    import time

    import run_ab_initio as R  # noqa: WPS433 — repo driver for Mayer partitioning

    if bond_fn is None:
        bond_fn = R.mayer_bond_order_and_charges

    t0 = time.time()
    mol = build(element, r, charge, spin, basis, ecp)
    dm0 = None if (legacy or prev_state is None) else prev_state["dm_hf"]
    mf, res = rohf(mol, x2c, dm0, follow=not legacy)
    res["R"] = r
    try:
        if legacy:
            mc = mcscf.CASSCF(mf, ncas, nel)
            mc.max_cycle_macro = 60
            mc.conv_tol = 1e-6
            e_cas = float(mc.kernel(mf.mo_coeff)[0])
            res["guess"] = "legacy-frontier"
        else:
            mo0, src = avas_mo(mf, nel, element, ncas)
            if mo0 is None:
                raise RuntimeError(f"AVAS did not give {ncas} orbitals / {nel} electrons")
            ssv = 0.5 * spin * (0.5 * spin + 1)
            mr = mcscf.CASCI(mf, ncas, nel)
            mr.fcisolver.nroots = 3
            mr.fix_spin_(shift=fix_spin_shift(), ss=ssv)
            mr.kernel(mo0)
            res["casci_roots_Eh"] = [float(x) for x in mr.e_tot]
            roots = []
            for i in range(len(mr.e_tot)):
                ssi, _ = mr.fcisolver.spin_square(mr.ci[i], ncas, mr.nelecas)
                da_, db_ = mr.fcisolver.make_rdm1s(mr.ci[i], ncas, mr.nelecas)
                oc = sorted(np.linalg.eigvalsh(da_ + db_))[::-1]
                try:
                    en = float(mr.e_tot[i]) + float(mrpt.NEVPT(mr, root=i).kernel())
                except Exception:
                    en = None
                roots.append(
                    {
                        "e_casci": float(mr.e_tot[i]),
                        "e_nevpt2": en,
                        "ss": round(float(ssi), 4),
                        "n_open": int(sum(1 for x in oc if 0.5 < x < 1.5)),
                        "no_occ": [round(float(x), 4) for x in oc],
                    }
                )
            res["roots"] = roots
            mc = mcscf.CASCI(mf, ncas, nel)
            mc.fix_spin_(shift=fix_spin_shift(), ss=ssv)
            e_cas = float(mc.kernel(mo0, ci0=mr.ci[0])[0])
            res["guess"] = f"avas-{src}-casci"
            res["root0_consistent"] = bool(abs(e_cas - mr.e_tot[0]) < 1e-6)
        a, C = analyse(mol, mc, metal_idx, None if prev_state is None else (prev_state["mol"], prev_state["C"]), bond_fn)
        res.update({"e_cas": e_cas, "cas_conv": bool(mc.converged)})
        res.update(a)
        if nevpt:
            res["e_nevpt2"] = e_cas + float(mrpt.NEVPT(mc).kernel())
        st = {"mol": mol, "mo": mc.mo_coeff, "ci": mc.ci, "C": C, "dm_hf": mf.make_rdm1()}
    except Exception as exc:
        res["cas_error"] = repr(exc)[:300]
        st = prev_state
    res["seconds"] = round(time.time() - t0, 1)
    return res, st


def sweep(
    kind: str,
    element: str,
    charge: int,
    spin: int,
    ncas: int,
    nel: int,
    lgrid: list[float],
    name: str,
    basis: str | dict,
    ecp: str | dict | None,
    x2c: bool,
    grid: list[float] | None = None,
    log_prefix: str = "",
) -> list[dict]:
    import time

    grid = grid or DEFAULT_GRID
    pts, st = [], None
    rgrid = lgrid if kind == "legacy" else (grid if kind == "fwd" else grid[::-1])
    for r in rgrid:
        p, st2 = point(
            element,
            r,
            charge,
            spin,
            basis,
            ecp,
            x2c,
            ncas,
            nel,
            None if kind == "legacy" else st,
            legacy=(kind == "legacy"),
            nevpt=(kind != "legacy"),
        )
        pts.append(p)
        st = st2 if kind != "legacy" else None
        print(
            f"{time.strftime('%F %T')} {log_prefix}{name} {kind} R={r:.2f} "
            f"S2={p.get('ss')} Enev={p.get('e_nevpt2')} {p['seconds']}s",
            flush=True,
        )
    return sorted(pts, key=lambda p: p["R"])