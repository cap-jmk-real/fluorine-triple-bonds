#!/usr/bin/env python3
"""Optional MnF semicore check: spin-pure CASSCF(20 e⁻, 13 orb) + SC-NEVPT2.

Protocol (fixed choices — also documented in docs/mnf_semicore_protocol.md):
  - Active space: Mn 3s, 3p, 3d, 4s and F 2p via AVAS (exactly 13 orbitals, 20 electrons).
  - X2C scalar-relativistic ROHF; basis cc-pVTZ-DK (default) or x2c-TZVPall (--basis).
  - States MnF_7 (7Σ⁺, spin=6) and MnF_5 (5Σ⁺, spin=4).
  - Grid: 10 points from r_e-0.20 Å to r_e+0.25 Å; r_e is required per state (--MnF_7-re, --MnF_5-re
    or --state + --re). Use v3 3d envelope minima (corrected_summary.csv) when available.
  - Forward/backward sweeps with orbital reuse; JSON checkpoint after each point.
  - Abort if MemAvailable < 14 GiB (override with env MNF_SEMICORE_MIN_MEM_GIB).

CI / local smoke (no heavy QC):
  python scripts/mnf_semicore_cas20_13.py --smoke

Full compute (compute host only):
  python scripts/mnf_semicore_cas20_13.py --run --out data/mnf_semicore/cas20_13_raw.json \\
    --basis cc-pVTZ-DK --MnF_7-re <r_e_7> --MnF_5-re <r_e_5>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import __version__ as PV
from pyscf import mrpt
from pyscf.mcscf import avas

NCAS = 13
NELEC = 20
SEMICORE_LABELS = ("Mn 3s", "Mn 3p", "Mn 3d", "Mn 4s", "F 2p")
GRID_DELTA_LO = 0.20
GRID_DELTA_HI = 0.25
N_GRID = 10

SPECS = (
    ("MnF_7", "Mn", 0, 6),
    ("MnF_5", "Mn", 0, 4),
)

BASIS_ARMS = {
    "cc-pVTZ-DK": ("AE_X2C_ccDK", "cc-pVTZ-DK"),
    "x2c-TZVPall": ("AE_X2C_x2cTZ", "x2c-tzvpall"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/mnf_semicore/cas20_13_raw.json"))
    p.add_argument("--basis", choices=tuple(BASIS_ARMS), default="cc-pVTZ-DK")
    p.add_argument("--run", action="store_true", help="Execute full PES sweeps (not for CI)")
    p.add_argument("--smoke", action="store_true", help="One cheap point to validate the workflow")
    p.add_argument("--MnF_7-re", type=float, default=None, dest="re_mnf7", metavar="ANG")
    p.add_argument("--MnF_5-re", type=float, default=None, dest="re_mnf5", metavar="ANG")
    p.add_argument("--state", choices=("MnF_7", "MnF_5"), default=None)
    p.add_argument("--re", type=float, default=None, help="r_e (Å) for --state")
    p.add_argument(
        "--fix-spin-shift",
        type=float,
        default=None,
        help="Override FIX_SPIN_SHIFT env (default 0.5)",
    )
    return p.parse_args(argv)


def min_mem_gib() -> float:
    return float(os.environ.get("MNF_SEMICORE_MIN_MEM_GIB", "14"))


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except OSError:
        return None
    return None


def check_memory() -> None:
    avail = mem_available_gib()
    need = min_mem_gib()
    if avail is not None and avail < need:
        raise SystemExit(
            f"MemAvailable {avail:.2f} GiB < required {need:.2f} GiB "
            f"(set MNF_SEMICORE_MIN_MEM_GIB to override floor)"
        )


def semicore_grid(re_ang: float) -> list[float]:
    if re_ang <= 0:
        raise ValueError(f"r_e must be positive, got {re_ang}")
    lo = re_ang - GRID_DELTA_LO
    hi = re_ang + GRID_DELTA_HI
    return [float(x) for x in np.linspace(lo, hi, N_GRID)]


def avas_semicore_mo(mf, *, thresholds: tuple[float, ...] | None = None):
    """AVAS selection; raises unless exactly NCAS orbitals and NELEC electrons."""
    thr_list = list(thresholds) if thresholds else [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
    last = None
    for minao in ("ano", "minao"):
        for thr in thr_list:
            try:
                n, ne, mo = avas.kernel(
                    mf,
                    list(SEMICORE_LABELS),
                    threshold=thr,
                    openshell_option=3,
                    minao=minao,
                    verbose=0,
                )
                ne_int = int(round(float(ne if not isinstance(ne, tuple) else sum(ne))))
                last = (n, ne_int, thr, minao)
                if n == NCAS and ne_int == NELEC:
                    return mo, minao, thr
            except Exception:
                continue
    msg = f"AVAS failed: need exactly {NCAS} orbitals and {NELEC} electrons"
    if last:
        msg += f"; last attempt gave n={last[0]}, ne={last[1]} (thr={last[2]}, minao={last[3]})"
    raise RuntimeError(msg)


def re_map_from_args(args: argparse.Namespace) -> dict[str, float]:
    m: dict[str, float] = {}
    if args.re_mnf7 is not None:
        m["MnF_7"] = args.re_mnf7
    if args.re_mnf5 is not None:
        m["MnF_5"] = args.re_mnf5
    if args.state and args.re is not None:
        m[args.state] = args.re
    return m


def point_semicore(
    element: str,
    r: float,
    charge: int,
    spin: int,
    basis: str,
    prev_state: dict | None,
    *,
    common,
    bond_fn,
) -> tuple[dict[str, Any], dict | None]:
    t0 = time.time()
    res: dict[str, Any] = {"R": r}
    mol = common.build(element, r, charge, spin, basis, None)
    dm0 = None if prev_state is None else prev_state.get("dm_hf")
    mf, hfres = common.rohf(mol, True, dm0, follow=True)
    res.update(hfres)
    try:
        if prev_state is not None and prev_state.get("mo") is not None:
            mo0 = prev_state["mo"]
            res["guess"] = "prev-cas-mo"
        else:
            mo0, avas_src, avas_thr = avas_semicore_mo(mf)
            res["guess"] = f"avas-{avas_src}"
            res["avas_minao"] = avas_src
            res["avas_threshold"] = avas_thr
        res["avas_norb"] = NCAS
        res["avas_nelec"] = NELEC

        ssv = 0.5 * spin * (0.5 * spin + 1.0)
        mc = __import__("pyscf.mcscf", fromlist=["mcscf"]).mcscf.CASSCF(mf, NCAS, NELEC)
        mc.max_cycle_macro = 80
        mc.conv_tol = 1e-6
        mc.fix_spin_(shift=common.fix_spin_shift(), ss=ssv)
        ci0 = prev_state.get("ci") if prev_state is not None else None
        mc.kernel(mo0, ci0=ci0)
        e_cas = float(mc.e_tot)
        res["e_cas"] = e_cas
        res["cas_conv"] = bool(mc.converged)

        e_nevpt = e_cas + float(mrpt.NEVPT(mc).kernel())
        res["e_nevpt2"] = e_nevpt

        a, C = common.analyse(mol, mc, 0, None if prev_state is None else (prev_state["mol"], prev_state["C"]), bond_fn)
        res.update(a)
        res["no_occ"] = [n["occ"] for n in a.get("no", [])]

        st = {
            "mol": mol,
            "mo": mc.mo_coeff,
            "ci": mc.ci,
            "C": C,
            "dm_hf": mf.make_rdm1(),
        }
    except Exception as exc:
        res["cas_error"] = repr(exc)[:400]
        st = prev_state
    res["seconds"] = round(time.time() - t0, 1)
    return res, st


def sweep_kind(
    kind: str,
    key: str,
    element: str,
    charge: int,
    spin: int,
    basis: str,
    grid: list[float],
    *,
    common,
    bond_fn,
    existing: list[dict] | None = None,
) -> list[dict]:
    done = {round(float(p["R"]), 8): p for p in (existing or [])}
    pts = list(existing or [])
    st = None
    rlist = grid if kind == "fwd" else grid[::-1]
    for r in rlist:
        rk = round(float(r), 8)
        prev = done.get(rk)
        if prev is not None and ("e_cas" in prev or "cas_error" in prev):
            continue
        check_memory()
        p, st = point_semicore(element, r, charge, spin, basis, st, common=common, bond_fn=bond_fn)
        if rk in done:
            pts = [x for x in pts if round(float(x["R"]), 8) != rk]
        pts.append(p)
        done[rk] = p
        print(
            f"{time.strftime('%F %T')} {key} {kind} R={r:.3f} S2={p.get('ss')} "
            f"Ecas={p.get('e_cas')} Enev={p.get('e_nevpt2')} conv={p.get('cas_conv')} {p['seconds']}s",
            flush=True,
        )
    return sorted(pts, key=lambda x: x["R"])


def run_sweeps(args: argparse.Namespace, re_by_state: dict[str, float]) -> int:
    if not args.run:
        raise SystemExit("Refusing heavy semicore CAS(20,13) without --run (see docs/mnf_semicore_protocol.md)")
    if not re_by_state:
        raise SystemExit("--run requires r_e for each state: --MnF_7-re, --MnF_5-re, or --state with --re")
    check_memory()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "pd_matched_v3"))
    import run_ab_initio as R  # noqa: WPS433
    from state_check_common import fix_spin_shift  # noqa: WPS433
    import state_check_common as common  # noqa: WPS433

    if args.fix_spin_shift is not None:
        os.environ["FIX_SPIN_SHIFT"] = str(args.fix_spin_shift)

    arm_name, basis_key = BASIS_ARMS[args.basis]
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.loads(out.read_text())
        if out.exists()
        else {
            "meta": {
                "pyscf_version": PV,
                "protocol": "mnf_semicore_cas20_13_v1",
                "basis": args.basis,
                "arm": arm_name,
                "ncas": NCAS,
                "nelec": NELEC,
                "avas_labels": list(SEMICORE_LABELS),
                "grid_spec": f"{N_GRID} pts from r_e-{GRID_DELTA_LO} to r_e+{GRID_DELTA_HI} Ang",
                "fix_spin_shift": fix_spin_shift(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "min_mem_gib": min_mem_gib(),
            },
            "scans": {},
        }
    )

    for key, element, charge, spin in SPECS:
        if key not in re_by_state:
            continue
        re_ang = re_by_state[key]
        grid = semicore_grid(re_ang)
        slot = data["scans"].setdefault(key, {}).setdefault(arm_name, {"re_ang": re_ang, "grid": grid})
        for kind in ("fwd", "bwd"):
            existing = slot.get(kind, {}).get("points")
            t0 = time.time()
            points = sweep_kind(
                kind,
                key,
                element,
                charge,
                spin,
                basis_key,
                grid,
                common=common,
                bond_fn=R.mayer_bond_order_and_charges,
                existing=existing,
            )
            slot[kind] = {"points": points, "seconds": round(time.time() - t0, 1)}
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
            tmp.replace(out)
    print("DONE", flush=True)
    return 0


def smoke_one_point() -> dict:
    """Tiny basis sanity: ROHF + AVAS sizing only (no CASSCF/NEVPT2)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pd_matched_v3"))
    import state_check_common as common  # noqa: WPS433

    mol = common.build("Mn", 1.85, 0, 6, "cc-pVDZ-DK", None)
    mf, _ = common.rohf(mol, True, None, follow=False)
    mo, src, thr = avas_semicore_mo(mf, thresholds=(0.05, 0.1, 0.2, 0.3))
    return {
        "smoke": True,
        "basis": "cc-pVDZ-DK",
        "avas_minao": src,
        "avas_threshold": thr,
        "avas_norb": NCAS,
        "avas_nelec": NELEC,
        "n_mo_cols": int(mo.shape[1]) if mo is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.smoke:
        payload = smoke_one_point()
        print(json.dumps(payload, indent=2))
        return 0

    re_by = re_map_from_args(args)
    if args.run:
        return run_sweeps(args, re_by)

    if args.re_mnf7 is not None or args.re_mnf5 is not None or args.re is not None:
        grids = {k: semicore_grid(v) for k, v in re_by.items()}
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "note": "Pass --run on a compute host to execute",
                    "basis": args.basis,
                    "re_ang": re_by,
                    "grids": grids,
                },
                indent=2,
            )
        )
        return 0

    print(
        "No action: use --smoke for CI, or --run with r_e values for production "
        "(docs/mnf_semicore_protocol.md)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())