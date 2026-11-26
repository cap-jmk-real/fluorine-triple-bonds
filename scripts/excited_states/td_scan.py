#!/usr/bin/env python3
"""Vertical TD-DFT/TDHF scan for Pd/Mn/Ni monofluorides (matched v3 geometry conventions).

CI / smoke:
  python scripts/excited_states/td_scan.py --smoke

Full compute (compute host only):
  OMP_NUM_THREADS=16 PYSCF_MAX_MEMORY=8000 \\
    python scripts/excited_states/td_scan.py --run \\
      --out data/td_scan/td_raw.json \\
      [--x2c] [--species PdF_2] [--grid 1.55:2.60:0.05] [--nroots 20]

Resumes after each (species, method, bond length) checkpoint in --out.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from td_common import (
    ARMS,
    SMOKE_BASIS,
    SPIN_FLIP_JOBS,
    TD_METHODS,
    TD_SPECS,
    bad_roots,
    check_memory,
    estimate_excited_s2,
    max_memory_mb,
    min_mem_gib,
    mr_flags_for_point,
    nto_character,
    parse_grid,
    stats_lowest,
    target_s2,
    transition_character,
)

REPO = Path(__file__).resolve().parents[2]
PD_MATCHED = Path(__file__).resolve().parents[1] / "pd_matched_v3"

_PV: str | None = None
_scf_mod = None
_tdscf_mod = None


def _pyscf():
    global _PV, _scf_mod, _tdscf_mod
    if _scf_mod is None:
        from pyscf import __version__ as pv
        from pyscf import scf as scf_mod
        from pyscf import tdscf as tdscf_mod

        _PV = pv
        _scf_mod = scf_mod
        _tdscf_mod = tdscf_mod
    return _PV, _scf_mod, _tdscf_mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/td_scan/td_raw.json"))
    p.add_argument("--run", action="store_true", help="Execute heavy scans (not for CI)")
    p.add_argument("--smoke", action="store_true", help="Tiny basis, 2 points, 3 roots, one species")
    p.add_argument("--x2c", action="store_true", help="Include X2C all-electron basis arms")
    p.add_argument("--species", default=None, help="Comma filter of TD_SPECS keys")
    p.add_argument("--methods", default=None, help="Comma filter of method ids")
    p.add_argument("--grid", default=None, help="start:stop:step or comma list (Å)")
    p.add_argument("--nroots", type=int, default=20)
    p.add_argument("--skip-spin-flip", action="store_true")
    p.add_argument(
        "--mem-wait-seconds",
        type=float,
        default=300,
        help="Seconds to sleep between MemAvailable rechecks when below TD_MIN_MEM_GIB",
    )
    p.add_argument(
        "--mem-wait-max-hours",
        type=float,
        default=24,
        help="Stop the scan if memory stays low longer than this (hours)",
    )
    return p.parse_args(argv)


def _import_common():
    sys.path.insert(0, str(PD_MATCHED))
    import state_check_common as common  # noqa: WPS433

    return common


def build_mf(mol, driver: str, xc: str | None, x2c: bool, *, guess: str, dm0=None):
    _, scf, _ = _pyscf()
    if driver == "UKS":
        mf = scf.UKS(mol)
        mf.xc = xc or "pbe0"
    else:
        mf = scf.UHF(mol)
    if x2c:
        mf = mf.x2c()
    mf.verbose = 0
    mf.max_cycle = 200
    mf.level_shift = 0.2
    mf.diis_space = 12
    mf.max_memory = max_memory_mb()
    mf.init_guess = guess
    if dm0 is not None:
        mf.kernel(dm0=dm0)
    else:
        mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)  # noqa: same scf module from _pyscf()
        mf.max_memory = max_memory_mb()
        mf.kernel(mf.make_rdm1())
    return mf


def scf_stability(mf) -> dict[str, Any]:
    stable_int = True
    stable_ext = True
    n_int = 0
    for _ in range(4):
        _, _, stable_int, _ = mf.stability(return_status=True)
        if stable_int:
            break
        n_int += 1
        mo_i, _, _, _ = mf.stability(return_status=True)
        mf.kernel(mf.make_rdm1(mo_i, mf.mo_occ))
    _, _, stable_ext, _ = mf.stability(external=True, return_status=True)
    s2, mult = mf.spin_square()
    return {
        "e_tot_ha": float(mf.e_tot),
        "scf_converged": bool(mf.converged),
        "s2_ref": float(s2),
        "mult_ref": float(mult),
        "internal_stable": bool(stable_int),
        "external_stable": bool(stable_ext),
        "internal_follow_steps": n_int,
        "reference_instability": not (stable_int and stable_ext),
    }


def scf_reference_with_guesses(
    common,
    element: str,
    r: float,
    charge: int,
    spin: int,
    basis,
    ecp,
    x2c: bool,
    driver: str,
    xc: str | None,
    prev: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Run three SCF seeds; keep the lowest-energy internally stable solution."""
    dm_prev = None if prev is None else prev.get("dm")
    mfs_by_guess: dict[str, Any] = {}
    guesses: list[dict[str, Any]] = []
    candidates: list[tuple[Any, dict[str, Any]]] = []

    for seed, gkind in enumerate(("minao", "atom", "prev")):
        if gkind == "prev" and dm_prev is None:
            continue
        mol = common.build(element, r, charge, spin, basis, ecp)
        dm0 = dm_prev if gkind == "prev" else None
        guess = "minao" if gkind == "prev" else gkind
        mf = build_mf(mol, driver, xc, x2c, guess=guess, dm0=dm0)
        meta = scf_stability(mf)
        meta["guess"] = gkind
        meta["seed"] = seed
        guesses.append(meta)
        mfs_by_guess[gkind] = mf
        if meta["internal_stable"]:
            candidates.append((mf, meta))

    if not candidates:
        mol = common.build(element, r, charge, spin, basis, ecp)
        mf = build_mf(mol, driver, xc, x2c, guess="minao")
        meta = scf_stability(mf)
        meta["guess"] = "minao_forced"
        meta["seed"] = -1
        guesses.append(meta)
        candidates.append((mf, meta))
        mfs_by_guess["minao_forced"] = mf

    mf, best = min(candidates, key=lambda t: t[1]["e_tot_ha"])

    lowest_exc: list[float] = []
    for gkind, gmf in mfs_by_guess.items():
        gmeta = next(x for x in guesses if x.get("guess") == gkind)
        if not gmeta.get("internal_stable"):
            continue
        try:
            _, _, tdscf = _pyscf()
            td = tdscf.TDA(gmf)
            td.nstates = 1
            td.verbose = 0
            td.kernel()
            from td_common import HARTREE2EV

            lowest_exc.append(float(td.e[0]) * HARTREE2EV)
        except Exception:
            continue

    winning = dict(best)
    winning["guesses"] = guesses
    winning["guess_spread_lowest_exc_ev"] = stats_lowest(lowest_exc)
    winning["winning_guess"] = winning.get("guess")
    return mf, winning


def run_td_on_mf(mf, driver: str, xc: str | None, algorithm: str, nroots: int) -> dict[str, Any]:
    _, _, tdscf = _pyscf()
    if driver == "UKS" and xc and getattr(mf, "xc", None) != xc:
        mf.xc = xc
    td_cls = tdscf.TDA if algorithm == "TDA" else tdscf.TDDFT
    td = td_cls(mf)
    td.nstates = nroots
    td.conv_tol = 1e-3
    td.verbose = 0
    t0 = time.time()
    converged = True
    try:
        td.kernel()
    except Exception as exc:
        return {"error": repr(exc)[:400], "seconds": round(time.time() - t0, 2)}
    if hasattr(td, "converged"):
        converged = bool(np.all(td.converged)) if isinstance(td.converged, (list, tuple, np.ndarray)) else bool(td.converged)

    from td_common import HARTREE2EV

    energies_ha = np.asarray(td.e, dtype=float)
    energies_ev = energies_ha * HARTREE2EV
    osc = np.asarray(td.oscillator_strength(), dtype=float).tolist()
    roots: list[dict[str, Any]] = []
    mol = mf.mol
    for i in range(len(energies_ev)):
        x, _y = td.xy[i]
        t1a, t1b = x
        s2 = estimate_excited_s2(mf, t1a, t1b)
        lam = transition_character(mol, mf, t1a, t1b)
        lam_sym = lam[0]["symmetry"] if lam else "unknown"
        from td_common import lambda_label_from_ao

        occ0 = int(np.argmax(np.abs(t1a))) if t1a.size else 0
        vir0 = mf.mo_coeff[0][:, mf.mo_occ[0] == 0]
        vir_vec = vir0[:, min(occ0, vir0.shape[1] - 1)] if vir0.size else mf.mo_coeff[0][:, 0]
        roots.append(
            {
                "index": i,
                "e_ev": float(np.real(energies_ev[i])),
                "f_osc": float(osc[i]) if i < len(osc) else None,
                "s2_exc": s2,
                "lambda": lambda_label_from_ao(mol, vir_vec),
                "dominant_excitations": transition_character(mol, mf, t1a, t1b),
                "nto": nto_character(mol, mf, td, i),
                "cinf_sym": lam_sym,
            }
        )
    br = bad_roots(energies_ev)
    return {
        "driver": driver,
        "xc": xc,
        "algorithm": algorithm,
        "nroots_requested": nroots,
        "nroots_returned": len(roots),
        "converged": converged,
        "roots": roots,
        "bad_roots": br,
        "seconds": round(time.time() - t0, 2),
    }


def solve_point(
    common,
    species: str,
    element: str,
    charge: int,
    spin: int,
    r: float,
    basis,
    ecp,
    x2c: bool,
    method_id: str,
    driver: str,
    xc: str | None,
    algorithm: str,
    nroots: int,
    prev_state,
) -> dict[str, Any]:
    t0 = time.time()
    ref_driver = "UKS" if driver == "UKS" else "UHF"
    ref_xc = xc if ref_driver == "UKS" else None
    mf, ref_meta = scf_reference_with_guesses(
        common, element, r, charge, spin, basis, ecp, x2c, ref_driver, ref_xc, prev_state
    )

    td_main = run_td_on_mf(mf, driver, xc if driver == "UKS" else None, algorithm, nroots)

    rpa_tda_max = None
    pair_alg = "TDA" if algorithm == "RPA" else "RPA"
    td_pair = run_td_on_mf(mf, driver, xc if driver == "UKS" else None, pair_alg, nroots)
    if "roots" in td_main and "roots" in td_pair:
        n = min(len(td_main["roots"]), len(td_pair["roots"]))
        diffs = [abs(td_main["roots"][i]["e_ev"] - td_pair["roots"][i]["e_ev"]) for i in range(n)]
        rpa_tda_max = max(diffs) if diffs else None
        td_main["rpa_tda_diff_ev"] = diffs
        td_main["rpa_tda_max_diff_ev"] = rpa_tda_max
        td_main[f"paired_{pair_alg.lower()}_lowest_ev"] = td_pair["roots"][0]["e_ev"] if td_pair["roots"] else None

    ref_meta["bad_roots"] = td_main.get("bad_roots")
    flags = mr_flags_for_point(
        ref_meta,
        td_main.get("roots", []),
        spin=spin,
        rpa_tda_max=rpa_tda_max,
    )

    out = {
        "species": species,
        "method": method_id,
        "R": r,
        "charge": charge,
        "spin": spin,
        "reference": ref_meta,
        "td": td_main,
        "mr_flags": flags,
        "seconds": round(time.time() - t0, 1),
    }
    out["_state"] = {"dm": mf.make_rdm1(), "e_tot": float(mf.e_tot)}
    return out


def try_spin_flip(mf, job: dict[str, Any], r: float, nroots: int) -> dict[str, Any]:
    reason = None
    results: dict[str, Any] = {"job": job["id"], "R": r, "skipped": True}
    try:
        import pyscf_forge.sftda as sftda  # noqa: WPS433
    except ImportError:
        reason = "pyscf-forge.sftda not installed"
        results["skip_reason"] = reason
        return results

    target = job["target_2S"]
    for mode in ("collinear", "noncollinear"):
        try:
            sft = sftda.SFTDA(mf)
            sft.nstates = nroots
            if mode == "noncollinear" and hasattr(sft, "collinear"):
                sft.collinear = False
            sft.kernel()
            from td_common import HARTREE2EV

            e_ev = np.asarray(sft.e) * HARTREE2EV
            roots = [{"e_ev": float(np.real(x)), "target_2S": target} for x in e_ev[: min(5, len(e_ev))]]
            results[mode] = {"roots": roots, "nroots": len(roots)}
            results["skipped"] = False
        except Exception as exc:
            results[mode] = {"error": repr(exc)[:300]}
    if results.get("skipped") and reason:
        results["skip_reason"] = reason
    return results


def _jsonable(obj: Any, _seen: set[int] | None = None) -> Any:
    """Convert numpy types and break graph cycles before json.dumps."""
    if _seen is None:
        _seen = set()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj.item()

    if isinstance(obj, np.ndarray):
        return [_jsonable(x, _seen) for x in obj.tolist()]

    oid = id(obj)
    if oid in _seen:
        return None

    if isinstance(obj, dict):
        _seen.add(oid)
        return {k: _jsonable(v, _seen) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        _seen.add(oid)
        return [_jsonable(x, _seen) for x in obj]

    return str(obj)


def checkpoint_key(species: str, method: str, r: float) -> str:
    return f"{species}|{method}|{r:.2f}"


def load_store(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    pv, _, _ = _pyscf()
    return {
        "meta": {
            "pyscf_version": pv,
            "protocol": "td_vertical_scan_v1",
            "min_mem_gib": min_mem_gib(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "pyscf_max_memory": max_memory_mb(),
        },
        "points": {},
        "spin_flip": {},
    }


def save_store(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    serial = {k: v for k, v in data.items() if k != "_state"}
    tmp.write_text(json.dumps(_jsonable(serial), indent=1) + "\n")
    tmp.replace(path)


def run_scan(args: argparse.Namespace) -> int:
    if not args.run:
        raise SystemExit("Refusing heavy TD scan without --run (see docs/td_protocol.md)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _pyscf()
    mem_kw = {
        "wait_seconds": args.mem_wait_seconds,
        "max_wait_hours": args.mem_wait_max_hours,
    }

    common = _import_common()
    grid = parse_grid(args.grid)
    arms = ["ECP_def2"]
    if args.x2c:
        arms.extend(["AE_X2C_ccDK", "AE_X2C_x2cTZ"])

    only = set(args.species.split(",")) if args.species else None
    methods = TD_METHODS
    if args.methods:
        keep = set(args.methods.split(","))
        methods = [m for m in TD_METHODS if m[0] in keep]

    out = args.out.resolve()
    data = load_store(out)
    data["meta"].update(
        {
            "grid": grid,
            "nroots": args.nroots,
            "arms": arms,
            "species": [s[0] for s in TD_SPECS],
            "methods": [m[0] for m in methods],
        }
    )

    prev_by_species: dict[str, Any] = {}

    for key, el, charge, spin in TD_SPECS:
        if only and key not in only:
            continue
        for arm_name in arms:
            _an, basis, ecp, x2c = ARMS[arm_name]
            for method_id, driver, xc, alg in methods:
                prev_state = None
                for r in grid:
                    check_memory(**mem_kw)
                    ck = checkpoint_key(key, method_id, r)
                    if ck in data["points"]:
                        prev = data["points"][ck]
                        prev_state = prev.get("_state") or prev_state
                        continue
                    pt = solve_point(
                        common,
                        key,
                        el,
                        charge,
                        spin,
                        r,
                        basis,
                        ecp,
                        x2c,
                        method_id,
                        driver,
                        xc,
                        alg,
                        args.nroots,
                        prev_state,
                    )
                    prev_state = pt.pop("_state")
                    data["points"][ck] = pt
                    save_store(out, data)
                    print(
                        f"{time.strftime('%F %T')} {key} {method_id} R={r:.2f} "
                        f"E0={pt['reference'].get('e_tot_ha')} "
                        f"E1={pt['td'].get('roots', [{}])[0].get('e_ev')} "
                        f"MR={pt['mr_flags']}",
                        flush=True,
                    )

    if not args.skip_spin_flip:
        data["meta"]["spin_flip_jobs"] = [j["id"] for j in SPIN_FLIP_JOBS]
        for job in SPIN_FLIP_JOBS:
            if only and job["species"] not in only:
                continue
            arm = ARMS["ECP_def2"]
            basis, ecp, x2c = arm[1], arm[2], arm[3]
            prev_state = None
            for r in grid:
                sfk = f"{job['id']}|{r:.2f}"
                if sfk in data["spin_flip"]:
                    continue
                check_memory(**mem_kw)
                mol = common.build(job["element"], r, job["charge"], job["ref_spin"], basis, ecp)
                _, scf, _ = _pyscf()
                mf = scf.UKS(mol)
                mf.xc = "pbe0"
                mf.verbose = 0
                mf.max_memory = max_memory_mb()
                if prev_state and prev_state.get("dm") is not None:
                    mf.kernel(dm0=prev_state["dm"])
                else:
                    mf.kernel()
                scf_stability(mf)
                prev_state = {"dm": mf.make_rdm1()}
                sf = try_spin_flip(mf, job, r, min(5, args.nroots))
                data["spin_flip"][sfk] = sf
                save_store(out, data)

    print("DONE", flush=True)
    return 0


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _pyscf()
    common = _import_common()
    basis, ecp, x2c = SMOKE_BASIS
    grid = [1.80, 2.00]
    key, el, charge, spin = TD_SPECS[0]
    pt = solve_point(
        common,
        key,
        el,
        charge,
        spin,
        grid[0],
        basis,
        ecp,
        x2c,
        "UKS_TDA_PBE0",
        "UKS",
        "pbe0",
        "TDA",
        3,
        None,
    )
    pt.pop("_state", None)
    ref = pt.get("reference", {})
    spread = ref.get("guess_spread_lowest_exc_ev", {})
    roots = pt.get("td", {}).get("roots") or []
    return {
        "smoke": True,
        "species": key,
        "grid": grid,
        "R": pt["R"],
        "e_ref_ha": ref.get("e_tot_ha"),
        "s2_ref": ref.get("s2_ref"),
        "stable": ref.get("internal_stable") and ref.get("external_stable"),
        "lowest_exc_ev": roots[0]["e_ev"] if roots else None,
        "lambda_lowest": roots[0].get("lambda") if roots else None,
        "guess_spread": spread,
        "mr_flags": pt.get("mr_flags", {}),
        "seconds": pt.get("seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        payload = run_smoke(args)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.run:
        return run_scan(args)
    print(
        json.dumps(
            {
                "dry_run": True,
                "note": "Use --smoke for CI or --run on a compute host",
                "grid_default": parse_grid(args.grid),
                "nroots_default": args.nroots,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())