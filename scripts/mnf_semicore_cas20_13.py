#!/usr/bin/env python3
"""Semicore vs valence-only correlated checks for MnF, NiF, and PdF species.

Electron counts (13 orbitals each; asserted via AVAS and in semicore_spec.ELECTRON_COUNTS):
  MnF (charge 0):  Mn 3s²3p⁶3d⁵4s² + F 2p⁵  → 20 active electrons
  NiF (charge 0):  Ni 3s²3p⁶3d⁸4s² + F 2p⁵  → 23 active electrons
  PdF* (charge q): Pd 4s²4p⁶4d¹⁰5s² + F 2p⁵ − q → 25−q  (e.g. PdF⁺² trio → 23)

See docs/mnf_semicore_protocol.md. CI: --smoke. Production: --run --species … with per-state --KEY-re.
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

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from semicore_spec import (  # noqa: E402
    ELECTRON_COUNTS,
    NCAS_SEMICORE,
    N_SEEDS_DEFAULT,
    SPECIES_BY_KEY,
    SpeciesSpec,
    arms_for_spec,
    filter_species,
    seed_hints,
)

ENV_MIN_MEM = "MNF_SEMICORE_MIN_MEM_GIB"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/semicore/semicore_raw.json"))
    p.add_argument("--basis", choices=("cc-pVTZ-DK", "x2c-TZVPall"), default="cc-pVTZ-DK")
    p.add_argument("--run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--species", default=None, help="Comma filter: MnF, NiF, PdF, or keys like MnF_7")
    p.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT, help="Initial-guess seeds per state (default 3)")
    p.add_argument("--fix-spin-shift", type=float, default=None)
    for key in SPECIES_BY_KEY:
        p.add_argument(f"--{key}-re", type=float, default=None, dest=f"re_{key.replace('+', 'p')}")
    p.add_argument("--state", default=None)
    p.add_argument("--re", type=float, default=None)
    return p.parse_args(argv)


def min_mem_gib() -> float:
    return float(os.environ.get(ENV_MIN_MEM, "14"))


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def check_memory() -> None:
    avail = mem_available_gib()
    need = min_mem_gib()
    if avail is not None and avail < need:
        raise SystemExit(
            f"MemAvailable {avail:.2f} GiB < required {need:.2f} GiB (set {ENV_MIN_MEM} to override)"
        )


def re_map_from_args(args: argparse.Namespace) -> dict[str, float]:
    m: dict[str, float] = {}
    for key in SPECIES_BY_KEY:
        dest = f"re_{key.replace('+', 'p')}"
        val = getattr(args, dest, None)
        if val is not None:
            m[key] = val
    if args.state and args.re is not None:
        m[args.state] = args.re
    return m


def avas_mo(
    mf,
    labels: tuple[str, ...],
    ncas: int,
    nelec: int,
    *,
    hints: dict | None = None,
):
    hints = hints or {}
    bias = float(hints.get("avas_threshold_bias", 0.0))
    base = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]
    thr_list = [max(0.005, t + bias) for t in base]
    order = hints.get("minao_order", ("ano", "minao"))
    last = None
    for minao in order:
        for thr in thr_list:
            try:
                n, ne, mo = avas.kernel(
                    mf,
                    list(labels),
                    threshold=thr,
                    openshell_option=3,
                    minao=minao,
                    verbose=0,
                )
                ne_int = int(round(float(ne if not isinstance(ne, tuple) else sum(ne))))
                last = (n, ne_int, thr, minao)
                if n == ncas and ne_int == nelec:
                    return mo, minao, thr
            except Exception:
                continue
    msg = f"AVAS failed: need exactly {ncas} orbitals and {nelec} electrons for labels {labels}"
    if last:
        msg += f"; last n={last[0]}, ne={last[1]} (thr={last[2]}, minao={last[3]})"
    raise RuntimeError(msg)


def rohf_follow(mol, x2c: bool, dm0, hints: dict, follow: bool = True):
    from pyscf import scf

    mf = scf.ROHF(mol)
    if x2c:
        mf = mf.x2c()
    mf.max_cycle = 250
    mf.level_shift = float(hints.get("level_shift", 0.25))
    mf.diis_space = 12
    mf.kernel(dm0)
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel(mf.make_rdm1())
    stable = None
    nfol = 0
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


def point_correlated(
    spec: SpeciesSpec,
    r: float,
    basis: str,
    ecp: str | dict | None,
    x2c: bool,
    mode: str,
    prev_state: dict | None,
    *,
    common,
    bond_fn,
    hints: dict,
    seed: int,
) -> tuple[dict[str, Any], dict | None]:
    import pyscf.mcscf as mcscf_mod

    t0 = time.time()
    res: dict[str, Any] = {"R": r, "mode": mode, "seed": seed}
    mol = common.build(spec.element, r, spec.charge, spec.spin, basis, ecp)
    dm0 = None if prev_state is None else prev_state.get("dm_hf")
    mf, hfres = rohf_follow(mol, x2c, dm0, hints, follow=True)
    res.update(hfres)
    ncas = NCAS_SEMICORE if mode == "semicore" else spec.valence_ncas
    nelec = spec.semicore_nelec if mode == "semicore" else spec.valence_nelec
    labels = spec.semicore_labels if mode == "semicore" else None

    try:
        metal = spec.element
        if mode == "valence":
            if prev_state is not None and prev_state.get("mo") is not None:
                mo0 = prev_state["mo"]
                res["guess"] = "prev-cas-mo"
            else:
                mo0, src = common.avas_mo(mf, nelec, metal, spec.valence_ncas)
                if mo0 is None:
                    raise RuntimeError(f"Valence AVAS failed for {spec.key}")
                res["guess"] = f"avas-{src}"
            ssv = 0.5 * spec.spin * (0.5 * spec.spin + 1.0)
            shift = common.fix_spin_shift() + float(hints.get("spin_shift_delta", 0.0))
            mc = mcscf_mod.CASSCF(mf, spec.valence_ncas, spec.valence_nelec)
            mc.max_cycle_macro = 60
            mc.conv_tol = 1e-6
            mc.fix_spin_(shift=shift, ss=ssv)
            ci0 = prev_state.get("ci") if prev_state else None
            mc.kernel(mo0, ci0=ci0)
        else:
            if prev_state is not None and prev_state.get("mo") is not None:
                mo0 = prev_state["mo"]
                res["guess"] = "prev-cas-mo"
            else:
                mo0, avas_src, avas_thr = avas_mo(mf, labels, NCAS_SEMICORE, spec.semicore_nelec, hints=hints)
                res["guess"] = f"avas-{avas_src}"
                res["avas_minao"] = avas_src
                res["avas_threshold"] = avas_thr
            res["avas_norb"] = ncas
            res["avas_nelec"] = nelec
            assert nelec == ELECTRON_COUNTS[spec.key], (
                f"{spec.key}: expected {ELECTRON_COUNTS[spec.key]} electrons, got {nelec}"
            )
            ssv = 0.5 * spec.spin * (0.5 * spec.spin + 1.0)
            shift = common.fix_spin_shift() + float(hints.get("spin_shift_delta", 0.0))
            mc = mcscf_mod.CASSCF(mf, NCAS_SEMICORE, spec.semicore_nelec)
            mc.max_cycle_macro = 80
            mc.conv_tol = 1e-6
            mc.fix_spin_(shift=shift, ss=ssv)
            ci0 = prev_state.get("ci") if prev_state else None
            mc.kernel(mo0, ci0=ci0)

        e_cas = float(mc.e_tot)
        res["e_cas"] = e_cas
        res["cas_conv"] = bool(mc.converged)
        res["e_nevpt2"] = e_cas + float(mrpt.NEVPT(mc).kernel())
        a, C = common.analyse(
            mol,
            mc,
            0,
            None if prev_state is None else (prev_state["mol"], prev_state["C"]),
            bond_fn,
        )
        res.update(a)
        res["no_occ"] = [n["occ"] for n in a.get("no", [])]
        st = {"mol": mol, "mo": mc.mo_coeff, "ci": mc.ci, "C": C, "dm_hf": mf.make_rdm1()}
    except Exception as exc:
        res["cas_error"] = repr(exc)[:400]
        st = prev_state
    res["seconds"] = round(time.time() - t0, 1)
    return res, st


def sweep(
    kind: str,
    spec: SpeciesSpec,
    basis: str,
    ecp: str | dict | None,
    x2c: bool,
    mode: str,
    grid: list[float],
    seed: int,
    *,
    common,
    bond_fn,
    existing: list[dict] | None,
) -> list[dict]:
    hints = seed_hints(seed)
    done = {round(float(p["R"]), 8): p for p in (existing or [])}
    pts = list(existing or [])
    st = None
    rlist = grid if kind == "fwd" else grid[::-1]
    tag = f"{spec.key}/{mode}/s{seed}"
    for r in rlist:
        rk = round(float(r), 8)
        prev = done.get(rk)
        if prev is not None and ("e_cas" in prev or "cas_error" in prev):
            continue
        check_memory()
        p, st = point_correlated(
            spec, r, basis, ecp, x2c, mode, st, common=common, bond_fn=bond_fn, hints=hints, seed=seed
        )
        if rk in done:
            pts = [x for x in pts if round(float(x["R"]), 8) != rk]
        pts.append(p)
        done[rk] = p
        print(
            f"{time.strftime('%F %T')} {tag} {kind} R={r:.3f} S2={p.get('ss')} "
            f"Enev={p.get('e_nevpt2')} conv={p.get('cas_conv')} {p.get('seconds')}s",
            flush=True,
        )
    return sorted(pts, key=lambda x: x["R"])


def run_sweeps(args: argparse.Namespace) -> int:
    if not args.run:
        raise SystemExit("Refusing heavy semicore runs without --run (docs/mnf_semicore_protocol.md)")
    re_by = re_map_from_args(args)
    if not re_by:
        raise SystemExit("--run requires --KEY-re for each selected state (no defaults)")
    check_memory()

    root = _scripts
    sys.path.insert(0, str(root / "pd_matched_v3"))
    import run_ab_initio as R  # noqa: WPS433
    import state_check_common as common  # noqa: WPS433
    from state_check_common import fix_spin_shift  # noqa: WPS433

    if args.fix_spin_shift is not None:
        os.environ["FIX_SPIN_SHIFT"] = str(args.fix_spin_shift)

    species_list = filter_species(args.species.split(",") if args.species else None)
    selected = [s for s in species_list if s.key in re_by]
    if not selected:
        raise SystemExit("No selected species have r_e on the CLI")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.loads(out.read_text())
        if out.exists()
        else {
            "meta": {
                "pyscf_version": PV,
                "protocol": "fluorine_semicore_valence_v2",
                "electron_counts": ELECTRON_COUNTS,
                "fix_spin_shift": fix_spin_shift(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "min_mem_gib": min_mem_gib(),
                "seeds": args.seeds,
            },
            "scans": {},
        }
    )

    for spec in selected:
        grid = spec.grid(re_by[spec.key]) if spec.grid_kind == "re_window" else spec.grid(re_by.get(spec.key, 1.9))
        for arm_name, basis, ecp, x2c in arms_for_spec(spec, args.basis):
            arm_slot = data["scans"].setdefault(spec.key, {}).setdefault(
                arm_name, {"re_ang": re_by[spec.key], "grid": grid}
            )
            for seed in range(args.seeds):
                seed_slot = arm_slot.setdefault("seeds", {}).setdefault(str(seed), {})
                for mode in ("semicore", "valence"):
                    mode_slot = seed_slot.setdefault(mode, {})
                    for kind in ("fwd", "bwd"):
                        existing = mode_slot.get(kind, {}).get("points")
                        t0 = time.time()
                        points = sweep(
                            kind,
                            spec,
                            basis,
                            ecp,
                            x2c,
                            mode,
                            grid,
                            seed,
                            common=common,
                            bond_fn=R.mayer_bond_order_and_charges,
                            existing=existing,
                        )
                        mode_slot[kind] = {"points": points, "seconds": round(time.time() - t0, 1)}
                        tmp = out.with_suffix(".tmp")
                        tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
                        tmp.replace(out)
    print("DONE", flush=True)
    return 0


def smoke_one_point() -> dict:
    sys.path.insert(0, str(_scripts / "pd_matched_v3"))
    import state_check_common as common  # noqa: WPS433

    spec = SPECIES_BY_KEY["MnF_7"]
    mol = common.build("Mn", 1.85, 0, 6, "cc-pVDZ-DK", None)
    mf, _ = rohf_follow(mol, True, None, seed_hints(0), follow=False)
    mo, src, thr = avas_mo(mf, spec.semicore_labels, NCAS_SEMICORE, spec.semicore_nelec, hints=seed_hints(0))
    nif = SPECIES_BY_KEY["NiF_2"]
    return {
        "smoke": True,
        "electron_counts": ELECTRON_COUNTS,
        "MnF_sem_mo_cols": int(mo.shape[1]),
        "NiF_sem_nelec": nif.semicore_nelec,
        "avas_minao": src,
        "avas_threshold": thr,
    }


def dry_run_payload(args: argparse.Namespace) -> dict:
    re_by = re_map_from_args(args)
    species_list = filter_species(args.species.split(",") if args.species else None)
    grids = {}
    for spec in species_list:
        if spec.key not in re_by:
            continue
        grids[spec.key] = spec.grid(re_by[spec.key])
    return {
        "dry_run": True,
        "species": [s.key for s in species_list if s.key in re_by],
        "electron_counts": {k: ELECTRON_COUNTS[k] for k in grids},
        "re_ang": re_by,
        "grids": grids,
        "arms_note": "Pd runs ECP_def2 + AE_X2C_x2cTZ; 3d runs selected --basis arm",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke_one_point(), indent=2))
        return 0
    if args.run:
        return run_sweeps(args)
    if re_map_from_args(args):
        print(json.dumps(dry_run_payload(args), indent=2))
        return 0
    print("Use --smoke (CI) or --run with --species and --KEY-re values (docs/mnf_semicore_protocol.md)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())