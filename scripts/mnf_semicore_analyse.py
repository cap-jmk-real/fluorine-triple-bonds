#!/usr/bin/env python3
"""Analyse semicore JSON: r_e, gaps, and semicore−valence shifts (mean ± std over seeds)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HA_EV = 27.211386245988

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from semicore_spec import SPECIES_BY_KEY  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", type=Path)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def target_spin(key: str) -> int:
    return int(key.rsplit("_", 1)[-1]) - 1


def ss_target(spin: int) -> float:
    s = spin / 2.0
    return s * (s + 1.0)


def spin_ok(pt: dict, spin: int) -> bool:
    return (
        "e_cas" in pt
        and "e_nevpt2" in pt
        and abs(pt.get("ss", 99.0) - ss_target(spin)) < 0.05
    )


def envelope_points(fwd: list[dict], bwd: list[dict], spin: int) -> list[dict]:
    by_r: dict[float, dict] = {}
    for sweep, pts in (("fwd", fwd), ("bwd", bwd)):
        for p in pts:
            if not spin_ok(p, spin):
                continue
            r = round(float(p["R"]), 6)
            if r not in by_r or p["e_cas"] < by_r[r]["e_cas"]:
                by_r[r] = {**p, "sweep": sweep}
    return [by_r[r] for r in sorted(by_r)]


def de_ha(env: list[dict]) -> float | None:
    if len(env) < 2:
        return None
    valid = [p for p in env if p.get("e_nevpt2") is not None]
    if not valid:
        return None
    emin = min(valid, key=lambda p: p["e_nevpt2"])
    eouter = max(valid, key=lambda p: p["R"])
    return float(eouter["e_nevpt2"] - emin["e_nevpt2"])


def species_metrics(key: str, arm_data: dict) -> dict:
    import run_ab_initio as R  # noqa: WPS433

    spin = target_spin(key)
    el = SPECIES_BY_KEY[key].element
    out: dict = {"per_seed": {}, "shifts_vs_valence": {}}
    shift_rows: list[dict] = []

    seeds = arm_data.get("seeds", {})
    for seed, sdata in sorted(seeds.items(), key=lambda x: int(x[0])):
        row: dict = {}
        for mode in ("semicore", "valence"):
            sw = sdata.get(mode, {})
            if "fwd" not in sw or "bwd" not in sw:
                continue
            env = envelope_points(sw["fwd"]["points"], sw["bwd"]["points"], spin)
            valid = [p for p in env if p.get("e_nevpt2") is not None]
            spec = (
                R.fit_spectroscopic(el, [p["R"] for p in valid], [p["e_nevpt2"] for p in valid])
                if len(valid) >= 3
                else {"bound": False}
            )
            emin = min(valid, key=lambda p: p["e_nevpt2"]) if valid else None
            row[mode] = {
                "spec_nevpt2": spec,
                "e_min_ha": emin["e_nevpt2"] if emin else None,
                "D_e_ha": de_ha(env),
            }
        out["per_seed"][seed] = row
        if "semicore" in row and "valence" in row:
            sc, vc = row["semicore"], row["valence"]
            sc_spec, vc_spec = sc.get("spec_nevpt2", {}), vc.get("spec_nevpt2", {})
            delta = {
                "delta_r_e_ang": _delta(sc_spec.get("r_e_ang"), vc_spec.get("r_e_ang")),
                "delta_omega_cm1": _delta(sc_spec.get("omega_e_cm1"), vc_spec.get("omega_e_cm1")),
                "delta_D_e_ha": _delta(sc.get("D_e_ha"), vc.get("D_e_ha")),
                "delta_e_min_ha": _delta(sc.get("e_min_ha"), vc.get("e_min_ha")),
            }
            shift_rows.append(delta)
            out["per_seed"][seed]["delta_semicore_minus_valence"] = delta

    for field in ("delta_r_e_ang", "delta_omega_cm1", "delta_D_e_ha", "delta_e_min_ha"):
        vals = [r[field] for r in shift_rows if r[field] is not None]
        if vals:
            out["shifts_vs_valence"][field] = {
                "mean": round(statistics.mean(vals), 6),
                "std": round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }
    return out


def _delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def gap_shifts(data: dict) -> dict:
    """State energy gaps: semicore vs valence (grid minima), per arm."""
    gaps: dict = {}
    pairs = (("Mn", "MnF_7", "MnF_5"), ("Ni", "NiF_2", "NiF_4"))
    for fam, k_hi, k_lo in pairs:
        if k_hi not in data.get("scans", {}) or k_lo not in data.get("scans", {}):
            continue
        arms = set(data["scans"][k_hi]) & set(data["scans"][k_lo])
        for arm in arms:
            for mode in ("semicore", "valence"):
                e_hi = _emin_seed_avg(data, k_hi, arm, mode, target_spin(k_hi))
                e_lo = _emin_seed_avg(data, k_lo, arm, mode, target_spin(k_lo))
                if e_hi is not None and e_lo is not None:
                    slot = gaps.setdefault(f"{fam}:{arm}", {})
                    slot[f"{mode}_gap_eV"] = round((e_lo - e_hi) * HA_EV, 4)
            slot = gaps.get(f"{fam}:{arm}", {})
            if "semicore_gap_eV" in slot and "valence_gap_eV" in slot:
                slot["delta_gap_semicore_minus_valence_eV"] = round(
                    slot["semicore_gap_eV"] - slot["valence_gap_eV"], 4
                )
    return gaps


def _emin_seed_avg(data, key, arm, mode, spin) -> float | None:
    arm_data = data["scans"].get(key, {}).get(arm, {})
    seeds = arm_data.get("seeds", {})
    vals = []
    for sdata in seeds.values():
        sw = sdata.get(mode, {})
        if "fwd" not in sw or "bwd" not in sw:
            continue
        env = envelope_points(sw["fwd"]["points"], sw["bwd"]["points"], spin)
        valid = [p for p in env if p.get("e_nevpt2") is not None]
        if valid:
            vals.append(min(p["e_nevpt2"] for p in valid))
    return statistics.mean(vals) if vals else None


def summarise(data: dict) -> dict:
    out = {
        "meta": data.get("meta", {}),
        "electron_counts": data.get("meta", {}).get("electron_counts", {}),
        "states": {},
        "gap_summary": gap_shifts(data),
    }
    for key, arms in data.get("scans", {}).items():
        out["states"][key] = {}
        for arm, arm_data in arms.items():
            out["states"][key][arm] = species_metrics(key, arm_data)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = json.loads(args.json_path.read_text())
    summary = summarise(data)
    text = json.dumps(summary, indent=2, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}", flush=True)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())