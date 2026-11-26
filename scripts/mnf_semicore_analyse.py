#!/usr/bin/env python3
"""Analyse mnf_semicore_cas20_13 JSON: r_e, 7–5 gap, spectroscopic constants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HA_EV = 27.211386245988


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", type=Path, help="cas20_13_raw.json from the compute driver")
    p.add_argument("--out", type=Path, default=None, help="Write summary JSON (default: stdout only)")
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
            cand = {**p, "sweep": sweep}
            if r not in by_r or p["e_cas"] < by_r[r]["e_cas"]:
                by_r[r] = cand
    return [by_r[r] for r in sorted(by_r)]


def summarise(data: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_ab_initio as R  # noqa: WPS433

    out: dict = {"meta": data.get("meta", {}), "states": {}, "gap_eV": None}
    e_min: dict[str, float] = {}
    for key, arms in data.get("scans", {}).items():
        spin = target_spin(key)
        out["states"][key] = {}
        for arm, sw in arms.items():
            if "fwd" not in sw or "bwd" not in sw:
                continue
            env = envelope_points(sw["fwd"]["points"], sw["bwd"]["points"], spin)
            valid = [p for p in env if p.get("e_nevpt2") is not None]
            spec = (
                R.fit_spectroscopic("Mn", [p["R"] for p in valid], [p["e_nevpt2"] for p in valid])
                if len(valid) >= 3
                else {"bound": False, "note": "<3 spin-pure NEVPT2 points"}
            )
            emin = min(valid, key=lambda p: p["e_nevpt2"]) if valid else None
            if emin is not None:
                e_min[f"{key}:{arm}"] = float(emin["e_nevpt2"])
            out["states"][key][arm] = {
                "r_e_input_ang": sw.get("re_ang"),
                "n_envelope_points": len(env),
                "spec_nevpt2": spec,
                "e_min_nevpt2_ha": emin["e_nevpt2"] if emin else None,
                "r_at_e_min": emin["R"] if emin else None,
            }
    keys_7 = [k for k in e_min if k.startswith("MnF_7:")]
    keys_5 = [k for k in e_min if k.startswith("MnF_5:")]
    if keys_7 and keys_5:
        e7 = min(e_min[k] for k in keys_7)
        e5 = min(e_min[k] for k in keys_5)
        out["gap_eV"] = round((e5 - e7) * HA_EV, 4)
        out["gap_note"] = "E(5Σ⁺) − E(7Σ⁺) at grid NEVPT2 minima (negative ⇒ 5Σ⁺ below 7Σ⁺)"
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