#!/usr/bin/env python3
"""Summarise TD scan JSON → CSV/JSON + markdown digest.

  python scripts/excited_states/td_analyse.py \\
    --raw data/td_scan/td_raw.json \\
    --out-dir data/td_scan/analysed \\
    [--v3-summary data/pd_matched_v3/corrected_summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from td_common import TD_SPECS, default_grid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--v3-summary",
        type=Path,
        default=Path("data/pd_matched_v3/corrected_summary.csv"),
    )
    return p.parse_args(argv)


def load_re_table(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("method") == "ECP_def2":
                try:
                    re = float(row["r_e_NEVPT2_lowest_root_A"])
                except (TypeError, ValueError):
                    continue
                out[row["species"]] = re
    return out


def infer_re(data: dict, species: str) -> float | None:
    pts = [p for k, p in data["points"].items() if p["species"] == species]
    if not pts:
        return None
    return float(min(pts, key=lambda p: p["reference"]["e_tot_ha"])["R"])


def summarise_vertical(data: dict, re_by: dict[str, float]) -> list[dict]:
    rows = []
    for species, _, _charge, spin in TD_SPECS:
        re = re_by.get(species) or infer_re(data, species)
        if re is None:
            continue
        best = None
        for pt in data["points"].values():
            if pt["species"] != species or abs(pt["R"] - re) > 0.03:
                continue
            roots = pt.get("td", {}).get("roots") or []
            if not roots:
                continue
            cand = {
                "species": species,
                "method": pt["method"],
                "R": pt["R"],
                "re_used": re,
                "e1_ev": roots[0]["e_ev"],
                "f1": roots[0].get("f_osc"),
                "lambda1": roots[0].get("lambda"),
                "character1": (roots[0].get("dominant_excitations") or [{}])[0].get("character"),
                "mr_flags": pt.get("mr_flags", {}),
            }
            if best is None or cand["e1_ev"] < best["e1_ev"]:
                best = cand
        if best:
            rows.append(best)
    return rows


def lowest_root_curves(data: dict) -> dict[str, list[dict]]:
    curves: dict[str, list[dict]] = defaultdict(list)
    for pt in data["points"].values():
        roots = pt.get("td", {}).get("roots") or []
        if not roots:
            continue
        key = f"{pt['species']}|{pt['method']}"
        curves[key].append(
            {
                "R": pt["R"],
                "e1_ev": roots[0]["e_ev"],
                "lambda": roots[0].get("lambda"),
                "mr_flags": pt.get("mr_flags"),
            }
        )
    for k in curves:
        curves[k].sort(key=lambda x: x["R"])
    return dict(curves)


def mr_table(data: dict) -> list[dict]:
    rows = []
    for pt in data["points"].values():
        flags = pt.get("mr_flags") or {}
        if not flags:
            continue
        rows.append(
            {
                "species": pt["species"],
                "method": pt["method"],
                "R": pt["R"],
                **flags,
            }
        )
    return sorted(rows, key=lambda r: (r["species"], r["method"], r["R"]))


def quintet_crossings(data: dict) -> list[dict]:
    """Excited-state adiabatic proxy vs PdF3+_5 reference in 1.70–2.15 Å."""
    band = [r for r in default_grid() if 1.70 <= r <= 2.15]
    qgs = {
        p["R"]: p["reference"]["e_tot_ha"]
        for p in data["points"].values()
        if p["species"] == "PdF3+_5"
    }
    rows = []
    for ref_spec in ("PdF3+_1", "PdF3+_3"):
        for method in sorted({p["method"] for p in data["points"].values() if p["species"] == ref_spec}):
            for r in band:
                ref_pt = next(
                    (
                        p
                        for p in data["points"].values()
                        if p["species"] == ref_spec and p["method"] == method and abs(p["R"] - r) < 1e-3
                    ),
                    None,
                )
                if ref_pt is None or r not in qgs:
                    continue
                roots = ref_pt["td"].get("roots") or []
                for i, rt in enumerate(roots[:5]):
                    e_exc = ref_pt["reference"]["e_tot_ha"] + rt["e_ev"] / 27.211386245988
                    eq = qgs[r]
                    rows.append(
                        {
                            "ref_species": ref_spec,
                            "method": method,
                            "R": r,
                            "root": i,
                            "E_ref_plus_exc_ha": e_exc,
                            "E_quintet_gs_ha": eq,
                            "delta_ha": e_exc - eq,
                            "crossing_like": e_exc <= eq,
                            "e_exc_ev": rt["e_ev"],
                        }
                    )
    return rows


def markdown_digest(
    vertical: list[dict],
    curves: dict,
    mr: list[dict],
    crossings: list[dict],
) -> str:
    lines = [
        "# TD vertical-excitation digest",
        "",
        "## Scientific targets",
        "",
        "1. Lowest vertical roots and NTO/LMCT character at r_e (ECP_def2 where available).",
        "2. MR flags (low roots, RPA–TDA gap, S² deviation, SCF instability, imaginary roots).",
        "3. PdF₃⁺ quintet vs singlet/triplet excited-state crossings in 1.70–2.15 Å.",
        "4. Spin-flip TDA entries (if pyscf-forge available) in raw `spin_flip` block.",
        "",
        "## Vertical spectrum (one row per species, lowest root at r_e)",
        "",
        "| species | method | r_e (Å) | E₁ (eV) | Λ | character | MR |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for v in vertical:
        mr = ",".join(sorted(v["mr_flags"])) if v["mr_flags"] else "—"
        lines.append(
            f"| {v['species']} | {v['method']} | {v['re_used']:.3f} | {v['e1_ev']:.4f} | "
            f"{v.get('lambda1', '?')} | {v.get('character1', '?')} | {mr} |"
        )
    lines.extend(
        [
            "",
            f"## MR-flagged points: **{len(mr)}**",
            "",
            f"## Quintet-crossing checks (band 1.70–2.15 Å): **{len(crossings)}** tuples",
            f"(**{sum(1 for c in crossings if c['crossing_like'])}** with E_exc ≤ E_quintet)",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = json.loads(args.raw.read_text())
    re_by = load_re_table(args.v3_summary)

    vertical = summarise_vertical(data, re_by)
    curves = lowest_root_curves(data)
    mr = mr_table(data)
    crossings = quintet_crossings(data)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "vertical_at_re": vertical,
        "lowest_root_curves": curves,
        "mr_flags": mr,
        "quintet_crossings_1p70_2p15": crossings,
        "re_angstrom": re_by,
        "spin_flip": data.get("spin_flip", {}),
    }
    (args.out_dir / "td_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")

    with (args.out_dir / "td_vertical.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["species", "method", "re_used", "R", "e1_ev", "f1", "lambda1", "character1"],
        )
        w.writeheader()
        for row in vertical:
            w.writerow({k: row.get(k) for k in w.fieldnames})

    with (args.out_dir / "td_mr_flags.csv").open("w", newline="", encoding="utf-8") as fh:
        if mr:
            fields = sorted(set().union(*(r.keys() for r in mr)))
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in mr:
                w.writerow(row)

    (args.out_dir / "td_digest.md").write_text(markdown_digest(vertical, curves, mr, crossings))
    print(f"Wrote analysis to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())