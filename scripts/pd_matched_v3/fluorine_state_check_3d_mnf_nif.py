#!/usr/bin/env python3
"""Spin-pure matched audit driver for 3d MnF / NiF species (same JSON schema as Pd v3).

Usage:
  python fluorine_state_check_3d_mnf_nif.py REPO_ROOT OUT_DIR [--mode run|test] [--species MnF_7,NiF_2]

Outputs OUT_DIR/state_check_raw.json (resumable). Heavy production scans are intended for a compute host.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pyscf import __version__ as PV

# SPECS: (key, element, charge, pyscf_spin, ncas, nelecas)
SPECS_3D = [
    ("MnF_1", "Mn", 0, 0, 9, 12),
    ("MnF_3", "Mn", 0, 2, 9, 12),
    ("MnF_5", "Mn", 0, 4, 9, 12),
    ("MnF_7", "Mn", 0, 6, 9, 12),
    ("MnF3+_2", "Mn", 3, 1, 9, 9),
    ("MnF3+_4", "Mn", 3, 3, 9, 9),
    ("MnF3+_6", "Mn", 3, 5, 9, 9),
    ("NiF_2", "Ni", 0, 1, 9, 15),
    ("NiF_4", "Ni", 0, 3, 9, 15),
    ("NiF2+_2", "Ni", 2, 1, 9, 13),
    ("NiF2+_4", "Ni", 2, 3, 9, 13),
    ("NiF3+_1", "Ni", 3, 0, 9, 12),
    ("NiF3+_3", "Ni", 3, 2, 9, 12),
    ("NiF3+_5", "Ni", 3, 4, 9, 12),
]

ARMS_3D = [
    ("AE_X2C_ccDK", "cc-pVTZ-DK", None, True),
    ("AE_X2C_x2cTZ", "x2c-tzvpall", None, True),
]

LEGACY_GRID = [1.60, 1.70, 1.80, 1.90, 2.15]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--mode", choices=("run", "test"), default="run")
    p.add_argument("--species", default=None, help="Comma filter, e.g. MnF_7,NiF_2")
    p.add_argument(
        "--fix-spin-shift",
        type=float,
        default=None,
        help="Override FIX_SPIN_SHIFT env (default 0.5)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.fix_spin_shift is not None:
        import os

        os.environ["FIX_SPIN_SHIFT"] = str(args.fix_spin_shift)

    repo = args.repo_root.resolve()
    out = args.out_dir.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from state_check_common import DEFAULT_GRID, point, sweep  # noqa: WPS433

    if args.mode == "test":
        p, _ = point(
            "Mn",
            1.85,
            0,
            6,
            "cc-pVDZ-DK",
            None,
            True,
            9,
            12,
            None,
            legacy=False,
            nevpt=True,
        )
        assert "e_cas" in p or "cas_error" in p
        print(json.dumps({k: v for k, v in p.items() if k != "no"}, default=str)[:2000])
        return 0

    only = set(args.species.split(",")) if args.species else None
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "state_check_raw.json"
    data = (
        json.loads(raw_path.read_text())
        if raw_path.exists()
        else {
            "meta": {
                "pyscf_version": PV,
                "grid": DEFAULT_GRID,
                "protocol": "3d_mnf_nif_spin_pure_v1",
                "target_spin": {k: sp for k, _, _, sp, _, _ in SPECS_3D},
            },
            "scans": {},
        }
    )

    for key, el, q, spin, ncas, nel in SPECS_3D:
        if only and key not in only:
            continue
        for arm, basis, ecp, x2c in ARMS_3D:
            for kind in ("fwd", "bwd"):
                slot = data["scans"].setdefault(key, {}).setdefault(arm, {})
                if kind in slot:
                    continue
                t0 = time.time()
                slot[kind] = {
                    "points": sweep(
                        kind,
                        el,
                        q,
                        spin,
                        ncas,
                        nel,
                        LEGACY_GRID,
                        arm,
                        basis,
                        ecp,
                        x2c,
                        log_prefix=f"{key} ",
                    ),
                    "seconds": round(time.time() - t0, 1),
                }
                tmp = raw_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
                tmp.replace(raw_path)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())