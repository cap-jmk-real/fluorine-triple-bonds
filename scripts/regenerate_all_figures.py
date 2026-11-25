#!/usr/bin/env python3
"""Regenerate all manuscript figures from saved JSON (no new QC runs)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    steps = [
        (REPO / "scripts" / "plot_fluorine_results.py", "fig1–2 from ab_initio_results.json"),
        (REPO / "scripts" / "scan_highres_pes.py", "--plot", "fig3 from highres_pes_curves.json"),
        (REPO / "scripts" / "plot_ligand_engineering.py", "fig4 from ligand JSON screens"),
    ]
    for spec in steps:
        script = spec[0]
        extra = list(spec[1:-1]) if len(spec) > 2 else []
        label = spec[-1]
        cmd = [sys.executable, str(script)] + extra
        print(f"=== {label} ===", flush=True)
        subprocess.run(cmd, cwd=str(REPO / "scripts"), check=True)
    print("All figures written to", REPO / "figures")


if __name__ == "__main__":
    main()