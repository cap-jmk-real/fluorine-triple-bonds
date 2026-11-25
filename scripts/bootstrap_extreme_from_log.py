#!/usr/bin/env python3
"""One-off: seed extreme_bo_results.json from logs_extreme_bo.txt partial run."""
import json
import re
from pathlib import Path

from repo_paths import RESULTS_DIR

LOG = RESULTS_DIR / "logs_extreme_bo.txt"
OUT = RESULTS_DIR / "extreme_bo_results.json"

pattern = re.compile(
    r"\s+\[([^\]]+)\]\s+\| bound=(\w+)\s+\| r_e=([\d.]+) A \| "
    r"PBE0 BO=([\d.]+) \(sig=([\d.]+), pi=([\d.]+)\) qF=([+-]?[\d.]+) \| "
    r"HF BO=([\d.]+) \(pi=([\d.]+)\)"
)

def main():
    text = LOG.read_text()
    candidates = []
    for m in pattern.finditer(text):
        name = f"[{m.group(1)}]"
        candidates.append(
            {
                "name": name,
                "bound": m.group(2) == "True",
                "r_e_ang": float(m.group(3)),
                "pbe0": {
                    "mayer_total": float(m.group(4)),
                    "mayer_sigma": float(m.group(5)),
                    "mayer_pi": float(m.group(6)),
                    "q_F": float(m.group(7)),
                },
                "hf": {
                    "mayer_total": float(m.group(8)),
                    "mayer_pi": float(m.group(9)),
                },
            }
        )
    payload = {"candidates": candidates, "status": "partial", "seeded_from_log": str(LOG)}
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(candidates)} rows to {OUT}")

if __name__ == "__main__":
    main()