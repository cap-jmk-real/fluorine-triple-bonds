#!/usr/bin/env python3
"""Append [MoF]5+ and [WF]5+ to extreme_bo_results.json (idempotent)."""
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts))

from design_fluorine_triple_bonds import eval_candidate, collect_runtime_meta
from repo_paths import RESULTS_DIR

OUT = RESULTS_DIR / "extreme_bo_results.json"
extra = [
    ("[MoF]5+ (4d0, 1S+)", lambda r: f"Mo 0 0 0; F 0 0 {r}", 5, 0, (1.45, 2.20)),
    ("[WF]5+ (5d0, 1S+)", lambda r: f"W 0 0 0; F 0 0 {r}", 5, 0, (1.50, 2.20)),
]

data = json.loads(OUT.read_text()) if OUT.exists() else {"candidates": []}
names = {c["name"] for c in data.get("candidates", [])}
for name, fn, q, spin, bounds in extra:
    if name in names:
        print("skip", name)
        continue
    data["candidates"].append(eval_candidate(name, fn, q, spin, bounds))
data["meta"] = collect_runtime_meta()
data["status"] = "partial"
OUT.write_text(json.dumps(data, indent=2))
print("wrote", OUT, "n=", len(data["candidates"]))