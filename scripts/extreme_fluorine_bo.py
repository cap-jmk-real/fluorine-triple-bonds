#!/usr/bin/env python3
"""Screen high-valent monofluoride cations for Mayer bond orders (X2C-PBE0 / cc-pVTZ-DK)."""
import json
import sys
import time
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from design_fluorine_triple_bonds import collect_runtime_meta, eval_candidate
from repo_paths import RESULTS_DIR

OUT_JSON = RESULTS_DIR / "extreme_bo_results.json"

candidates = [
    ("[OF]3+ (10ve, 1S+)", lambda r: f"O 0 0 0; F 0 0 {r}", 3, 0, (0.95, 1.80)),
    ("[NF]2+ (10ve, 1S+)", lambda r: f"N 0 0 0; F 0 0 {r}", 2, 0, (0.98, 1.60)),
    ("[VF]4+ (3d0, 1S+)", lambda r: f"V 0 0 0; F 0 0 {r}", 4, 0, (1.40, 2.10)),
    ("[NbF]4+ (4d0, 1S+)", lambda r: f"Nb 0 0 0; F 0 0 {r}", 4, 0, (1.55, 2.20)),
    ("[TaF]4+ (5d0, 1S+)", lambda r: f"Ta 0 0 0; F 0 0 {r}", 4, 0, (1.55, 2.20)),
    ("[TiF]3+ (3d0, 1S+)", lambda r: f"Ti 0 0 0; F 0 0 {r}", 3, 0, (1.40, 2.15)),
    ("[ZrF]3+ (4d0, 1S+)", lambda r: f"Zr 0 0 0; F 0 0 {r}", 3, 0, (1.60, 2.20)),
    ("[CrF]5+ (3d0, 1S+)", lambda r: f"Cr 0 0 0; F 0 0 {r}", 5, 0, (1.35, 2.20)),
    ("[MoF]5+ (4d0, 1S+)", lambda r: f"Mo 0 0 0; F 0 0 {r}", 5, 0, (1.45, 2.20)),
    ("[WF]5+ (5d0, 1S+)", lambda r: f"W 0 0 0; F 0 0 {r}", 5, 0, (1.50, 2.20)),
]

t0 = time.time()
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
res = []
done = set()
if OUT_JSON.exists():
    try:
        prev = json.loads(OUT_JSON.read_text())
        res = list(prev.get("candidates", []))
        done = {c["name"] for c in res}
        print(f"Resuming with {len(res)} cached candidates", flush=True)
    except json.JSONDecodeError:
        pass


def flush_partial(complete: bool = False):
    payload = {
        "meta": collect_runtime_meta(),
        "candidates": res,
        "elapsed_s": round(time.time() - t0, 1),
        "status": "complete" if complete else "partial",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)


print("=== EXTREME FLUORINE BOND-ORDER FRONTIER (X2C-PBE0 & X2C-HF / cc-pVTZ-DK) ===", flush=True)
for name, fn, q, spin, bounds in candidates:
    if name in done:
        continue
    try:
        res.append(eval_candidate(name, fn, q, spin, bounds))
        flush_partial()
    except Exception as exc:
        print(f"  {name:22s} | ERROR: {exc}", flush=True)

flush_partial(complete=True)
print(f"\nDONE in {time.time() - t0:.1f}s -> {OUT_JSON}", flush=True)