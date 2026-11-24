#!/usr/bin/env python3
"""Push fluorine sigma + 2*pi bond order to the absolute physical extreme across 3d/4d/5d d^0 and main-group super-cations."""
import json, sys, time
sys.path.insert(0, "/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/scripts")
from design_fluorine_triple_bonds import eval_candidate

candidates = [
    ("[OF]3+ (10ve, 1S+)",  lambda r: f"O 0 0 0; F 0 0 {r}",  3, 0, (0.95, 1.80)),
    ("[PF]2+ (10ve, 1S+)",  lambda r: f"P 0 0 0; F 0 0 {r}",  2, 0, (1.30, 1.95)),
    ("[NbF]4+ (4d0, 1S+)",  lambda r: f"Nb 0 0 0; F 0 0 {r}", 4, 0, (1.55, 2.20)),
    ("[TaF]4+ (5d0, 1S+)",  lambda r: f"Ta 0 0 0; F 0 0 {r}", 4, 0, (1.55, 2.20)),
    ("[CrF]5+ (3d0, 1S+)",  lambda r: f"Cr 0 0 0; F 0 0 {r}", 5, 0, (1.35, 2.20)),
    ("[MoF]5+ (4d0, 1S+)",  lambda r: f"Mo 0 0 0; F 0 0 {r}", 5, 0, (1.45, 2.20)),
    ("[WF]5+ (5d0, 1S+)",   lambda r: f"W 0 0 0; F 0 0 {r}",  5, 0, (1.50, 2.20)),
    ("[ReF]6+ (5d0, 1S+)",  lambda r: f"Re 0 0 0; F 0 0 {r}", 6, 0, (1.45, 2.20)),
    ("[HfF]3+ (5d0, 1S+)",  lambda r: f"Hf 0 0 0; F 0 0 {r}", 3, 0, (1.60, 2.20)),
]

t0 = time.time()
print("=== EXTREME FLUORINE BOND-ORDER FRONTIER (X2C-PBE0 & X2C-HF / cc-pVTZ-DK) ===", flush=True)
res = []
for name, fn, q, spin, bounds in candidates:
    try:
        res.append(eval_candidate(name, fn, q, spin, bounds))
    except Exception as e:
        print(f"  {name:22s} | ERROR: {e}", flush=True)

with open("/home/s4il0r/.gemini/antigravity/scratch/fluorine-triple-bonds/results/extreme_bo_results.json", "w") as f:
    json.dump({"candidates": res, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
print(f"\nDONE in {time.time() - t0:.1f}s", flush=True)
