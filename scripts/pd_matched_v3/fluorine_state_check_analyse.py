#!/usr/bin/env python3
"""Analyse state_check/w*/state_check_raw.json -> corrected curves, spectroscopic constants, legacy state audit.
Read-only on inputs; writes state_check/corrected_curves.json, corrected_summary.csv, legacy_state_audit.csv."""
import json, sys, csv, glob
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/scripts")
import run_ab_initio as R
SC = Path(sys.argv[2]); PDM = Path(sys.argv[3]); OLD = Path(sys.argv[1]) / "results/ab_initio_results.json"
HA_EV = 27.211386245988
def target_spin_from_key(key: str) -> int:
    """PySCF spin (2S) from species key suffix (2S+1), e.g. MnF_7 -> 6."""
    return int(key.rsplit("_", 1)[-1]) - 1
raw = {}
raw_files = sorted(glob.glob(str(SC / "w*/state_check_raw.json")))
if (SC / "state_check_raw.json").exists():
    raw_files.append(str(SC / "state_check_raw.json"))
for f in raw_files:
    for k, arms in json.load(open(f))["scans"].items():
        for a, v in arms.items():
            raw.setdefault(k, {})[a] = v
pdm = json.load(open(PDM))["audit"]; old = json.load(open(OLD))["PdF_AE_vs_ECP"]
OLDARM = {"AE_X2C_ccDK": "AE_X2C", "ECP_def2": "ECP_def2"}
def ss_t(spin): s = spin / 2; return s * (s + 1)
def fpkey(p): fp = p.get("fingerprint", {}); return f"{fp.get('n_open')}:{'+'.join(fp.get('open_shells', []))}"
def ok(p, spin): return "e_cas" in p and "e_nevpt2" in p and abs(p.get("ss", 99) - ss_t(spin)) < 0.05
def S_of(ss): return round((-1 + (1 + 4 * ss) ** 0.5) / 2, 2)
def fit(pts):
    pts = [p for p in pts if p["R"] <= 3.2]
    return R.fit_spectroscopic("Pd", [p["R"] for p in pts], [p["e_nevpt2"] for p in pts]) if len(pts) >= 3 else {"bound": False, "note": "<3 points"}
out, rows, aud = {}, [], []
for key, arms in raw.items():
    spin = target_spin_from_key(key)
    for arm, sw in arms.items():
        if not all(k in sw for k in ("fwd", "bwd")): continue
        fwd = {p["R"]: p for p in sw["fwd"]["points"]}; bwd = {p["R"]: p for p in sw["bwd"]["points"]}
        env = []
        for r in sorted(set(fwd) | set(bwd)):
            c = [(s, d[r]) for s, d in (("fwd", fwd), ("bwd", bwd)) if r in d and ok(d[r], spin)]
            if not c: continue
            s, p = min(c, key=lambda t: t[1]["e_cas"])
            gap = abs(fwd[r]["e_cas"] - bwd[r]["e_cas"]) * 1e3 if (r in fwd and r in bwd and "e_cas" in fwd[r] and "e_cas" in bwd[r]) else None
            env.append({"R": r, "sweep": s, "e_cas": p["e_cas"], "e_nevpt2": p["e_nevpt2"], "ss": p["ss"], "fp": fpkey(p), "fwd_bwd_gap_mEh": None if gap is None else round(gap, 3),
                        "bond_cas": p.get("bond_cas"), "comp": p.get("comp"), "roots": [{k: r[k] for k in ("e_casci", "e_nevpt2", "ss", "n_open")} for r in p.get("roots", [])],
                        "e_nevpt2_lowroot": min([r["e_nevpt2"] for r in p.get("roots", []) if r.get("e_nevpt2") is not None and abs(r["ss"] - ss_t(spin)) < 0.05] or [p["e_nevpt2"]]),
                        "lowroot_n_open": min([(r["e_nevpt2"], r["n_open"]) for r in p.get("roots", []) if r.get("e_nevpt2") is not None and abs(r["ss"] - ss_t(spin)) < 0.05] or [(0, None)])[1], "guess": p.get("guess"), "no_occ": [n["occ"] for n in p.get("no", [])], "ov_prev": p.get("act_overlap_prev")})
        # segments of constant fingerprint along R
        segs, cur = [], []
        for e in env:
            if cur and e["fp"] != cur[-1]["fp"]: segs.append(cur); cur = []
            cur.append(e)
        if cur: segs.append(cur)
        spec_env = fit(env)
        spec_low = fit([dict(e, e_nevpt2=e["e_nevpt2_lowroot"]) for e in env])
        emin = min((e for e in env if e["R"] <= 3.2), key=lambda e: e["e_nevpt2"]) if env else None
        seg_min = next((s for s in segs if emin in s), [])
        spec_seg = fit(seg_min) if len(seg_min) >= 3 else {"bound": False, "note": "minimum segment <3 points"}
        e5 = next((e for e in env if abs(e["R"] - 5.0) < 1e-6), None)
        de = None if (e5 is None or emin is None) else round((e5["e_nevpt2"] - emin["e_nevpt2"]) * HA_EV, 3)
        interior = bool(spec_env.get("bound"))
        cls = ("bound well, below the 5.0 A point" if interior and de is not None and de > 0 else
               "metastable well, above the 5.0 A point" if interior else "no interior minimum on 1.55-3.20 A")
        spec_f = fit([p for p in sw["fwd"]["points"] if ok(p, spin)]); spec_b = fit([p for p in sw["bwd"]["points"] if ok(p, spin)])
        spinfail = sum(1 for d in (fwd, bwd) for p in d.values() if "e_cas" in p and not ok(p, spin))
        out.setdefault(key, {})[arm] = {"target_2S+1": spin + 1, "envelope": env, "segments": [{"R": [s[0]["R"], s[-1]["R"]], "fp": s[0]["fp"], "n": len(s)} for s in segs],
            "spec_envelope": spec_env, "spec_nevpt2_lowest_root": spec_low, "spec_min_segment": spec_seg, "spec_fwd": spec_f, "spec_bwd": spec_b, "De_vs_5A_eV": de, "classification": cls,
            "at_min": emin, "spin_fail_points": spinfail, "seconds": sw["fwd"]["seconds"] + sw["bwd"]["seconds"] + sw.get("legacy", {}).get("seconds", 0)}
        bc = (emin or {}).get("bond_cas") or {}
        rows.append([key, arm, spin + 1, spec_low.get("r_e_ang", spec_low.get("r_min_grid", "")), spec_low.get("omega_e_cm1", ""), cls, spec_env.get("r_e_ang", ""), spec_env.get("omega_e_cm1", ""), spec_seg.get("r_e_ang", ""), spec_seg.get("omega_e_cm1", ""),
                     spec_f.get("r_e_ang", spec_f.get("r_min_grid", "")), spec_b.get("r_e_ang", spec_b.get("r_min_grid", "")), de, emin["R"] if emin else "", emin["fp"] if emin else "",
                     round(bc.get("mayer_sigma", float("nan")), 3) if bc else "", round(bc.get("mayer_pi", float("nan")), 3) if bc else "", round(bc.get("mayer_total", float("nan")), 3) if bc else "",
                     round(bc.get("q_lowdin_F", float("nan")), 3) if bc else "", len(segs), spinfail])
        envR = {e["R"]: e for e in env}
        for lp in sw.get("legacy", {}).get("points", []):
            r = lp["R"]; e = envR.get(r)
            pm = next((p for p in pdm[key][arm]["points"] if abs(p["R"] - r) < 1e-6), {})
            op = next((p for p in old[key][OLDARM[arm]]["points"] if abs(p["R"] - r) < 1e-6), {}) if arm in OLDARM else {}
            d = lambda a, b: None if (a is None or b is None) else round((a - b) * 1e3, 2)
            aud.append([key, arm, r, (lp.get("comp") or {}).get("Pd_d"), (lp.get("comp") or {}).get("F_p"), (lp.get("comp") or {}).get("Pd_sp"), lp.get("ss"), S_of(lp["ss"]) * 2 + 1 if "ss" in lp else "", "yes" if abs(lp.get("ss", 99) - ss_t(spin)) < 0.05 else "NO", fpkey(lp),
                        d(lp.get("e_cas"), pm.get("e_cas")), d(op.get("e_cas"), pm.get("e_cas")) if op else "", d(lp.get("e_cas"), e["e_cas"]) if e else "", e["fp"] if e else ""])
(SC / "corrected_curves.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
with open(SC / "corrected_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["species", "method", "2S+1", "r_e_NEVPT2_lowest_root_A", "omega_e_NEVPT2_lowest_root_cm1", "classification", "r_e_env_A", "omega_e_env_cm1", "r_e_minseg_A", "omega_e_minseg_cm1", "r_e_fwd", "r_e_bwd", "De_vs_5A_eV",
                                   "R_min_grid", "state_at_min(n_open:open_shells)", "Mayer_sigma", "Mayer_pi", "Mayer_total", "q_F_lowdin", "n_state_segments", "spin_fail_points"]); w.writerows(rows)
with open(SC / "legacy_state_audit.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["species", "method", "R", "legacy_active_Pd_d", "legacy_active_F_p", "legacy_active_Pd_sp", "legacy_S2", "legacy_2S+1_from_S2", "spin_ok", "legacy_state", "legacy_replay_minus_pd_matched_Ecas_mEh",
                                   "ab_initio_old_minus_pd_matched_Ecas_mEh", "legacy_minus_corrected_Ecas_mEh", "corrected_state"]); w.writerows(aud)
for r in rows: print(*r, sep=" | ")
