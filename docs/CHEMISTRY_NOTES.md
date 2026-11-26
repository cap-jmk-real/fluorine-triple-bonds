# PdF / [PdF]2+ / [PdF]3+ — electronic-state check (25.11.2026)

Scope: the 20 (species, method) scans of fluorine PR #2 ledger item 11 (2×2 matched audit: ECP_ccPP, AE_X2C_x2cTZ, AE_X2C_ccDK, ECP_def2), same CAS(n,9) active spaces and SC-NEVPT2 as `run_ab_initio.py` PART 4. PySCF 2.15.1, CPU only.

## 1. Diagnosis from the existing data (before any rerun)
The legacy protocol (ROHF canonical guess → `mcscf.CASSCF(mf, 9, nelec)`; no spin penalty, no <S^2>, no state tracking, 5-point grid) does not control which electronic state it lands on:
- "Singlet" [PdF]3+ (Ms = 0) and "triplet" [PdF]3+ (Ms = 1) give the same CASSCF energy (|ΔE| < 1e-6 Eh) at R = 1.80 and 1.95 Å in all four protocols and at 1.70/2.15 Å in three. Both have four open shells (natural occupations ≈ 1.01–1.18), i.e. both solutions are the Ms components of one higher-spin state (S ≥ 1, most likely the quintet). The CI solver returns the lowest root of a given Ms, which is not spin-adapted unless a spin penalty is used.
- "Doublet" and "quartet" [PdF]2+ coincide at 1.85 Å (both AE protocols), three open shells: the doublet calculation fell onto the quartet.
- Along single scans the natural-occupation pattern jumps (e.g. [PdF]2+ doublet: one open shell at 1.65–1.75 Å, three open shells at 1.85–2.15 Å), so curves splice different states and the fitted r_e/ω_e/"bound" labels have no single-state meaning.
- The 21.11 run (`results/ab_initio_results.json`) and the 25.11 recomputation (same code, same PySCF) differ for 5/10 legacy curves; at the differing points the ROHF energy differs by up to 90 mEh (different SCF solutions) or the CASSCF lands on a different root. Neither old nor new legacy data is "right"; both are protocol accidents.
- "Bound?" in Table 6 means only "minimum not at the end of a 5-point grid that stops at 2.15 Å".

## 2. Protocol of the state check (`scripts/pd_matched_v3/fluorine_state_check.py`)
Grid 1.55–3.20 Å (15 points) + 5.00 Å. For each (species, method):
- `legacy`: the 5 legacy points with the legacy protocol, now recording <S^2>, natural occupations and NO character (CASSCF only).
- `fwd` (ascending R) and `bwd` (descending from 5.00 Å): ROHF from the previous point's density with ROHF internal-stability analysis followed until stable (≤4 steps); spin-pure CASSCF (`fix_spin_(shift=0.5, ss=S(S+1))`, conv_tol 1e-7, ≤100 macro-iterations) started from the previous point's CAS orbitals (`project_init_guess`) and CI vector; SC-NEVPT2.
- Per point: <S^2>, CAS natural occupations with σ/π/δ (AO m_l) and Pd/F character, open-shell fingerprint, minimal singular value of the active-space overlap with the previous point (cross-geometry AO overlap), Mayer σ/π/total and Löwdin q(F), ROHF stability status.
- Corrected curve = per-R lower CASSCF envelope of the spin-pure fwd/bwd solutions; the state fingerprint segments along R are reported; r_e/ω_e from the envelope (≤3.20 Å) and from the fingerprint segment containing the minimum; binding = interior minimum, plus E(5.00 Å) − E_min (positive = below the 5 Å point; for multiply charged species the 5 Å point still contains Coulomb repulsion, so this is a metastability indicator, not D_e).
Limitations: no point-group/Λ symmetry labels (C1 code; σ/π/δ character of the open shells is used instead); one root per spin (no state averaging), so a near-degenerate excited state of the same spin can still be missed where fwd and bwd disagree; 5 Å is not the separated-ion limit; scalar X2C, no SOC.

## 3. Results
(appended automatically by the analysis step; see corrected_summary.csv, corrected_curves.json, legacy_state_audit.csv)

## 3. v3 results: chemical interpretation (2026-11-26)
Protocol: density-followed, stability-checked X2C ROHF -> AVAS (Pd 4d, Pd 5s, F 2p) -> spin-pure CASCI (3 roots, fix_spin 0.5) -> SC-NEVPT2 per root; lowest spin-pure root envelope. Data data/pd_matched_v3/corrected_summary.csv, corrected_curves.json.

| species | state | ECP_ccPP | AE_X2C_ccDK | AE_X2C_x2cTZ | ECP_def2 | verdict |
|---|---|---|---|---|---|---|
| PdF | 2Sigma+ | 1.912 A / 589 cm-1 | 1.900 / 584 | 1.914 / 562 | 1.920 / 571 | bound in all; single sigma(Pd) open shell; BO 0.40-0.61, q(F) -0.36..-0.54 |
| [PdF]2+ | quartet | 2.125 / 311 | 2.108 / 323 | 2.149 / 290 | 2.144 / 287 | bound, shallow (1-2 eV below 5 A point); delta(Pd)+pi(F)+sigma(Pd); q(F) +0.24..+0.34 |
| [PdF]2+ | doublet | - | - | - | - | 0.09-0.20 eV above quartet; 5-7 state segments, no single-state minimum |
| [PdF]3+ | triplet | repulsive | repulsive | repulsive | repulsive | E falls 7-8 eV 1.55->5 A; q(F)->+1 beyond ~2.2 A (Coulomb explosion) |
| [PdF]3+ | singlet | repulsive | repulsive | repulsive | repulsive | 0.13-0.25 eV above triplet; kinked, multi-state |

Notes:
- No protocol reversal: core (AE vs ECP) changes r_e(PdF) <=0.011 A, basis family <=0.02 A; [PdF]2+ quartet r_e spread 0.04 A.
- CSV artefacts: De_vs_5A for [PdF]2+ quartet (6-18 eV) is root-0 based and unreliable; [PdF]3+ "bound" rows with omega 1200-7800 cm-1 are fits on kinked multi-segment curves; the 5 A asymptotes differ (AE: Pd3+ + F, q(F)=0; ECP: Pd2+ + F+, q(F)=1), so no De for [PdF]3+.
- ~0.4-0.5 eV bump at 1.70 A in [PdF]2+ = ROHF solution switch, not physical.
- Legacy values on wrong/ill-defined states: PdF AE 2.025 A/1119 cm-1 and ECP 1.917/476 (semicore in active space); [PdF]2+ "AE bound 1.838 A/481 cm-1, pi-BO 0.401; ECP unbound" (both wrong); [PdF]3+ "ECP bound at 1.976 A" wrong, "AE unbound" right in conclusion but on a spin-mixed state (singlet==triplet to <1e-6 Eh).
- def2 arm: "ECP def2-TZVP not found for F" harmless - F gets all-electron def2-TZVP (no F ECP exists); Pd ECP28. No rerun.
- Spin failures: 4 points PdF3+_1 AE_X2C_x2cTZ (fix_spin 0.5). Targeted rerun FIX_SPIN_SHIFT=2.0 -> data/pd_matched_v3/v3-shift2/ (see 3.1).

### 3.1 Targeted spin-penalty rerun (data/pd_matched_v3/v3-shift2/, PdF3+_1 AE_X2C_x2cTZ, FIX_SPIN_SHIFT=2.0; 07:10-07:20 CET)
- Spin failures fixed: 4 -> 0 points without a spin-pure (S2<0.05) root.
- But lowest spin-pure NEVPT2 energies at 1.80-2.50 A shift by -4.3 ... +1.0 eV versus v3 (identical <0.01 eV at 1.55-1.75 and 2.8-5.0 A): in this window several ROHF reference solutions (charge distributions Pd3+-F vs Pd2+-F+) compete and the followed reference is path-dependent. The singlet curve stays non-monotonic with an overall drop of ~8.4 eV from 1.55 to 5 A.
- Verdict unchanged: singlet [PdF]3+ repulsive / no reliable minimum, ~0.1-0.3 eV above the triplet near 1.6-1.8 A. Its curve at 1.8-2.5 A is not quantitatively converged; report no r_e/omega_e. Fixing it needs state-averaged CASSCF over the charge-transfer states or a larger active space - more than a short rerun, so it was not attempted.

## Repository note
Copied from the compute host on 2026-11-26 into branch data/pd-matched-v3. Host-specific paths were replaced by repo-relative ones. `scripts/pd_matched_v3/fluorine_state_check.py` is the v3 script plus an optional FIX_SPIN_SHIFT environment variable (default 0.5 = exactly the v3 run; 2.0 = v3-shift2 rerun). Usage: `python fluorine_state_check.py <repo_root> <out_dir> [run] [species,...]`; `python fluorine_state_check_analyse.py <repo_root> <state_check_dir> <pd_matched_audit.json>`. The legacy-protocol input pd_matched_audit.json (25.11 run) is not included; the analysis outputs derived from it are (legacy_state_audit.csv).
