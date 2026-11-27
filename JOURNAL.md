# Research journal (sanitized snapshot)

This is a sanitized research-journal snapshot dated **2026-11-27 07:46 Europe/Berlin**. It records scientific status, publication status and named git heads for [fluorine-triple-bonds](https://github.com/cap-jmk-real/fluorine-triple-bonds). Operational hostnames, machine paths and process identifiers are omitted on purpose. This file is not permission to merge a pull request or to submit a paper.

Canonical operational handover notes live outside this repository.

## Goal

Reproduce and, if the evidence supports it, revise the ChemRxiv preprint *Can fluorine form triple bonds?* (PySCF recalculation of Pd/Mn/Ni fluorides and related species) until paper-review ≥8.5 with no blockers. The matched-protocol audit has already retracted the headline AE-vs-ECP binding-reversal claim for [PdF]²⁺ / [PdF]³⁺ (spin-contamination artefact). Bring the manuscript and supporting computations (spin-pure 3d reruns, bond-index check, semicore, TD-DFT/TDA/TDHF, MR/SOC) in line with that finding. Nothing is merged or submitted without Julian's yes.

## Publication status

- **Stage:** drafting / evidence revision. Semicore, TD and remaining MR/SOC work are still in flight.
- **paper-review:** **6.0 / 10**, 1 blocker, dated 2026-11-26, version “PR #3 head 5569e3ae”. GitHub PR #3 head at 2026-11-27 is `f16e9c5f20a7d610c23487b8b477a81971be95b8` (newer). Whether 6.0 still applies to that head: UNKNOWN. Earlier: 4.9 with 2 blockers on PR #2 head `3987ab1d` (2026-11-25). Julian, 2026-11-26, verbatim: „Ja okay gehe bis 8.5 mit den Rechnungen auf jeden Fall.“
- **Preprint:** ChemRxiv preprint *Can fluorine form triple bonds?* is the starting point being recalculated; a new DOI / resubmission for the revised manuscript has not been made. No journal submission.
- **Venue target:** diamond open access, no APC, Scopus/WoS or similar (Julian, 2026-11-24 17:29 Europe/Berlin). TMLR and BJOC fit; JCIM, JCTC, and Minds and Machines do not.
- **Repo visibility:** public (verified 2026-11-27). Licence (merged PR #1): MIT code, CC BY 4.0 paper and data. Default `main` is `73c1979d3e12435f2beeebb553287e8fcd7d9f25`.
- **Ready to submit / merge?** No. Review 6.0 < 8.5; PR #3 is draft; compute unfinished; nothing is merged or submitted without Julian's yes on the named head.

## Current step

As of 2026-11-27 07:44 Europe/Berlin:

1. **TD vertical-excitation campaign** is still running on PdF3+_5, ECP def2 window 1.70–2.15 Å. The raw ECP file has **20 points**, last write 2026-11-26 18:36:41 Europe/Berlin: UKS_TDDFT_PBE0 (1.55 leftover + full window), UKS_TDA_PBE0 (full window), UKS_TDDFT_CAMB3LYP (full window), UKS_TDA_CAMB3LYP only **1.70** so far. The worker process was still alive at 07:44. Spin-flip TDA is skipped (`pyscf-forge` not available). Do not start a second copy.
2. **MR/SOC check.** The previously recorded worker was not running at 07:43. Last recorded partial verdict: 40/63 units written. How many of the remaining AE / legacy / SOC units finished after that snapshot is UNKNOWN in this pass. Do not start a second copy.
3. **Semicore queue** is active but often waiting on a 14 GiB free-memory floor. Current job is still **MnF**. A short failed relaunch (rc=1, 16 s) was recorded at 07:16:09 Europe/Berlin; the queue then returned to the memory wait (40 unit restarts by 07:18). Earlier overnight MnF_5/semicore/s2 points were written when memory allowed (R around 1.600–1.750 Å, mixed convergence). PdF, NiF and 3d x2cTZ have not been reached as the current job. Do not lower the memory gate and do not start a second copy.

## Decision log

- **2026-11-27 07:39–07:40 Europe/Berlin — Julian via Grok Bot:** keep this journal in the four-paper set (with Publication status) and open a sanitized journal PR on this repo. Do not merge the PR until his yes.
- **2026-11-26 — Julian, verbatim:** „Ja okay gehe bis 8.5 mit den Rechnungen auf jeden Fall.“ Also: „Mache gerne semicore Rechnung“. Later the same day: „Ja mach vollen Umfang …“ — full semicore scope, including the complete Pd set. Order recorded: MnF first, then the full Pd set, then NiF, then 3d x2cTZ when memory allows. Local compute only; no Azure.
- **2026-11-24 — Julian, via Grok Bot:** diamond-OA venue target (no APC; Scopus/WoS or similar).

## Results (selected)

- Matched-protocol audit: headline AE-vs-ECP binding-reversal claim for [PdF]²⁺ / [PdF]³⁺ retracted as a spin-contamination artefact (see draft PR #3).
- 3d ccDK arm finished 2026-11-26 09:38:03 Europe/Berlin, rc=0, wall 5493 s, 28/28 sweeps (recorded in the earlier journal notes).
- PySCF version in the 3d meta: 2.15.1.
- Open drafts: [#3](https://github.com/cap-jmk-real/fluorine-triple-bonds/pull/3) `f16e9c5f20a7d610c23487b8b477a81971be95b8` (Pd matched-v3 audit); [#2](https://github.com/cap-jmk-real/fluorine-triple-bonds/pull/2) `3987ab1de88f1e6007600a352ead851db145b477` (conflicting).

## Blockers

- paper-review 6.0 < 8.5 on the dated #3 head; not re-measured on `f16e9c5f`.
- TD window for TDA_CAMB3LYP incomplete; semicore still on MnF and often memory-waiting; MR/SOC remainder UNKNOWN.
- No merge or submission without Julian's yes.

## Next steps

1. Let the running TD and semicore work finish under the existing memory gates. Do not start duplicates.
2. Re-measure paper-review on the current PR #3 head after the outstanding computations are in.
3. Nothing is merged or submitted without Julian's yes on the named head.

## Last updated

2026-11-27 07:46 Europe/Berlin.
