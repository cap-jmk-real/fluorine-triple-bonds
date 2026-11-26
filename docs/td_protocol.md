# TD vertical-excitation protocol (`scripts/excited_states/`)

## Purpose

Answer, from single-reference linear-response theory:

1. What lowest vertical excitations imply about the claimed F triple bond / bond nature.
2. Whether TD confirms multireference character for `[PdF]²⁺` doublet and `[PdF]³⁺` singlet/triplet references.
3. Whether a low quintet-like root appears when TD is run from singlet/triplet along 1.70–2.15 Å (crossing vs `PdF3+_5` ground curve).
4. Where UKS/UHF TD breaks down (MR flags, unstable SCF, negative/imaginary roots, large RPA–TDA gaps).

## Species and reference spins (2S)

| key | charge | PySCF `spin` (2S) |
|-----|--------|-------------------|
| `PdF_2` | 0 | 1 |
| `PdF2+_2` | 2 | 1 |
| `PdF2+_4` | 2 | 3 |
| `PdF3+_1` | 3 | 0 |
| `PdF3+_3` | 3 | 2 |
| `PdF3+_5` | 3 | 4 |
| `MnF_7` | 0 | 6 |
| `MnF_5` | 0 | 4 |
| `NiF_2` | 0 | 1 |
| `NiF_4` | 0 | 3 |

Geometries: linear MF along +z; builders reuse `scripts/pd_matched_v3/state_check_common.py` (`build`).

## Grid

Default **22 points**: 1.55, 1.60, …, 2.60 Å (`--grid start:stop:step` or comma list).

## Basis arms

- **Default (matched ECP):** `def2-TZVP` + `def2-TZVP` ECP on Pd/Mn/Ni (`ECP_def2`).
- **Optional `--x2c`:** `cc-pVTZ-DK` (AE_X2C_ccDK) and Pd `x2c-TZVPall` / F `cc-pVTZ-DK` (AE_X2C_x2cTZ).

## Methods (per reference SCF)

| id | SCF | response |
|----|-----|----------|
| `UKS_TDDFT_PBE0` | UKS PBE0 | RPA |
| `UKS_TDA_PBE0` | UKS PBE0 | TDA |
| `UKS_TDDFT_CAMB3LYP` | UKS CAM-B3LYP | RPA |
| `UKS_TDA_CAMB3LYP` | UKS CAM-B3LYP | TDA |
| `UHF_TDDFT` | UHF | TDHF RPA |
| `UHF_TDA` | UHF | CIS/TDA |

Default **`--nroots 20`**.

### Spin-flip TDA (optional)

When `pyscf-forge` / `sftda` is importable:

- `PdF2+_4` → doublet
- `PdF3+_5` → triplet / singlet
- `MnF_7` → quintet

Collinear and noncollinear modes are attempted; otherwise a `skip_reason` is recorded.

## Per-point outputs

- Reference energy, `<S²>`, internal/external SCF stability (instability ⇒ MR flag).
- Three SCF seeds (`minao`, `atom`, neighbour density); winning lowest stable energy; mean ± std of lowest TDA root across stable guesses.
- Per root: ΔE (eV), oscillator strength, approximate excited `<S²>`, dominant amplitudes, NTO tags, Λ (Σ/Π/Δ/Φ via cylindrical projection), RPA–TDA differences.
- Negative/imaginary roots flagged (triplet-instability signal).

### MR diagnostics (per point)

| flag | criterion |
|------|-----------|
| `lowest_excitation_lt_0p5eV` | first root &lt; 0.5 eV |
| `rpa_tda_gap_gt_0p3eV` | max \|RPA−TDA\| &gt; 0.3 eV |
| `excited_s2_deviation_gt_0p5` | \|⟨S²⟩_exc − S(S+1)\| &gt; 0.5 for target spin |
| `reference_instability` | SCF not internally and externally stable |
| `negative_or_imaginary_root` | bad root in TD spectrum |

## Checkpointing

JSON at `--out` (default `data/td_scan/td_raw.json`); one entry per `(species, method, R)`. Safe to resume.

## Memory / threads

- Before each grid point, if `MemAvailable` &lt; `TD_MIN_MEM_GIB` (default **6** GiB), the driver **sleeps and rechecks** in the same process instead of exiting (avoids a full PySCF cold start on every transient dip).
- `--mem-wait-seconds` (default **300**): pause between rechecks.
- `--mem-wait-max-hours` (default **24**): exit only if memory stays low for longer than this wall time; each wait is logged.
- `PYSCF_MAX_MEMORY` (MB, default 4000) and `OMP_NUM_THREADS` respected.
- `python scripts/excited_states/td_scan.py --help` does not import PySCF (heavy imports load only on `--smoke` / `--run`).

## Guards

| mode | use |
|------|-----|
| *(none)* | dry-run JSON hint |
| `--smoke` | CI: `def2-SVP`, 2 points, 3 roots, `PdF_2`, &lt; 60 s |
| `--run` | full production scan (compute host only) |

## Analysis

```bash
python scripts/excited_states/td_analyse.py \\
  --raw data/td_scan/td_raw.json \\
  --out-dir data/td_scan/analysed \\
  --v3-summary data/pd_matched_v3/corrected_summary.csv
```

Writes vertical spectrum at r_e, lowest-root curves, MR table, quintet-band crossings, and `td_digest.md`.

## Compute-host commands (16 threads)

```bash
export OMP_NUM_THREADS=16
export PYSCF_MAX_MEMORY=8000
export TD_MIN_MEM_GIB=6

# ECP matched arm (production)
python scripts/excited_states/td_scan.py --run \\
  --out data/td_scan/td_raw_ecp_def2.json \\
  --species PdF_2,PdF2+_2,PdF2+_4,PdF3+_1,PdF3+_3,PdF3+_5,MnF_7,MnF_5,NiF_2,NiF_4 \\
  --nroots 20

# Optional all-electron arms
python scripts/excited_states/td_scan.py --run --x2c \\
  --out data/td_scan/td_raw_x2c.json --nroots 20

python scripts/excited_states/td_analyse.py \\
  --raw data/td_scan/td_raw_ecp_def2.json \\
  --out-dir data/td_scan/analysed_ecp_def2
```

### Rough cost (16 cores, def2-TZVP ECP, 22 points × 20 roots)

| species group | methods × points | wall-time order (per species, all 6 methods) |
|---------------|------------------|-----------------------------------------------|
| PdF / NiF (low spin) | 6 × 22 | ~20–40 min |
| PdF²⁺ / PdF³⁺ open-shell | 6 × 22 | ~35–90 min |
| MnF₇ / MnF₅ | 6 × 22 | ~45–120 min |

Full default campaign (10 species × 6 methods × 22 points) is **~80–200 CPU-hours** depending on convergence; run species slices in parallel jobs sharing the same JSON with distinct `--species` filters if the filesystem allows merged post-processing.