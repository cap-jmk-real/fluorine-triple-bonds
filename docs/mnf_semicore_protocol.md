# MnF semicore check: CAS(20,13) / SC-NEVPT2

Optional sensitivity calculation for neutral MnF **7Σ⁺** and **5Σ⁺** with Mn **3s, 3p, 3d, 4s** plus F **2p** in the active space (13 orbitals, 20 electrons).

## Method

| Item | Choice |
|------|--------|
| Hamiltonian | X2C scalar-relativistic ROHF reference |
| Basis (default) | `cc-pVTZ-DK` (matched 3d v3 AE arm `AE_X2C_ccDK`) |
| Optional basis | `x2c-TZVPall` (`AE_X2C_x2cTZ` naming in v3 manifests) |
| Active space | AVAS labels `Mn 3s`, `Mn 3p`, `Mn 3d`, `Mn 4s`, `F 2p`; threshold chosen so **exactly** 13 orbitals and 20 electrons; abort otherwise |
| Correlation | Spin-pure **CASSCF(13,20)** with `fix_spin_`, then **SC-NEVPT2** on the converged CAS |
| States | `MnF_7` (PySCF spin 6, 7Σ⁺); `MnF_5` (spin 4, 5Σ⁺) |
| Grid | 10 points, linear from **r_e − 0.20 Å** to **r_e + 0.25 Å** |
| r_e | **Required** per state on the CLI (no baked-in default); take envelope minima from the spin-pure 3d v3 scan (`fluorine_state_check_3d_mnf_nif.py` → `corrected_summary.csv` column `r_e_NEVPT2_lowest_root_A`, or legacy `results/ab_initio_results.json` until v3 3d data land) |
| Sweeps | Forward and backward with ROHF density / CAS orbital reuse between adjacent points (same pattern as v3 drivers) |
| Output | Resumable JSON checkpoint after every point (`<S²>`, CASSCF and NEVPT2 energies, natural occupations, convergence flags) |

## Compute host

Heavy runs are **not** executed in CI (`--run` guard). Example (both states, cc-pVTZ-DK):

```bash
export OMP_NUM_THREADS=16
export MNF_SEMICORE_MIN_MEM_GIB=14   # optional; default 14 GiB MemAvailable floor

python scripts/mnf_semicore_cas20_13.py --run \
  --out data/mnf_semicore/cas20_13_raw.json \
  --basis cc-pVTZ-DK \
  --MnF_7-re 1.8341 \
  --MnF_5-re 1.6738
```

Replace `r_e` values with the v3 3d envelope minima when available. Analysis:

```bash
python scripts/mnf_semicore_analyse.py data/mnf_semicore/cas20_13_raw.json \
  --out data/mnf_semicore/cas20_13_summary.json
```

## Dependencies

Open-source **PySCF** only (`numpy` implicit). Threads via `OMP_NUM_THREADS`.