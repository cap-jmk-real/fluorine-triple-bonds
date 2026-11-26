# Fluorine semicore vs valence-only checks

Optional **semicore-correlated** active spaces compared to the v3 **valence-only** control at the **same geometries** (forward/backward sweeps, three initial-guess seeds). Open-source **PySCF** only; `OMP_NUM_THREADS` for parallelism.

## Active spaces and electron counts (asserted)

| Species group | Orbitals (13) | Active electrons | Notes |
|---------------|---------------|------------------|--------|
| **MnF** neutral | Mn 3s, 3p, 3d, 4s; F 2p | **20** | Mn 3s²3p⁶3d⁵4s² (15) + F 2p⁵ (5) |
| **NiF** neutral | Ni 3s, 3p, 3d, 4s; F 2p | **23** | Ni 3s²3p⁶3d⁸4s² (18) + F 2p⁵ (5) — not 20 |
| **PdF\*** | Pd 4s, 4p, 4d, 5s; F 2p | **25 − charge** | Pd 4s²4p⁶4d¹⁰5s² (20) + F 2p⁵ (5); e.g. [PdF]²⁺ → **23**, [PdF]³⁺ → **22** |

Valence-only control: v3 frontier space (9 orbitals; 12/15/13/12 electrons per species) with semicore **frozen** in NEVPT2 via the smaller CAS.

Correlation: spin-pure **CASSCF** + `fix_spin_`, then **SC-NEVPT2** (semicore orbitals in the CAS for Pd ECP 4s/4p).

## States

| Key | 2S+1 (target) | PySCF spin |
|-----|----------------|------------|
| MnF_7, MnF_5 | 7Σ⁺, 5Σ⁺ | 6, 4 |
| NiF_2, NiF_4 | 2Π/2Δ, 4Σ/4Φ (v3 labels) | 1, 3 |
| PdF_2 | 2Σ | 1 |
| PdF2+_2, PdF2+_4 | doublet, quartet | 1, 3 |
| PdF3+_1, PdF3+_3, **PdF3+_5** | singlet, triplet, **quintet** | 0, 2, 4 |

## Basis arms

- **3d (Mn, Ni):** X2C `cc-pVTZ-DK` (default `--basis`) or `x2c-TZVPall`.
- **Pd:** always both **ECP_def2** (`def2-TZVP` + ECP, no X2C) and **AE X2C** `x2c-tzvpall` with the Pd semicore labels above.

## Grids

- Default: **10 points**, linear **r_e − 0.20 Å** to **r_e + 0.25 Å** (`r_e` required per state on CLI).
- **[PdF]³⁺** singlet/triplet/quintet: **1.70–2.15 Å** (8 points) **plus 2.3 and 2.6 Å** (12 points total), independent of r_e spacing.

## Seeds

Each state × arm × mode runs **3 seeds** (ROHF level shift / `fix_spin_` / AVAS minao order). Analysis reports **semicore − valence** shifts Δr_e, Δω_e, ΔD_e, ΔE_min and **mean ± std** over seeds.

## Memory and CI

- Abort if **MemAvailable < 14 GiB** (`MNF_SEMICORE_MIN_MEM_GIB`) at each heavy point when `--run`.
- CI: `python scripts/mnf_semicore_cas20_13.py --smoke` only.

## Example commands (16 threads)

**MnF only**

```bash
export OMP_NUM_THREADS=16
python scripts/mnf_semicore_cas20_13.py --run --species MnF \
  --out data/semicore/mnf_raw.json \
  --MnF_7-re 1.8341 --MnF_5-re 1.6738
```

**NiF**

```bash
python scripts/mnf_semicore_cas20_13.py --run --species NiF \
  --out data/semicore/nif_raw.json \
  --NiF_2-re 1.75 --NiF_4-re 1.68
```

**Pd (both ECP + AE arms per species)**

```bash
python scripts/mnf_semicore_cas20_13.py --run --species PdF \
  --out data/semicore/pdf_raw.json \
  --PdF_2-re 1.91 --PdF2+_2-re 2.20 --PdF2+_4-re 2.11 \
  --PdF3+_1-re 1.90 --PdF3+_3-re 2.35 --PdF3+_5-re 2.10
```

Analysis:

```bash
python scripts/mnf_semicore_analyse.py data/semicore/pdf_raw.json --out data/semicore/pdf_summary.json
```

## Rough wall-time (16 threads, one seed)

Estimates assume ~15–45 min/point for TZ semicore CASSCF+NEVPT2 (valence ~3× faster). One **state** ≈ **grid × 2 sweeps × 2 modes × 3 seeds**:

| Job slice | Grid×2 | Points (approx) | Order-of-magnitude |
|-----------|--------|-----------------|--------------------|
| One MnF state | 10×2 | 20 points × 2 modes × 3 seeds ≈ 120 CAS | **~1–3 days** |
| NiF (2 states) | same | ≈ 240 CAS | **~2–6 days** |
| One Pd state (2 arms) | 10×2 | ×2 arms | **~2–6 days** |
| Full Pd suite (6 keys) | mixed | includes 12-pt PdF3+ grids | **~2–4 weeks** total if run serially by species |

Run with `--species` to queue jobs sequentially on one host.