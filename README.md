# Can Fluorine Form Triple Bonds? Code, data and manuscript source

Staging repository for the computational study on Mn–F multiple bonding by Julian M. Kleber (ORCID [0000-0001-5518-0932](https://orcid.org/0000-0001-5518-0932)).

**Status: private staging copy.** The files are an as-is snapshot of the working project (`fluorine-snap-1124.tgz`, 24 Nov 2026). The manuscript in `paper/` is the 22 Nov 2026 source and still carries the earlier title; the current preprint title is *Can Fluorine Form Triple Bonds? Relativistic Multireference Electronic Structure, σ/π Bond-Order Decomposition, and All-Electron vs. ECP Audit of Transition-Metal Monofluorides* (ChemRxiv). Manuscript fixes will land here through pull requests. Nothing has been changed for this release yet.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | PySCF calculation, screening and plotting scripts (`run_ab_initio.py`, `scan_highres_pes.py`, `reproduce_preprint_tables.py`, ligand and supramolecular screens, tests) |
| `results/` | Result JSON files written by the scripts (energies, bond orders, PES curves, ligand screens) |
| `figures/` | Figures used in the manuscript (PDF/PNG) |
| `paper/` | LaTeX source and compiled PDF of the manuscript |
| `preprint_handin/` | Preprint submission bundle (source package, PDF, metadata) |

Molecular geometries are defined inside the scripts (e.g. bond lengths in `reproduce_preprint_tables.py`); there is no separate coordinates directory.

## Requirements

Python 3 with `pyscf`, `numpy`, `scipy` and `matplotlib`. Multireference runs (CASSCF/NEVPT2, X2C) are CPU- and memory-intensive.

```bash
pip install pyscf numpy scipy matplotlib
python scripts/smoke_pyscf.py               # quick installation check
python scripts/reproduce_preprint_tables.py # recompute Tables 1-2 reference values
```

## Licences (proposal, pending author approval)

- Code (`scripts/`): MIT, see `LICENSE`.
- Data, figures and manuscript text (`results/`, `figures/`, `paper/`, `preprint_handin/`): CC BY 4.0, see `LICENSE-DATA`.

## Citation

Cite the ChemRxiv preprint (DOI to be added on public release).
