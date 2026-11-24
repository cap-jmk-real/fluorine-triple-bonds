# Can Fluorine Form Triple Bonds? Code, data and manuscript source

Staging repository for the computational study on Mn–F multiple bonding by Julian M. Kleber (ORCID [0000-0001-5518-0932](https://orcid.org/0000-0001-5518-0932)).

**Status: publication-evidence hardening** (traceable JSON, macro-generated headline numbers, CI drift checks). Manuscript: *Fluorine Multiple Bonding in Monofluoride Cations: A Relativistic All-Electron Audit* (journal-neutral; ChemRxiv preprint DOI [10.26434/chemrxiv-2022-xlsjn-v3](https://doi.org/10.26434/chemrxiv-2022-xlsjn-v3)).

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
pip install -r requirements.txt
python scripts/smoke_pyscf.py                    # quick installation check
python scripts/reproduce_preprint_tables.py      # forensic Table 1 + forensic_mnf_reproduction.json
python scripts/extreme_fluorine_bo.py            # extreme_bo_results.json (resumes partial files)
python scripts/design_fluorine_triple_bonds.py   # triple_bond_design_screen.json
python scripts/emit_manuscript_macros.py       # paper/generated/macros.tex + manifest
pytest tests/test_manuscript_traceability.py     # drift check (CI runs this)
```

## Licences (proposal, pending author approval)

- Code (`scripts/`): MIT, see `LICENSE`.
- Data, figures and manuscript text (`results/`, `figures/`, `paper/`, `preprint_handin/`): CC BY 4.0, see `LICENSE-DATA`.

## Citation

Cite the ChemRxiv preprint (DOI to be added on public release).
