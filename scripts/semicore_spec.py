"""Species definitions for fluorine semicore vs valence-only control calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Family = Literal["3d", "pd"]
NCAS_SEMICORE = 13

# --- Electron bookkeeping (asserted against AVAS and CAS nelecas) ----------------
# Orbitals: 3s(1)+3p(3)+3d(5)+4s(1)+F2p(3) = 13
# Mn neutral fragment in active space: 3s²3p⁶3d⁵4s² (15) + F 2p⁵ (5) = 20
# Ni neutral fragment in active space: 3s²3p⁶3d⁸4s² (18) + F 2p⁵ (5) = 23
# Pd semicore orbitals: 4s+4p+4d+5s+F2p = 13
# Pd neutral fragment in active space: 4s²4p⁶4d¹⁰5s² (20) + F 2p⁵ (5) = 25;
#   subtract molecular charge: nelecas = 25 - charge

LABELS_3D = ("{M} 3s", "{M} 3p", "{M} 3d", "{M} 4s", "F 2p")
LABELS_PD = ("Pd 4s", "Pd 4p", "Pd 4d", "Pd 5s", "F 2p")

ELECTRONS_3D = {"Mn": 20, "Ni": 23}
ELECTRONS_PD_BASE = 25  # charge 0

GRID_DELTA_LO = 0.20
GRID_DELTA_HI = 0.25
N_GRID_DEFAULT = 10

N_SEEDS_DEFAULT = 3


@dataclass(frozen=True)
class SpeciesSpec:
    key: str
    element: str
    charge: int
    spin: int
    family: Family
    valence_ncas: int
    valence_nelec: int
    semicore_nelec: int
    grid_kind: Literal["re_window", "pdf3p_extended"] = "re_window"

    @property
    def semicore_labels(self) -> tuple[str, ...]:
        if self.family == "3d":
            return tuple(s.format(M=self.element) for s in LABELS_3D)
        return LABELS_PD

    def grid(self, re_ang: float) -> list[float]:
        if self.grid_kind == "pdf3p_extended":
            core = [float(x) for x in np.linspace(1.70, 2.15, 8)]
            extra = [2.3, 2.6]
            return sorted(set(core + extra))
        if re_ang <= 0:
            raise ValueError(f"r_e must be positive for {self.key}, got {re_ang}")
        lo = re_ang - GRID_DELTA_LO
        hi = re_ang + GRID_DELTA_HI
        return [float(x) for x in np.linspace(lo, hi, N_GRID_DEFAULT)]


SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec("MnF_7", "Mn", 0, 6, "3d", 9, 12, ELECTRONS_3D["Mn"]),
    SpeciesSpec("MnF_5", "Mn", 0, 4, "3d", 9, 12, ELECTRONS_3D["Mn"]),
    SpeciesSpec("NiF_2", "Ni", 0, 1, "3d", 9, 15, ELECTRONS_3D["Ni"]),
    SpeciesSpec("NiF_4", "Ni", 0, 3, "3d", 9, 15, ELECTRONS_3D["Ni"]),
    SpeciesSpec("PdF_2", "Pd", 0, 1, "pd", 9, 15, ELECTRONS_PD_BASE),
    SpeciesSpec("PdF2+_2", "Pd", 2, 1, "pd", 9, 13, ELECTRONS_PD_BASE - 2),
    SpeciesSpec("PdF2+_4", "Pd", 2, 3, "pd", 9, 13, ELECTRONS_PD_BASE - 2),
    SpeciesSpec("PdF3+_1", "Pd", 3, 0, "pd", 9, 12, ELECTRONS_PD_BASE - 3, "pdf3p_extended"),
    SpeciesSpec("PdF3+_3", "Pd", 3, 2, "pd", 9, 12, ELECTRONS_PD_BASE - 3, "pdf3p_extended"),
    SpeciesSpec("PdF3+_5", "Pd", 3, 4, "pd", 9, 12, ELECTRONS_PD_BASE - 3, "pdf3p_extended"),
)

SPECIES_BY_KEY = {s.key: s for s in SPECIES}

GROUP_PREFIXES = {
    "MnF": ("MnF_7", "MnF_5"),
    "NiF": ("NiF_2", "NiF_4"),
    "PdF": tuple(k for k in SPECIES_BY_KEY if k.startswith("Pd")),
    "3d": tuple(k for k in SPECIES_BY_KEY if SPECIES_BY_KEY[k].family == "3d"),
    "Pd": tuple(k for k in SPECIES_BY_KEY if SPECIES_BY_KEY[k].family == "pd"),
}

ARMS_3D_AE = (
    ("AE_X2C_ccDK", "cc-pVTZ-DK", None, True),
    ("AE_X2C_x2cTZ", "x2c-tzvpall", None, True),
)

ARMS_PD = (
    ("ECP_def2_semicore", "def2-TZVP", "def2-TZVP", False),
    ("AE_X2C_x2cTZ", "x2c-tzvpall", None, True),
)


def arms_for_spec(spec: SpeciesSpec, basis_cli: str | None) -> tuple[tuple[str, str | dict, str | dict | None, bool], ...]:
    if spec.family == "3d":
        if basis_cli == "x2c-TZVPall":
            return (ARMS_3D_AE[1],)
        return (ARMS_3D_AE[0],)
    return ARMS_PD


def filter_species(keys: list[str] | None) -> list[SpeciesSpec]:
    if not keys:
        return list(SPECIES)
    want: set[str] = set()
    for k in keys:
        k = k.strip()
        if k in SPECIES_BY_KEY:
            want.add(k)
        elif k in GROUP_PREFIXES:
            want.update(GROUP_PREFIXES[k])
        elif k.endswith("*"):
            pref = k[:-1]
            want.update(sk for sk in SPECIES_BY_KEY if sk.startswith(pref))
    return [s for s in SPECIES if s.key in want]


def seed_hints(seed: int) -> dict:
    """Three ROHF/CAS initial-guess variants for shift statistics."""
    base = {"minao_order": ("ano", "minao"), "level_shift": 0.25, "spin_shift_delta": 0.0, "avas_threshold_bias": 0.0}
    if seed == 1:
        return {**base, "minao_order": ("minao", "ano"), "level_shift": 0.30, "spin_shift_delta": 0.08}
    if seed == 2:
        return {**base, "level_shift": 0.20, "spin_shift_delta": -0.08, "avas_threshold_bias": 0.02}
    return base


def assert_electron_table() -> dict[str, int]:
    """Runtime assertion of documented electron counts (import-time sanity)."""
    table = {}
    for s in SPECIES:
        assert s.semicore_nelec == (
            ELECTRONS_3D[s.element] if s.family == "3d" else ELECTRONS_PD_BASE - s.charge
        )
        table[s.key] = s.semicore_nelec
    return table


ELECTRON_COUNTS = assert_electron_table()