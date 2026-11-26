"""Shared TD vertical-excitation scan helpers (geometry/basis + MR diagnostics)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HA_EV = 27.211386245988
_HARTREE2EV: float | None = None


def __getattr__(name: str) -> Any:
    if name == "HARTREE2EV":
        global _HARTREE2EV
        if _HARTREE2EV is None:
            from pyscf.data import nist

            _HARTREE2EV = float(nist.HARTREE2EV)
        return _HARTREE2EV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

TD_SPECS: list[tuple[str, str, int, int]] = [
    ("PdF_2", "Pd", 0, 1),
    ("PdF2+_2", "Pd", 2, 1),
    ("PdF2+_4", "Pd", 2, 3),
    ("PdF3+_1", "Pd", 3, 0),
    ("PdF3+_3", "Pd", 3, 2),
    ("PdF3+_5", "Pd", 3, 4),
    ("MnF_7", "Mn", 0, 6),
    ("MnF_5", "Mn", 0, 4),
    ("NiF_2", "Ni", 0, 1),
    ("NiF_4", "Ni", 0, 3),
]

SPIN_FLIP_JOBS: list[dict[str, Any]] = [
    {"id": "PdF2+_4_to_doublet", "species": "PdF2+_4", "element": "Pd", "charge": 2, "ref_spin": 3, "target_2S": 1},
    {"id": "PdF3+_5_to_triplet", "species": "PdF3+_5", "element": "Pd", "charge": 3, "ref_spin": 4, "target_2S": 3},
    {"id": "PdF3+_5_to_singlet", "species": "PdF3+_5", "element": "Pd", "charge": 3, "ref_spin": 4, "target_2S": 1},
    {"id": "MnF_7_to_quintet", "species": "MnF_7", "element": "Mn", "charge": 0, "ref_spin": 6, "target_2S": 5},
]

ARMS: dict[str, tuple[str, str | dict, str | dict | None, bool]] = {
    "ECP_def2": (
        "ECP_def2",
        {"Pd": "def2-TZVP", "Mn": "def2-TZVP", "Ni": "def2-TZVP", "F": "def2-TZVP"},
        {"Pd": "def2-TZVP"},
        False,
    ),
    "AE_X2C_ccDK": ("AE_X2C_ccDK", "cc-pVTZ-DK", None, True),
    "AE_X2C_x2cTZ": (
        "AE_X2C_x2cTZ",
        {"Pd": "x2c-tzvpall", "Mn": "x2c-tzvpall", "Ni": "x2c-tzvpall", "F": "cc-pVTZ-DK"},
        None,
        True,
    ),
}

SMOKE_BASIS: tuple[str | dict, str | dict | None, bool] = (
    {"Pd": "def2-SVP", "Mn": "def2-SVP", "Ni": "def2-SVP", "F": "def2-SVP"},
    {"Pd": "def2-SVP"},
    False,
)

TD_METHODS: list[tuple[str, str, str | None, str]] = [
    ("UKS_TDDFT_PBE0", "UKS", "pbe0", "RPA"),
    ("UKS_TDA_PBE0", "UKS", "pbe0", "TDA"),
    ("UKS_TDDFT_CAMB3LYP", "UKS", "cam-b3lyp", "RPA"),
    ("UKS_TDA_CAMB3LYP", "UKS", "cam-b3lyp", "TDA"),
    ("UHF_TDDFT", "UHF", None, "RPA"),
    ("UHF_TDA", "UHF", None, "TDA"),
]

MR_THRESH = {
    "lowest_exc_ev": 0.5,
    "rpa_tda_ev": 0.3,
    "s2_deviation": 0.5,
}


def default_grid(start: float = 1.55, stop: float = 2.60, step: float = 0.05) -> list[float]:
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(n)]


def parse_grid(spec: str | None) -> list[float]:
    if not spec:
        return default_grid()
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        return default_grid(lo, hi, step)
    return [float(x) for x in spec.split(",")]


def min_mem_gib() -> float:
    return float(os.environ.get("TD_MIN_MEM_GIB", "6"))


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def check_memory(
    *,
    wait_seconds: float = 300,
    max_wait_hours: float = 24,
) -> None:
    """Wait until MemAvailable meets TD_MIN_MEM_GIB instead of exiting immediately."""
    need = min_mem_gib()
    deadline = time.monotonic() + max(0.0, max_wait_hours) * 3600.0
    while True:
        avail = mem_available_gib()
        if avail is None or avail >= need:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(
                f"MemAvailable {avail:.2f} GiB < required {need:.2f} GiB "
                f"after waiting up to {max_wait_hours:.2f} h "
                f"(set TD_MIN_MEM_GIB to override)"
            )
        sleep_for = min(wait_seconds, remaining)
        logger.warning(
            "MemAvailable %.2f GiB < required %.2f GiB; sleeping %.0f s before recheck",
            avail,
            need,
            sleep_for,
        )
        time.sleep(sleep_for)


def max_memory_mb() -> int:
    return int(os.environ.get("PYSCF_MAX_MEMORY", "4000"))


def target_s2(spin: int) -> float:
    s = spin * 0.5
    return s * (s + 1.0)


def ml_class(lbl: tuple) -> str:
    m = str(lbl[3]).strip()
    if m in ("", "z", "z^2", "z^3", "0"):
        return "sigma"
    if m in ("x", "y", "xz", "yz", "xz^2", "yz^2", "+1", "-1"):
        return "pi"
    return "delta"


def lambda_label_from_ao(mol, vec: np.ndarray) -> str:
    """C∞v Λ label from dominant projection on cylindrical axis (bond = z)."""
    w = vec * (mol.intor("int1e_ovlp") @ vec)
    lab = mol.ao_labels(fmt=False)
    m_sym: dict[str, float] = {"sigma": 0.0, "pi": 0.0, "delta": 0.0, "phi": 0.0}
    for i, val in enumerate(w):
        cls = ml_class(lab[i])
        if cls == "sigma":
            m_sym["sigma"] += abs(val)
        elif cls == "pi":
            m_sym["pi"] += abs(val)
        elif cls == "delta":
            m_sym["delta"] += abs(val)
        else:
            m_sym["phi"] += abs(val)
    best = max(m_sym, key=m_sym.get)
    return {"sigma": "Sigma", "pi": "Pi", "delta": "Delta", "phi": "Phi"}[best]


def ao_orbital_tag(mol, vec: np.ndarray) -> str:
    w = vec * (mol.intor("int1e_ovlp") @ vec)
    lab = mol.ao_labels(fmt=False)
    i = int(np.argmax(np.abs(w)))
    atom, sym, typ = lab[i][0], lab[i][2], lab[i][1]
    metal_idx = 0
    if atom == metal_idx:
        if typ.endswith("d"):
            return "M_d"
        if typ.endswith("p"):
            return "M_p"
        if typ.endswith("s"):
            return "M_s"
        return "M_other"
    if typ.endswith("p"):
        return "F_2p"
    if typ.endswith("s"):
        return "F_s"
    return "F_other"


def transition_character(mol, mf, t1a: np.ndarray, t1b: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def _tops(t1: np.ndarray, spin: str) -> list[dict[str, Any]]:
        if t1.size == 0:
            return []
        mo = mf.mo_coeff[0 if spin == "alpha" else 1]
        occ = mo[:, mf.mo_occ[0 if spin == "alpha" else 1] > 0]
        vir = mo[:, mf.mo_occ[0 if spin == "alpha" else 1] == 0]
        flat = np.abs(t1).ravel()
        order = np.argsort(-flat)[:5]
        rows = []
        for k in order:
            if flat[k] < 1e-5:
                continue
            i, a = np.unravel_index(k, t1.shape)
            vo = occ[:, i]
            vv = vir[:, a]
            ti, ta = ao_orbital_tag(mol, vo), ao_orbital_tag(mol, vv)
            sym = f"{ao_orbital_tag(mol, vo).split('_')[0]}→{ao_orbital_tag(mol, vv).split('_')[0]}"
            label = classify_lmct(ti, ta, sym)
            rows.append(
                {
                    "spin": spin,
                    "occ": int(i),
                    "vir": int(a),
                    "amplitude": float(t1[i, a]),
                    "occ_tag": ti,
                    "vir_tag": ta,
                    "symmetry": sym,
                    "character": label,
                }
            )
        return rows

    out.extend(_tops(t1a, "alpha"))
    out.extend(_tops(t1b, "beta"))
    out.sort(key=lambda r: -abs(r["amplitude"]))
    return out[:6]


def classify_lmct(ti: str, ta: str, sym: str) -> str:
    if "M_d" in ti and "M_d" in ta:
        return "d→d"
    if "F_2p" in ti and "M_d" in ta:
        return "F2p→M_d LMCT"
    if "M_d" in ti and "F_2p" in ta:
        return "M_d→F2p MLCT"
    if sym.startswith("sigma"):
        return "σ→σ*"
    if sym.startswith("pi"):
        return "π→π*"
    if "delta" in sym:
        return "δ→δ*"
    return sym


def nto_character(mol, mf, td, state: int) -> list[dict[str, Any]]:
    try:
        (wa, wb), (Ca, Cb) = td.get_nto(state + 1)
    except Exception:
        return []
    chars = []
    for channel, w, C in (("alpha", wa, Ca), ("beta", wb, Cb)):
        occ, vir = np.hsplit(C, [len(w)//2] if C.shape[1] > len(w) else [C.shape[1] // 2])
        # PySCF returns (weights), (occ_nto, vir_nto stacked)
        occ = C[:, : max(1, C.shape[1] // 2)]
        vir = C[:, max(1, C.shape[1] // 2) :]
        for tag, block, wt in (("occ", occ, w[: len(w) // 2]), ("vir", vir, w[len(w) // 2 :])):
            if len(wt) == 0:
                continue
            k = int(np.argmax(wt))
            v = block[:, k]
            chars.append(
                {
                    "channel": channel,
                    "role": tag,
                    "weight": float(wt[k]),
                    "tag": ao_orbital_tag(mol, v),
                    "lambda": lambda_label_from_ao(mol, v),
                }
            )
    return chars


def estimate_excited_s2(mf, t1a: np.ndarray, t1b: np.ndarray) -> float:
    s2_ref, _ = mf.spin_square()
    na = float(np.linalg.norm(t1a) ** 2)
    nb = float(np.linalg.norm(t1b) ** 2)
    tot = na + nb + 1e-30
    mix = min(na, nb) / tot
    return float(s2_ref + 2.0 * mix)


def bad_roots(energies_ev: np.ndarray) -> list[dict[str, Any]]:
    bad: list[dict[str, Any]] = []
    for i, ei in enumerate(energies_ev):
        if np.iscomplexobj(ei):
            bad.append({"root": i, "kind": "imaginary", "e_ev": complex(ei).__str__()})
        elif float(np.real(ei)) < -1e-4:
            bad.append({"root": i, "kind": "negative", "e_ev": float(np.real(ei))})
        elif not np.isfinite(ei):
            bad.append({"root": i, "kind": "non_converged", "e_ev": None})
    return bad


def mr_flags_for_point(
    ref: dict[str, Any],
    roots: list[dict[str, Any]],
    *,
    spin: int,
    rpa_tda_max: float | None,
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    tgt = target_s2(spin)
    if ref.get("reference_instability"):
        flags["reference_instability"] = True
    lowest = roots[0]["e_ev"] if roots else None
    if lowest is not None and lowest < MR_THRESH["lowest_exc_ev"]:
        flags["lowest_excitation_lt_0p5eV"] = True
    if rpa_tda_max is not None and rpa_tda_max > MR_THRESH["rpa_tda_ev"]:
        flags["rpa_tda_gap_gt_0p3eV"] = True
    for rt in roots:
        s2 = rt.get("s2_exc")
        if s2 is not None and abs(s2 - tgt) > MR_THRESH["s2_deviation"]:
            flags["excited_s2_deviation_gt_0p5"] = True
            break
    if ref.get("bad_roots"):
        flags["negative_or_imaginary_root"] = True
    return flags


def stats_lowest(lowest_vals: list[float]) -> dict[str, float | None]:
    if not lowest_vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(lowest_vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(len(arr))}