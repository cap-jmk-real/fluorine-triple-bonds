"""Fast tests for semicore driver (parse, electron counts, AVAS sizing)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mnf_semicore_cas20_13.py"


def _import_driver():
    sys.path.insert(0, str(REPO / "scripts"))
    import mnf_semicore_cas20_13 as drv

    return drv


def test_electron_counts_documented():
    sys.path.insert(0, str(REPO / "scripts"))
    from semicore_spec import ELECTRON_COUNTS, SPECIES_BY_KEY

    assert ELECTRON_COUNTS["MnF_7"] == 20
    assert ELECTRON_COUNTS["NiF_2"] == 23
    assert ELECTRON_COUNTS["PdF_2"] == 25
    assert ELECTRON_COUNTS["PdF2+_4"] == 23
    assert ELECTRON_COUNTS["PdF3+_5"] == 22
    assert SPECIES_BY_KEY["NiF_2"].semicore_nelec == 23


def test_semicore_grid_ten_points():
    drv = _import_driver()
    from semicore_spec import SPECIES_BY_KEY

    g = SPECIES_BY_KEY["MnF_7"].grid(1.834)
    assert len(g) == 10
    assert abs(g[0] - 1.634) < 1e-9
    assert abs(g[-1] - 2.084) < 1e-9


def test_pdf3p_extended_grid():
    from semicore_spec import SPECIES_BY_KEY

    g = SPECIES_BY_KEY["PdF3+_3"].grid(2.0)
    assert len(g) == 10
    assert min(g) >= 1.70 - 1e-9
    assert 2.3 in g and 2.6 in g


def test_filter_species_groups():
    from semicore_spec import filter_species

    mnf = filter_species(["MnF"])
    assert {s.key for s in mnf} == {"MnF_7", "MnF_5"}


def test_run_refuses_without_re():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "requires" in proc.stderr + proc.stdout or "Refusing" in proc.stderr + proc.stdout


def test_dry_run_parses_re_and_grid():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--species",
            "MnF",
            "--MnF_7-re",
            "1.8341",
            "--MnF_5-re",
            "1.6738",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert '"dry_run": true' in proc.stdout
    assert '"MnF_7": 20' in proc.stdout


def test_avas_mo_accepts_exact_space():
    drv = _import_driver()
    from semicore_spec import SPECIES_BY_KEY, seed_hints

    spec = SPECIES_BY_KEY["MnF_7"]
    fake_mo = np.zeros((10, 10))
    fake_mf = object()

    with mock.patch.object(drv.avas, "kernel", side_effect=[(12, 19, fake_mo), (13, 20, fake_mo)]):
        mo, minao, thr = drv.avas_mo(fake_mf, spec.semicore_labels, 13, 20, hints=seed_hints(0))
    assert mo is fake_mo
    assert minao == "ano"


def test_avas_mo_raises_on_wrong_space():
    drv = _import_driver()
    from semicore_spec import SPECIES_BY_KEY

    spec = SPECIES_BY_KEY["NiF_2"]
    fake_mo = np.zeros((5, 5))
    fake_mf = object()

    with mock.patch.object(drv.avas, "kernel", return_value=(13, 20, fake_mo)):
        with pytest.raises(RuntimeError, match="exactly 13 orbitals and 23"):
            drv.avas_mo(fake_mf, spec.semicore_labels, 13, 23, hints={"minao_order": ("ano",)})


def test_smoke_subprocess_under_60s():
    subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke"],
        cwd=str(REPO),
        check=True,
        timeout=55,
    )