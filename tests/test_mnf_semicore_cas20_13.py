"""Fast tests for MnF semicore CAS(20,13) driver (parse + AVAS sizing; no TZ PES)."""
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


def test_semicore_grid_ten_points():
    drv = _import_driver()
    g = drv.semicore_grid(1.834)
    assert len(g) == 10
    assert abs(g[0] - 1.634) < 1e-9
    assert abs(g[-1] - 2.084) < 1e-9


def test_run_refuses_without_re():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "requires r_e" in proc.stderr + proc.stdout or "Refusing" in proc.stderr + proc.stdout


def test_dry_run_parses_re_and_grid():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert "1.8341" in proc.stdout


def test_avas_semicore_mo_accepts_exact_13_20():
    drv = _import_driver()
    fake_mo = np.zeros((10, 10))
    fake_mf = object()

    with mock.patch.object(drv.avas, "kernel", side_effect=[(12, 19, fake_mo), (13, 20, fake_mo)]):
        mo, minao, thr = drv.avas_semicore_mo(fake_mf, thresholds=(0.1, 0.2))
    assert mo is fake_mo
    assert thr == 0.2
    assert minao == "ano"


def test_avas_semicore_mo_raises_on_wrong_space():
    drv = _import_driver()
    fake_mo = np.zeros((5, 5))
    fake_mf = object()

    with mock.patch.object(drv.avas, "kernel", return_value=(12, 18, fake_mo)):
        with pytest.raises(RuntimeError, match="exactly 13 orbitals"):
            drv.avas_semicore_mo(fake_mf, thresholds=(0.5,))


def test_smoke_subprocess_under_60s():
    subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke"],
        cwd=str(REPO),
        check=True,
        timeout=55,
    )