"""Tests for excited-state TD scan driver (parse, MR flags, smoke)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCAN = REPO / "scripts" / "excited_states" / "td_scan.py"
COMMON = REPO / "scripts" / "excited_states" / "td_common.py"
ANALYSE = REPO / "scripts" / "excited_states" / "td_analyse.py"


def _import_common():
    sys.path.insert(0, str(REPO / "scripts" / "excited_states"))
    import td_common as tc

    return tc


def test_default_grid_22_points():
    tc = _import_common()
    g = tc.default_grid()
    assert len(g) == 22
    assert g[0] == 1.55
    assert g[-1] == 2.60


def test_parse_grid_colon_syntax():
    tc = _import_common()
    g = tc.parse_grid("1.55:2.60:0.05")
    assert g == tc.default_grid()


def test_mr_flags_synthetic():
    tc = _import_common()
    ref = {"reference_instability": True, "bad_roots": [{"kind": "negative"}]}
    roots = [{"e_ev": 0.2, "s2_exc": 10.0}]
    flags = tc.mr_flags_for_point(ref, roots, spin=1, rpa_tda_max=0.8)
    assert flags["reference_instability"]
    assert flags["lowest_excitation_lt_0p5eV"]
    assert flags["rpa_tda_gap_gt_0p3eV"]
    assert flags["excited_s2_deviation_gt_0p5"]
    assert flags["negative_or_imaginary_root"]


def test_run_refuses_without_flag():
    proc = subprocess.run(
        [sys.executable, str(SCAN)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "dry_run" in proc.stdout


def test_dry_run_lists_grid():
    proc = subprocess.run(
        [sys.executable, str(SCAN), "--grid", "1.80:2.00:0.20"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "1.8" in proc.stdout


def test_smoke_subprocess_under_60s():
    subprocess.run(
        [sys.executable, str(SCAN), "--smoke"],
        cwd=str(REPO),
        check=True,
        timeout=55,
    )


def test_analyse_on_empty_points(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"points": {}, "spin_flip": {}}) + "\n")
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(ANALYSE), "--raw", str(raw), "--out-dir", str(out)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert (out / "td_digest.md").exists()


def test_stats_lowest():
    tc = _import_common()
    s = tc.stats_lowest([0.5, 0.7, 0.9])
    assert s["mean"] == pytest.approx(0.7)
    assert s["n"] == 3


def test_check_memory_waits_until_available():
    tc = _import_common()
    readings = [2.0, 2.0, 8.0]

    def fake_avail():
        return readings.pop(0) if readings else 8.0

    with (
        patch.object(tc, "mem_available_gib", side_effect=fake_avail),
        patch.object(tc, "min_mem_gib", return_value=6.0),
        patch.object(tc.time, "sleep") as slept,
        patch.object(tc.time, "monotonic", side_effect=[0.0, 1.0, 2.0, 3.0]),
    ):
        tc.check_memory(wait_seconds=10, max_wait_hours=1.0)
    assert slept.call_count == 2


def test_check_memory_exits_after_max_wait():
    tc = _import_common()
    with (
        patch.object(tc, "mem_available_gib", return_value=1.0),
        patch.object(tc, "min_mem_gib", return_value=6.0),
        patch.object(tc.time, "sleep"),
        patch.object(tc.time, "monotonic", side_effect=[0.0, 100.0, 200.0, 3601.0]),
        pytest.raises(SystemExit),
    ):
        tc.check_memory(wait_seconds=100, max_wait_hours=1.0)


def test_checkpoint_record_json_roundtrip(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "excited_states"))
    import td_scan as scan

    best = {
        "e_tot_ha": np.float64(-1.234),
        "guess": "minao",
        "s2_ref": np.float32(0.75),
    }
    guesses = [{"guess": "atom", "e_tot_ha": -1.2}, best]
    winning = dict(best)
    winning["guesses"] = guesses
    winning["winning_guess"] = winning.get("guess")
    record = {
        "species": "PdF_2",
        "reference": winning,
        "td": {"roots": [{"e_ev": np.float64(2.5)}]},
    }
    payload = scan._jsonable(record)
    text = json.dumps(payload)
    restored = json.loads(text)
    assert restored["species"] == "PdF_2"
    assert restored["reference"]["e_tot_ha"] == pytest.approx(-1.234)
    assert restored["reference"]["s2_ref"] == pytest.approx(0.75)
    assert restored["reference"]["winning_guess"] == "minao"
    assert restored["reference"]["guesses"][1]["guess"] == "minao"
    assert restored["td"]["roots"][0]["e_ev"] == pytest.approx(2.5)

    legacy = {"guess": "minao", "e_tot_ha": np.float64(-1.0)}
    legacy["guesses"] = [legacy]
    with pytest.raises(ValueError, match="Circular reference"):
        json.dumps(legacy)
    assert json.loads(json.dumps(scan._jsonable(legacy)))["guess"] == "minao"

    ck_path = tmp_path / "td_raw.json"
    scan.save_store(
        ck_path,
        {"points": {"PdF_2|UKS_TDA_PBE0|1.80": record}, "meta": {}, "spin_flip": {}},
    )
    loaded = json.loads(ck_path.read_text())
    assert loaded["points"]["PdF_2|UKS_TDA_PBE0|1.80"]["reference"]["winning_guess"] == "minao"


def test_help_without_importing_pyscf():
    proc = subprocess.run(
        [sys.executable, str(SCAN), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--mem-wait-seconds" in proc.stdout
    assert "--mem-wait-max-hours" in proc.stdout