"""Tests for excited-state TD scan driver (parse, MR flags, smoke)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
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