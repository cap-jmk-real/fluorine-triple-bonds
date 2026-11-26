"""Fast smoke test for 3d Mn/Ni state-check driver (tiny basis, one point)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "pd_matched_v3" / "fluorine_state_check_3d_mnf_nif.py"


def test_3d_state_check_test_mode_exits_zero():
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(REPO),
            str(REPO / "tests" / "_scratch_mnf_nif_smoke"),
            "--mode",
            "test",
        ],
        cwd=str(REPO),
        check=True,
        timeout=600,
    )