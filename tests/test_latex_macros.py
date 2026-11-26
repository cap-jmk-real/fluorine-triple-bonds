"""LaTeX macro integrity and PDF build."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_no_duplicate_macro_definitions():
    macros = (REPO / "paper" / "generated" / "macros.tex").read_text()
    names = re.findall(r"\\newcommand\{\\(\w+)\}", macros)
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"Duplicate \\newcommand names: {sorted(dupes)}"


def test_paper_pdf_builds_halt_on_error():
    subprocess.run(
        ["bash", str(REPO / "scripts" / "build_paper_pdf.sh")],
        cwd=str(REPO),
        check=True,
    )