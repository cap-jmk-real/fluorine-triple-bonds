"""Repository root and standard output paths for fluorine-triple-bonds scripts."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
PAPER_DIR = REPO_ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"