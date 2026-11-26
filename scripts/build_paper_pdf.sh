#!/usr/bin/env bash
# Build paper/main.pdf with halt-on-error (for CI and local verification).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper"
pdflatex -halt-on-error -interaction=nonstopmode main.tex >/dev/null
pdflatex -halt-on-error -interaction=nonstopmode main.tex >/dev/null
test -f main.pdf
echo "OK: paper/main.pdf"