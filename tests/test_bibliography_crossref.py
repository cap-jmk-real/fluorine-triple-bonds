"""Crossref metadata checks for every \\bibitem in paper/main.tex."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "main.tex"

# bibitem key -> (doi, year, first_author_family_prefix, title_fragment)
EXPECTED = {
    "harbison2002": ("10.1021/ja0159261", 2002, "Harbison", "dipole"),
    "schroder1999": (
        "10.1002/(SICI)1521-3773(19990115)38:1/2<137::AID-ANIE137>3.0.CO;2-M",
        1999,
        "Schro",
        "UF",
    ),
    "huber1979": ("10.1007/978-1-4757-0961-2_2", 1979, "Huber", "Diatomic"),
    "kleber2025": ("10.26434/chemrxiv-2022-xlsjn-v3", 2025, "Kleber", "fluorine"),
    "frenking2000": ("10.1021/cr980401l", 2000, "Frenking", "bonding"),
    "riedel2009": ("10.1016/j.ccr.2008.07.014", 2009, "Riedel", "oxidation"),
    "pyykko2012": ("10.1146/annurev-physchem-032511-143755", 2012, "Pyykkö", "Relativistic"),
    "brosi2016": ("10.1039/c5dt04827c", 2016, "Brosi", "manganese"),
    "devore1978": ("10.1063/1.436262", 1978, "DeVore", "MnF"),
    "sheridan2003": ("10.1016/j.cplett.2003.08.106", 2003, "Sheridan", "MnF"),
    "koukounas2004": ("10.1063/1.1738412", 2004, "Koukounas", "fluorides"),
    "seeger1977": ("10.1063/1.434318", 1977, "Seeger", "Hartree"),
    "sun2020": ("10.1063/5.0006074", 2020, "Sun", "program package"),
    "angeli2001": ("10.1063/1.1361246", 2001, "Angeli", "perturbation"),
    "balabanov2005": ("10.1063/1.1998907", 2005, "Balabanov", "transition metals"),
    "weigend2005": ("10.1039/b508541a", 2005, "Weigend", "basis"),
    "adamo1999": ("10.1063/1.478522", 1999, "Adamo", "PBE0"),
    "mayer1983": ("10.1016/0009-2614(83)80005-0", 1983, "Mayer", "bond order"),
    "krossing2004": ("10.1002/anie.200300620", 2004, "Krossing", "anions"),
    "court1973": ("10.1039/DT9730001995", 1973, "Court", "nickel"),
    "greb2018": ("10.1002/chem.201802698", 2018, "Greb", "superacids"),
    "peterson2007": (
        "10.1063/1.2647019",
        2007,
        "Peterson",
        "pseudopotentials",
    ),
    "andrae1990": ("10.1007/BF01114537", 1990, "Andrae", "pseudopotentials"),
    "pollak2017": ("10.1021/acs.jctc.7b00593", 2017, "Pollak", "Relativistic"),
}


def _fetch(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='')}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "fluorine-triple-bonds-bib/1.0 (mailto:agent@cursor.local)"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode())["message"]


def test_every_bibitem_listed_in_tex():
    tex = PAPER.read_text()
    keys = set(re.findall(r"\\bibitem\{(\w+)\}", tex))
    assert keys == set(EXPECTED.keys()), f"missing/extra bibitems: {keys ^ set(EXPECTED.keys())}"


def test_crossref_metadata_matches_expectations():
    import time

    for i, (key, (doi, year, author_prefix, title_frag)) in enumerate(EXPECTED.items()):
        if i:
            time.sleep(0.35)
        msg = _fetch(doi)
        assert msg["DOI"].lower() == doi.lower(), key
        pub_year = msg.get("published-print", msg.get("published-online", msg.get("issued", {})))
        yr = pub_year.get("date-parts", [[None]])[0][0]
        if yr is not None:
            assert int(yr) == year, f"{key}: year {yr} != {year}"
        authors = msg.get("author", [])
        assert authors, key
        fam = authors[0].get("family", "")
        nf, na = _ascii_fold(fam), _ascii_fold(author_prefix)
        assert na in nf or nf in na, f"{key}: first author {fam!r}"
        titles = msg.get("title", [])
        title_norm = re.sub(r"<[^>]+>", "", titles[0] if titles else "").lower()
        title_norm = title_norm.replace("\n", " ")
        assert titles and title_frag.lower() in title_norm, f"{key}: title {titles!r}"