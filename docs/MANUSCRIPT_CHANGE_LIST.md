# Fluorine manuscript change list (2026-11-26, NOT applied; to be executed by a Cursor cloud agent)
Repo github.com/cap-jmk-real/fluorine-triple-bonds, branch cursor/publication-evidence-trace-d0c6 (PR #2 draft), base head 3987ab1d.
Files: paper/main.tex, paper/generated/macros.tex, scripts/plot_fluorine_results.py, scripts/scan_highres_pes.py, data JSON ab_initio_results.json, tests/test_manuscript_traceability.py, figures/fig2_ae_vs_ecp_pdf.pdf, figures/fig1_*.pdf, figures/fig3_highres_pes_barriers.pdf.
Data on branch data/pd-matched-v3: data/pd_matched_v3/{corrected_curves.json, corrected_summary.csv, legacy_state_audit.csv, w1/state_check_raw.json, w2/state_check_raw.json, v3-shift2/state_check_raw.json (spin-penalty rerun PdF3+_1 AE_X2C_x2cTZ, FIX_SPIN_SHIFT=2.0)}; scripts scripts/pd_matched_v3/{fluorine_state_check.py, fluorine_state_check_analyse.py}; notes docs/CHEMISTRY_NOTES.md.

## A. Pd section (claims reversal -> retract)
1. Abstract L23: delete "unmatched X2C ... qualitative binding reversals on [PdF]2+ and [PdF]3+". Replace with: "In a matched 2x2 (core x basis family) audit with spin-pure AVAS-CASCI(12-15,9)/SC-NEVPT2, all four protocols agree qualitatively: PdF (2Sigma+) and quartet [PdF]2+ are bound, [PdF]3+ is repulsive; earlier apparent reversals were artefacts of spin-uncontrolled CASSCF."
2. Methods L32/L34: (a) replace SCF settings "max_cycle 150-200, level_shift 0.1-0.2" by the coded values (max_cycle 250, level_shift 0.25, DIIS space 12, Newton fallback) for the legacy part; (b) add the Pd protocol: X2C ROHF with internal stability check + density following, AVAS (Pd 4d, Pd 5s, F 2p; threshold 0.2; minao 'ano' for AE, 'minao' for ECP), CASCI 3 roots with fix_spin_(shift 0.5, ss target), SC-NEVPT2 per root, lowest spin-pure root envelope, grid 1.55-3.20 + 5.00 A, fwd/bwd sweeps; basis/ECP: Pd cc-pVTZ-PP+ECP28MDF / cc-pVTZ-DK / x2c-TZVPall / def2-TZVP(ECP28), F cc-pVTZ / cc-pVTZ-DK / x2c-TZVPall / def2-TZVP (F all-electron; no F ECP exists, PySCF warning harmless). Cite Peterson 2007, Andrae 1990, Pollak & Weigend 2017.
3. L127-128 text: delete "Even neutral PdF differs by 0.108 A ... 29% in BO" (legacy AE 2.025 A/1119 cm-1 was an ill-defined active space). New statement: core treatment changes r_e(PdF) by <=0.011 A, basis family by <=0.02 A.
4. Table 6 tab:pd_ecp (L130-152, literal numbers): replace entirely by macro-driven matched table (values from corrected_summary.csv, lowest spin-pure root NEVPT2):
   PdF 2Sigma+: ECP_ccPP 1.912 A/589 cm-1/BO 0.60/q(F) -0.37; AE_ccDK 1.900/584/0.61/-0.36; AE_x2cTZ 1.914/562/0.45/-0.50; ECP_def2 1.920/571/0.40/-0.54.
   [PdF]2+ 4Sigma: ECP_ccPP 2.125/311/0.27/+0.33; AE_ccDK 2.108/323/0.27/+0.34; AE_x2cTZ 2.149/290/0.23/+0.25; ECP_def2 2.144/287/0.22/+0.24 (bound, shallow; well 1-2 eV below 5 A point; do NOT print the CSV De_vs_5A 6-18 eV).
   [PdF]2+ 2Sigma: 0.09-0.20 eV above quartet, multi-state crossing, no single-state minimum -> report "no well-defined minimum".
   [PdF]3+ (singlet and triplet): repulsive in all four protocols (E drops 7-8 eV from 1.55 A to 5 A; q(F)->+1 beyond ~2.2 A); CSV "bound" rows with omega 1200-7800 cm-1 are fit artefacts -> report "repulsive / no minimum".
   Footnote: 5 A asymptotes differ (AE Pd3+ + F vs ECP Pd2+ + F+), so no De is reported for [PdF]3+.
5. Fig. 2 fig2_ae_vs_ecp_pdf.pdf (scripts/plot_fluorine_results.py, key PdF_AE_vs_ECP in ab_initio_results.json): regenerate from corrected_curves.json as 4-protocol PdF + [PdF]2+ quartet curves; new caption.
6. Fig. 1b PdF2+_4 AE bar (plot_fluorine_results.py ~L74): replace with v3 value or drop.
7. Fig. 3d (scan_highres_pes.py, RKS-PBE0 singlet [PdF]3+): restricted singlet is not the ground state (triplet lower by 0.13-0.25 eV at NEVPT2); either remove panel d or redo UKS triplet and label as DFT-only.
8. Conclusion L249: remove the protocol-reversal sentence; state matched protocols agree; legacy curves were on wrong/mixed states.
9. Data availability L252: add the v3 data paths/DOI-archived bundle and job scripts.
10. Add a short "Correction of earlier Pd results" paragraph listing legacy wrong-state values: PdF AE 2.025 A/1119 cm-1, ECP 1.917/476; [PdF]2+ "AE bound 1.838 A/481 cm-1 pi-BO 0.401, ECP unbound"; [PdF]3+ "ECP bound at 1.976 A"; singlet/triplet [PdF]3+ energies identical to <1e-6 Eh in legacy (spin not controlled).

## B. Other tables/literals
11. tab:superacid (L207ff, L214-221): untraced literals r_e 1.7595, 1.7593, 1.6358, 1.8267; BO 1.286 ([WF5]+), 1.292/1.404/1.617 (F-VH3); "+3.92 eV" -> must be macro-driven from a results JSON or removed; add to test_manuscript_traceability.py.
12. MnF / NiF / MnF3+ tables (tab:main_mnf, tab:forensic, tab:cations): add <S^2> and state-label column, and note that spin control was applied (or rerun with the same AVAS/fix_spin protocol) - same failure mode as legacy Pd.

## C. References (L258-278): 17 wrong DOIs, verified via the Crossref API and doi.org on 2026-11-26
harbison2002 10.1021/ja0172336 -> 10.1021/ja0159261
schroder1999 -> 10.1002/(SICI)1521-3773(19990115)38:1/2<137::AID-ANIE137>3.0.CO;2-M ; also fix authors (Schröder, Diefenbach, Klapötke, Schwarz) and pages 137-140
huber1979 10.18434/T42P4B (does not resolve) -> book 10.1007/978-1-4757-0961-2_2 (Springer 1979); keep NIST WebBook as URL without DOI
pyykko2012 ...-143757 -> 10.1146/annurev-physchem-032511-143755
brosi2016 10.1039/C5DT04695A -> 10.1039/c5dt04827c ; title -> "A combined quantum-chemical and matrix-isolation study on molecular manganese fluorides"
devore1978 10.1063/1.436028 -> 10.1063/1.436262
sheridan2003 10.1016/S0009-2614(03)01463-6 -> 10.1016/j.cplett.2003.08.106
koukounas2004 10.1063/1.1737442 -> 10.1063/1.1738412 ; title -> "Ab initio investigation of the ground and low-lying states of the diatomic fluorides TiF, VF, CrF, and MnF"
seeger1977 10.1063/1.434300 -> 10.1063/1.434318
sun2020 10.1063/1.5145178 -> 10.1063/5.0006074
angeli2001 10.1063/1.1376776 -> 10.1063/1.1361246
balabanov2005 10.1063/1.1947430 -> 10.1063/1.1998907
weigend2005 10.1039/B508623A -> 10.1039/b508541a
adamo1999 10.1063/1.478452 -> 10.1063/1.478522
mayer1983 10.1016/0009-2614(83)85093-3 -> 10.1016/0009-2614(83)80005-0
krossing2004 10.1002/anie.200303516 -> 10.1002/anie.200300620
greb2018 10.1002/chem.201802293 -> 10.1002/chem.201802698
Correct already: kleber2025, frenking2000, riedel2009, court1973.
ADD: Peterson, Figgen, Dolg, Stoll, JCP 126, 124101 (2007) 10.1063/1.2647019; Andrae et al., Theor. Chim. Acta 77, 123-141 (1990) 10.1007/BF01114537; Pollak & Weigend, JCTC 13, 3696-3705 (2017) 10.1021/acs.jctc.7b00593.
Add a DOI-resolution check (Crossref) to tests.
