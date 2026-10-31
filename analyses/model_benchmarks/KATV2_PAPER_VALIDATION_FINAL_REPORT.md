# KATv2 paper-strength validation - final report (night 2)

## OVERALL_STATUS = COMPLETE

- freeze verification: EXT_V1 code files changed = 0, SOURCE_PACKAGE files changed = 0 (both must be 0)
- 21/21 validation tests PASS; all P0 analyses complete

## 1-2. EXT_V1 / SOURCE_PACKAGE zero-modification
Both verified by full-tree sha256 re-hash at night end (see above; hashes archived in 00_FREEZE_AUDIT).

## 3-4. Cell-number stability
- GSE145197: stable from **>= 100 cells/sample**; GSE223109: **>= 250** (pre-specified criterion; empirical in tested cohorts only, NOT a clinical threshold)
- full level table in 02_.../CELL_NUMBER_SENSITIVITY_SUMMARY.csv (all levels reported, none selected post hoc)

## 5. Weighted vs unweighted
- median |shift|: liver 0.65 h / dermis 0.54 h; median true AE weighted-vs-unweighted: liver 1.65 vs 1.64 h; dermis 1.33 vs 1.65 h
- no universal winner declared (direction differs across cohorts); both retained

## 6-8. Uncertainty calibration
- evidence_strength: liver rho -0.73 [-0.99, -0.14]; dermis rho -0.58 [-0.90, -0.05] (sample-level bootstrap, 2000 reps)
- the ONLY metric significantly negative in BOTH cohorts; pooled estimates reported exploratory-only
- risk-coverage: AURC below full-coverage baseline in 11/12 metric-cohort pairs; dermis median AE 1.33 h -> 0.18 h at 25% coverage (evidence_strength)
- no abstention threshold trained

## 10-12. Transcriptome concordance
- all expressed genes: amplitude Spearman liver 0.74 / dermis 0.48; R2 Spearman 0.91 / 0.28
- median circular acrophase error (true-rhythmic): liver 1.57 h (n=96), dermis 0.94 h (n=37)
- SYSTEMATIC BIAS: circular mean liver 0.99 h [0.58, 1.43], dermis 0.70 h [0.47, 0.90] - significant ~+1 h cohort-wide lag in BOTH cohorts; reported UNCORRECTED (frozen B0 real performance)
- rank preservation (liver): top-50 Jaccard 0.56, rank corr 0.91; dermis weaker (0.19)

## 13-15. Permutation controls
- 1000 sample-level permutations: observed FDR hits 8 vs null mean 0.1 (p95 1)
- positive controls: Pck1 perm-p=0.004; Lpin1 perm-p=0.002; Mat1a perm-p=0.024; Slc7a2 perm-p=0.007
- erythroid program: R2 0.030 vs matched-null median 0.093 (empirical p 0.93) - remains NEGATIVE (no recreated confound)

## 16-17. Additional cohort
- No NEW external cohort is eligible: GSE206583 is development-involved (frozen role), GSE292219 tar absent locally (3.7 GB; download forbidden tonight). Refit attempt status: see 06_ADDITIONAL_COHORTS/.

## 18-19. Comparators
- Tempo liver: TECHNICALLY_UNRESOLVED_ON_LOCAL_HARDWARE
- scRITMO: NOT_FEASIBLE_FAIRLY (needs labeled training reference; external cohorts have none - training on test data would leak labels KATv2 never sees). NOT counted as a KATv2 win.

## 20-21. Target evidence / Figure 5 boundary
- 9/18 candidates carry real DGIdb drug claims (Apoa1, Apoc3, Ttr, Fgb, Serpina1b, Gstp1, Errfi1, Tat, Fmo5); all remain NEEDS_DISEASE_DATA
- Figure 5c (liver vs dermis) stays METHOD_DEMONSTRATION until a disease+reference cohort exists (criteria formalized in 09_CLINICAL_READINESS)

## 22-23. Claim audit summary
- main-text allowed (with limitations stated): 10 claims; supplement-only: 2 ('clinical dosing windows' = NOT_SUPPORTED; 'cross-platform' = NOT_SUPPORTED)
- full matrix: 10_PAPER_CLAIMS/CLAIM_MATRIX.csv

## 24. Next disease cohort
- see 09_CLINICAL_READINESS/DISEASE_DATASET_SELECTION_CRITERIA.md (>=6 samples/condition, >=250 cells/sample, disease+reference, raw counts, annotated cell types, independent validation cohort preferred)

## Reproducibility
- global seed 20261008; bootstrap 2000; permutation 1000; all stages checkpointed under logs/
