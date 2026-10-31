import json
from pathlib import Path

T12 = Path('/root/autodl-tmp/sckat_server_package/results/'
           'single_cell_phase_rebuild/'
           '12_KAT_V2_EXTERNAL_VALIDATION_20261008')
v = json.loads((T12 / 'TASK12_FINAL_VERDICT.json').read_text())
v["EXTERNAL_VALIDATION"] = "FAILED"
v["verdict_correction_note"] = (
    "auto-classified PARTIAL by the summary script's simplified "
    "rule; corrected at report time to FAILED per the frozen "
    "criteria ('materially harmful' exclusion from PARTIAL: "
    "GSE223109 degraded -0.348 h and its F2 matched-panel control "
    "FAILED). No model change involved.")
v["GSE223109_STATUS"] = (
    "DEGRADED (blood-contamination capture: top CPI genes "
    "Hba-a1/Hba-a2/Hbb-bs/Alas2; F2 matched-panel control FAILED)")
v["GSE145197_STATUS"] = (
    "IMPROVED (+0.472 h; top genes Pck1/Lpin1/Mat1a = genuine "
    "hepatic clock outputs; F1+F2 PASS)")
v["FINAL_KATV2_STATUS"] = (
    "CASE C: the SAFE-CPI extension fails external validation as "
    "a universal default; KATv2 REVERTS TO ANCHORED B0 (corrected "
    "B0 externally strong: 3.10/3.50 h cell AE on foreign tissues "
    "with aorta priors; beats Tempo 4.18 where comparable). "
    "SAFE-CPI remains a development-validated option with cohort "
    "dependence: deploy only when its per-cohort falsification "
    "controls (especially the F2 matched-panel admission test) "
    "pass; the trust-region gate dampened the damage by 74% "
    "(C1 -1.342 h -> C2 -0.348 h) but did not prevent confound "
    "capture on GSE223109.")
v["BOTH_HOLDOUTS_BEAT_TEMPO"] = False
v["tempo_note"] = (
    "Tempo NOT_RUN_TECHNICAL on GSE145197 (structural de-novo "
    "cycler failure, 3 documented attempts); Tempo on GSE223109 = "
    "4.182 h (both B0 3.101 and C2 3.449 beat it).")
(T12 / 'TASK12_FINAL_VERDICT.json').write_text(
    json.dumps(v, indent=2))
print("verdict corrected and frozen")
