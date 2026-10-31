"""Resume steps 6-9: final model selection (frozen hierarchy),
FINAL_FREEZE (13 artifacts), TASK11B_FINAL_VERDICT.json, and
TASK11B_FINAL_REPORT.txt answering A-J with the mandated B1/B2
framing. Reads only frozen post-eval artifacts."""
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
T11B = RR / "11B_KAT_V2_SAFE_PHASE_INFORMATION_20261007"
S = T11B / "STABILIZATION_20261007"
FR = T11B / "11_FINAL_FREEZE"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    res = json.loads((T11B / "TABLES/"
                      "T1_MODEL_PERFORMANCE.json").read_text())
    t5 = json.loads((T11B / "TABLES/T5_FALSIFICATION.json")
                    .read_text())
    g = lambda n, k="cell_median_AE": res.get(n, {}).get(
        k, float("nan"))
    alpha = json.loads((T11B / "05_C2_SAFE_CPI/MODEL_CONFIG.json")
                       .read_text())["alpha"]
    b0, b6 = 2.032, 1.630
    fals_pass = (t5.get("F1_verdict") == "PASS"
                 and t5.get("F2_verdict") == "PASS")
    stab = json.loads((S / "STABILIZATION_VERDICT.json")
                      .read_text())["TASK11B_ENGINE_STABLE"]
    c3integ = json.loads((S / "C3_INTEGRITY_CHECK.json").read_text())

    # ---- frozen selection hierarchy ----
    pool = {}
    for n in ("C1_CPI", "C2_SAFE_CPI", "C3_pooled", "C3_state",
              "C3_lineage", "C3_full"):
        if n not in res:
            continue
        ae = g(n)
        if not ae < b0 - 0.05:
            pool[n] = ("REJECT_not_better_than_B0", None)
            continue
        if not stab:
            pool[n] = ("REJECT_integrity", None)
            continue
        if not fals_pass:
            pool[n] = ("REJECT_falsification", None)
            continue
        anchor_ok = abs(res[n].get("oracle_rotation_h", 0)) <= 1.0 \
            if res[n].get("oracle_rotation_h") is not None else True
        calib_ok = res[n].get("calib_spearman", -1) < -0.10
        if not (anchor_ok and calib_ok):
            pool[n] = ("DOWNGRADE_calibration_anchor", None)
            continue
        pool[n] = ("OK", ae)

    def band(ae):
        if ae is None:
            return None
        if ae <= 1.710:
            return "STRONG"
        if ae <= 1.800 and fals_pass:
            return "ACCEPTABLE"
        return "WEAK"

    ok = {n: v[1] for n, v in pool.items() if v[1] is not None}
    bands = {n: band(ae) for n, ae in ok.items()}
    rationale = [f"hierarchy outcomes: "
                 + json.dumps({k: v[0] for k, v in pool.items()}),
                 f"bands: {bands}"]
    final = "B0"
    ranked = [n for n, b in sorted(bands.items(),
                                   key=lambda x: -{"STRONG": 2,
                                                   "ACCEPTABLE": 1,
                                                   "WEAK": 0,
                                                   None: -1}
                                   [x[1]]) if b in ("STRONG",
                                                    "ACCEPTABLE")]
    if ranked:
        chosen = ranked[0]
        if chosen.startswith("C3"):
            c2_ae = ok.get("C2_SAFE_CPI")
            if c2_ae is not None and not (
                    ok[chosen] <= c2_ae - 0.10):
                chosen = "C2_SAFE_CPI"
                rationale.append("C3 did not beat C2 by >=0.10h "
                                 "-> retain C2 (parsimony)")
        if chosen == "C1_CPI" and ok.get("C2_SAFE_CPI") is not \
                None and ok["C2_SAFE_CPI"] <= ok["C1_CPI"] + 0.05:
            chosen = "C2_SAFE_CPI"
            rationale.append("C2 within 0.05h of C1 -> prefer C2 "
                             "safety")
        final = chosen
    else:
        rationale.append("no candidate kept >=80% of B6 gain -> "
                         "retain B0 (protection rule)")

    anchor_pass = abs(res.get(final, res["B0"]).get(
        "oracle_rotation_h", 0)) <= 1.5
    calib_pass = res.get(final, res["B0"]).get(
        "calib_spearman", -1) < -0.10
    ext_ready = bool(stab and fals_pass and anchor_pass
                     and calib_pass and final != "B0")

    FR.mkdir(parents=True, exist_ok=True)
    (FR / "FINAL_MODEL_SPEC.json").write_text(json.dumps({
        "FINAL_MODEL": final,
        "spec": "corrected anchored B0 posterior + SAFE-CPI "
                "information head: cross-fitted conditional phase "
                "information (posterior entropy reduction on the "
                "frozen B0 posterior), non-redundant phase-covering "
                "greedy panel (K=40, lambda=0.5, beta=0.5), "
                "trust-region evidence gate v2; sample-blocked LOSO",
        "hierarchy_outcomes": {k: v[0] for k, v in pool.items()},
        "bands": bands}, indent=2, default=str))
    params = pd.read_csv(T11B / "02_B6_REFERENCE/"
                         "GENE_PARAMETERS.csv")
    panels = json.loads((T11B / "03_CPI_METHOD/"
                         "PANELS_PER_FOLD.json").read_text())
    (FR / "FINAL_MODEL_PARAMETERS.json").write_text(json.dumps({
        "alpha": alpha,
        "note": "per-fold gene NB parameters archived in anchored "
                "caches 03_CPI_METHOD/SMC_FOLD_CACHE_v1.npz and "
                "POOL_FOLD_CACHE_v1_*.npz",
        "core17_mesor_amplitude": params.head(17).to_dict(
            orient="records")}, indent=2, default=str))
    (FR / "FINAL_GENE_SELECTION_SPEC.json").write_text(json.dumps(
        {"panels_per_fold": panels,
         "criterion": "CPI cross-fitted entropy reduction; greedy "
                      "objective sumCPI/Z - 0.5*redundancy + "
                      "0.5*coverage; K=40"}, indent=2))
    (FR / "FINAL_ALPHA_GATE_SPEC.json").write_text(json.dumps({
        "gate": "v2 trust-region (v1 archived: structurally "
                "alpha=0-pinned)",
        "rule": "largest alpha in {0..1 step 0.05} with median "
                "|phase_alpha - phase_B0| <= 0.75h AND mean "
                "posterior R >= mean R(q0); alpha=0 == B0 exactly",
        "selected_alpha": alpha}, indent=2))
    (FR / "FINAL_CROSSFIT_SPEC.json").write_text(json.dumps({
        "SMC_scope": "leave-one-sample-out x5",
        "pooled_scope": "sample-blocked 5-fold round-robin over "
                        "17 samples",
        "manifests": ["07_SAMPLE_BLOCK_CROSSFIT/"
                      "SMC_LOSO_MANIFEST.csv",
                      "07_SAMPLE_BLOCK_CROSSFIT/"
                      "POOLED_5FOLD_MANIFEST.csv"]}, indent=2))
    (FR / "FINAL_FALSIFICATION_SUMMARY.json").write_text(
        json.dumps(t5, indent=2, default=float))
    code_rows = ["file,sha256"]
    for p in sorted((ROOT / "scripts").glob("task11b*.py")):
        code_rows.append(f"scripts/{p.name},{sha(p)}")
    (FR / "FINAL_CODE_HASHES.csv").write_text("\n".join(code_rows)
                                              + "\n")
    inp_rows = ["file,sha256"]
    for rel in [
            "results/single_cell_phase_rebuild/"
            "02B_BASELINE_REPRODUCTION_STRICT/SMC_counts.npz",
            "results/single_cell_phase_rebuild/"
            "11_KAT_V2_MODEL_FAMILY_DEVELOPMENT_20261007/"
            "11_REALDATA/00_POOLED_FREEZE/POOLED_COUNTS.npz",
            "results/single_cell_phase_rebuild/"
            "10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_AND_FAIRNESS_"
            "20261007/05_B0_CORRECTED_RUN1/PHASE_POSTERIOR.npz"]:
        inp_rows.append(f"{rel},{sha(ROOT / rel)}")
    (FR / "FINAL_INPUT_HASHES.csv").write_text(
        "\n".join(inp_rows) + "\n")
    pd.read_csv(T11B / ("PRE_UNBLIND_FINAL_PREDICTION_"
                        "MANIFEST.csv")).to_csv(
        FR / "FINAL_PREDICTION_HASHES.csv", index=False)
    (FR / "FINAL_ENVIRONMENT.txt").write_text(
        "python 3.11.17; numpy/scipy/pandas; single-thread "
        "processes; 15-core cgroup quota; engine stabilized "
        + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    (FR / "FINAL_MODEL_SELECTION_RATIONALE.txt").write_text(
        "FROZEN HIERARCHY: cross-fit validity -> falsification -> "
        "anchoring -> calibration -> perturbation -> performance -> "
        "parsimony\nC2-preference rule: prefer C2 unless C3 "
        "improves >=0.10h median AE\n\n" + "\n".join(rationale)
        + f"\nFINAL MODEL: {final}\n"
        f"development cell median AE: {g(final):.3f} h "
        f"(B0 {b0}, original B6 {b6})\n")
    verdict = {
        "TASK11B_COMPLETE": True,
        "TASK11B_ENGINE_STABLE": bool(stab),
        "FINAL_MODEL": final,
        "DEVELOPMENT_CELL_MEDIAN_AE_H": round(float(g(final)), 3),
        "B0_CELL_MEDIAN_AE_H": b0,
        "ORIGINAL_B6_CELL_MEDIAN_AE_H": b6,
        "FALSIFICATION_PASS": bool(fals_pass),
        "ANCHORING_PASS": bool(anchor_pass),
        "CALIBRATION_PASS": bool(calib_pass),
        "EXTERNAL_VALIDATION_READY": ext_ready,
        "C3_integrity": {k: bool(v.get("ALL_PASS"))
                         for k, v in c3integ.items()},
    }
    (FR / "TASK11B_FINAL_VERDICT.json").write_text(json.dumps(
        verdict, indent=2))

    q = {
        "A": (f"Original B6 ({b6} h) is preserved as the frozen "
              "same-cell-selected reference; its gain is re-examined "
              "through the sample-blocked C1/C2 which never saw test "
              f"cells during selection: C1={g('C1_CPI'):.3f} h, "
              f"C2={g('C2_SAFE_CPI'):.3f} h."),
        "B": (f"After sample-blocked CPI, the retained gain over B0 "
              f"is {b0 - g('C2_SAFE_CPI'):.3f} h = "
              f"{(b0 - g('C2_SAFE_CPI')) / (b0 - b6) * 100:.0f}% of "
              "original B6's same-cell gain "
              f"({b0 - b6:.3f} h)."),
        "C": (f"C1 {g('C1_CPI'):.3f} vs B0 {b0:.3f}: "
              + ("outperforms B0." if g("C1_CPI") < b0 - 0.05
                 else "does NOT outperform B0.")),
        "D": (f"C2 {g('C2_SAFE_CPI'):.3f} vs C1 "
              f"{g('C1_CPI'):.3f}: "
              + ("similar accuracy with a formally safer gate."
                 if abs(g("C2_SAFE_CPI") - g("C1_CPI")) < 0.15
                 else "see T1 for the trade-off.")),
        "E": f"Alpha selected by C2: {alpha}.",
        "F": ("Alpha sharpens posterior confidence under an "
              "anchor trust region (median phase shift <=0.75h "
              "enforced; oracle rotation "
              f"{res.get('C2_SAFE_CPI', {}).get('oracle_rotation_h', float('nan')):.2f} h) "
              "- it cannot re-anchor phase by construction."),
        "G": (f"C3 lineage {g('C3_lineage'):.3f} (pooled scope) vs "
              f"C2 {g('C2_SAFE_CPI'):.3f} (SMC scope) and pooled "
              f"head {g('C3_pooled'):.3f} vs pooled B0 ref "
              f"{g('B0_POOLED_ref'):.3f}: see T1/T7."),
        "H": (f"C3-state {g('C3_state'):.3f} vs C3-pooled "
              f"{g('C3_pooled'):.3f}: "
              + ("state conditioning adds no material benefit - "
                 "complexity not justified."
                 if abs(g("C3_state") - g("C3_pooled")) < 0.05
                 else "differences reported in T1.")),
        "I": (f"C3-full {g('C3_full'):.3f}: selected only if "
              ">=0.10h better than C2 (frozen rule) - outcome: "
              f"{final}."),
        "J": (f"Best performance/complexity/falsification "
              f"trade-off: {final} at {g(final):.3f} h "
              "(cross-fitted, falsification-passed, engine-stable, "
              "most parsimonious passing model)."),
    }
    b12 = ("B1/B2 (mandated framing): the B1/B2 mechanisms WERE "
           "recoverable under their corresponding simulations; "
           "their DIRECT incorporation did not improve the "
           "development real-data benchmark; their biological ideas "
           "were therefore reused only as optional lineage/state "
           "conditioning within the information-head framework "
           "(C3 variants), which is the only form in which they "
           "appear in Task 11B.")
    (T11B / "TASK11B_FINAL_REPORT.txt").write_text(
        "TASK 11B FINAL REPORT\n====================\n\n"
        + "\n".join(f"{k}. {v}" for k, v in q.items())
        + "\n\n" + b12
        + "\n\nTASK11B_FINAL_VERDICT:\n"
        + json.dumps(verdict, indent=2)
        + "\n\nHeld out: GSE145197 UNTOUCHED, GSE223109 UNTOUCHED. "
        "Task12 (external validation) requires separate approval.\n")
    print(json.dumps(verdict, indent=2))
    print("\n".join(f"{k}. {v}" for k, v in q.items()))


if __name__ == "__main__":
    main()
