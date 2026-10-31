"""Task 10A stages 00-02 rebuilt clean on katv2-clone.

00_INTEGRITY          pre-run integrity of frozen tasks 06-10 (+ cross-check
                      against the quarantined instance-A snapshot manifest)
01_OFFSET_FORENSIC    audit the OLD clock17-subtotal offset
02_CORRECTED_SIZE_FACTOR_FREEZE
                      whole-transcriptome total UMI + median-1 size factor

Reads only frozen inputs; writes only inside the new 10A directory.
"""
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results/single_cell_phase_rebuild"
A = RR / "10A_KAT_V2_B0_LIBRARY_OFFSET_CORRECTION_AND_FAIRNESS_20261007"
T10 = RR / "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"
FREEZE = T10 / "02_DATA_AND_PRIOR_FREEZE"
B02 = RR / "02B_BASELINE_REPRODUCTION_STRICT"
SNAP = RR / ("FAILED_ATTEMPTS/CLONED_PARTIAL_FROM_INSTANCE_A/"
             "00_INTEGRITY/PRE_RUN_MANIFEST.sha256")

FROZEN_TASKS = ["06_A1_A4_BLINDED_BATCH_20261006",
                "07_A6_PROTOCOL_RECONCILIATION_FORENSIC_20261006",
                "08_A6_A4_LOGA_INITIALIZATION_SWAP_CAUSAL_20261006",
                "09_A6_A4_BATCH_ORDER_SWAP_CAUSAL_20261006",
                "10_KAT_V2_FOUNDATION_SINGLECELL_BENCHMARK_20261006"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage00():
    out = A / "00_INTEGRITY"
    out.mkdir(parents=True, exist_ok=True)
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    keep = [ln for ln in cpu.splitlines() if any(
        k in ln for k in ("CPU(s)", "Model name", "Thread", "Core",
                          "Socket", "NUMA node(s)"))]
    (out / "CPU_INFO.txt").write_text(
        "\n".join(keep) + f"\nnproc={os.cpu_count()}\n")

    manifest, snap = {}, {}
    if SNAP.exists():
        for ln in SNAP.read_text().splitlines():
            if len(ln.split()) == 2:
                h, p = ln.split()
                snap[p] = h
    n_task10 = 0
    with open(out / "PRE_RUN_MANIFEST.sha256", "w") as f:
        for t in FROZEN_TASKS:
            for p in sorted((RR / t).rglob("*")):
                if p.is_file():
                    h = sha(p)
                    rel = str(p.relative_to(RR))
                    f.write(f"{h}  {rel}\n")
                    manifest[rel] = h
                    if t.startswith("10_"):
                        n_task10 += 1
    common = set(manifest) & set(snap)
    changed = [p for p in common if manifest[p] != snap[p]]
    # snapshot manifest paths were relative to RR on instance A as well
    try:
        cg = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    except OSError:
        cg = "unknown"
    integ = {
        "files_hashed": len(manifest),
        "task10_files_hashed": n_task10,
        "cross_instance_check": {
            "snapshot_manifest": "FAILED_ATTEMPTS/CLONED_PARTIAL_"
                                 "FROM_INSTANCE_A/00_INTEGRITY/"
                                 "PRE_RUN_MANIFEST.sha256",
            "files_in_both": len(common),
            "files_changed_since_instance_A_snapshot": len(changed),
            "changed_examples": changed[:5],
            "note": "0 changed => frozen tasks 06-10 identical to the "
                    "instance-A snapshot taken before its partial 10A"
        },
        "cgroup_memory_max_bytes": int(cg) if cg.isdigit() else cg,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "PRE_RUN_INTEGRITY.json").write_text(json.dumps(integ, indent=2))
    log(f"00_INTEGRITY: {len(manifest)} files hashed, "
        f"{len(changed)} changed vs instance-A snapshot")
    return integ


def _read_selection():
    with open(B02 / "01_SMC_CELL_SELECTION_AUDIT.csv") as f:
        r = csv.DictReader(f)
        return [row["cell_id"] for row in r]


def stage01():
    out = A / "01_OFFSET_FORENSIC"
    out.mkdir(parents=True, exist_ok=True)
    Xa = sp.load_npz(FREEZE / "anchored_counts.npz").tocsr()
    c17 = np.asarray(Xa.sum(axis=1)).ravel().astype(np.float64)
    np.save(out / "clock17_sum.npy", c17)
    old_lib = np.load(FREEZE / "anchored_lib.npy")
    exact = bool(np.array_equal(c17, old_lib))
    audit = {
        "ORIGINAL_OFFSET_TYPE": "CLOCK17_SUBTOTAL",
        "definition": "lib_i = sum of raw UMI over the 17 anchored genes",
        "min": float(c17.min()), "median": float(np.median(c17)),
        "mean": float(c17.mean()), "max": float(c17.max()),
        "zeros": int((c17 == 0).sum()),
        "exact_match_vs_task10_anchored_lib": exact,
        "clock17_sum_sha256": hashlib.sha256(
            c17.tobytes()).hexdigest(),
        "model_usage_note": "old B0 entered log(max(lib,1.0)); 257 cells "
                            "with clock17 subtotal 0 sat on the clamp",
    }
    (out / "ORIGINAL_OFFSET_AUDIT.json").write_text(
        json.dumps(audit, indent=2))
    (out / "ORIGINAL_OFFSET_PROVENANCE.txt").write_text(
        "OLD offset (task10 B0): clock17 subtotal from frozen "
        "anchored_counts.npz; identical to 02_DATA_AND_PRIOR_FREEZE/"
        "anchored_lib.npy (exact_match="
        f"{exact}).\nNEW offset (10A): whole-transcriptome size factor "
        "frozen in 02_CORRECTED_SIZE_FACTOR_FREEZE.\n")
    log(f"01_OFFSET_FORENSIC: clock17 exact_match={exact}, "
        f"zeros={audit['zeros']}")
    return c17


def stage02(c17):
    out = A / "02_CORRECTED_SIZE_FACTOR_FREEZE"
    out.mkdir(parents=True, exist_ok=True)
    X = sp.load_npz(B02 / "SMC_counts.npz").tocsr()
    tot = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    ids = _read_selection()
    assert len(ids) == tot.shape[0] == 18863, "cell count mismatch"

    # row-alignment audit: recompute clock17 subtotal from the FULL matrix
    # using the anchored gene indices and require exact equality with the
    # frozen 17-gene subset sums
    with open(B02 / "SMC_var.csv") as f:
        r = csv.DictReader(f)
        var_names = [row[list(row.keys())[0]] for row in r]
    prior = json.loads((FREEZE / "ACROPHASE_PRIOR_FREEZE.json")
                       .read_text())
    genes = list(prior["priors"].keys())
    idx17 = [var_names.index(g) for g in genes]
    Xa_full = X[:, idx17]
    c17_check = np.asarray(Xa_full.sum(axis=1)).ravel()
    align_exact = bool(np.array_equal(c17_check, c17))
    if not align_exact:
        raise RuntimeError("STOP_ROW_ALIGNMENT_FAILURE full matrix vs "
                           "frozen anchored subset")
    with open(out / "CELL_ALIGNMENT_AUDIT.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "value"])
        w.writerow(["n_cells", len(ids)])
        w.writerow(["cell_id_sha256", hashlib.sha256(
            "\n".join(ids).encode()).hexdigest()])
        w.writerow(["full_matrix_rows", X.shape[0]])
        w.writerow(["full_matrix_genes", X.shape[1]])
        w.writerow(["clock17_recomputed_from_full_exact",
                    str(align_exact)])

    if tot.min() <= 0:
        bad = int((tot <= 0).sum())
        raise RuntimeError(f"STOP_INVALID_QC_CELLS {bad} cells with "
                           "total UMI<=0")
    med = float(np.median(tot))
    sf = tot / med
    assert np.isfinite(sf).all() and (sf > 0).all()
    assert abs(float(np.median(sf)) - 1.0) < 1e-12

    np.save(out / "TOTAL_UMI_ALL_GENES.npy", tot)
    np.save(out / "SIZE_FACTOR_TOTAL_UMI_MEDIAN1.npy", sf)
    with gzip.open(out / "SIZE_FACTOR_TOTAL_UMI_MEDIAN1.csv.gz", "wt",
                   newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "total_umi", "size_factor"])
        for cid, t, s in zip(ids, tot, sf):
            w.writerow([cid, f"{t:.1f}", f"{s:.10g}"])

    # offset correlations old vs new
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    correlations = {
        "pearson_clock17_sum_vs_total_umi": corr(c17, tot),
        "spearman_clock17_sum_vs_total_umi": float(
            __import__("scipy.stats", fromlist=["spearmanr"])
            .spearmanr(c17, tot).statistic),
        "interpretation": "clock17 subtotal is a noisy proxy (257 zeros) "
                          "for whole-transcriptome depth; correcting the "
                          "offset removes a per-cell scaling error of up "
                          "to several-fold in shallow cells",
    }
    (A / "01_OFFSET_FORENSIC/OFFSET_CORRELATIONS.json").write_text(
        json.dumps(correlations, indent=2))
    # compact per-quantile comparison (old clamp offset vs new)
    with open(A / "01_OFFSET_FORENSIC/OFFSET_COMPARISON.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantile", "clock17_sum", "log_clamp_old",
                    "total_umi", "log_sf_new"])
        qs = np.linspace(0, 100, 9)
        for q in qs:
            w.writerow([f"Q{q:.0f}",
                        float(np.percentile(c17, q)),
                        float(np.log(np.clip(
                            np.percentile(c17, q), 1, None))),
                        float(np.percentile(tot, q)),
                        float(np.log(np.percentile(sf, q)))])

    def arr_sha(x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    freeze = {
        "formula": "SIZE_FACTOR_i = TOTAL_UMI_i / median(TOTAL_UMI)",
        "median_denominator": med,
        "total_umi_stats": {
            "N": int(tot.shape[0]), "min": float(tot.min()),
            "median": med, "mean": float(tot.mean()),
            "max": float(tot.max()), "zeros": int((tot == 0).sum())},
        "size_factor_stats": {
            "N": int(sf.shape[0]), "min": float(sf.min()), "median":
            float(np.median(sf)), "mean": float(sf.mean()),
            "max": float(sf.max()), "zeros": int((sf == 0).sum()),
            "finite": bool(np.isfinite(sf).all())},
        "source_full_count_sha256": sha(B02 / "SMC_counts.npz"),
        "cell_id_sha256": hashlib.sha256(
            "\n".join(ids).encode()).hexdigest(),
        "total_umi_sha256": arr_sha(tot),
        "size_factor_sha256": arr_sha(sf),
        "row_alignment_exact": align_exact,
    }
    # cross-instance consistency with the quarantined instance-A freeze
    qj = RR / ("FAILED_ATTEMPTS/CLONED_PARTIAL_FROM_INSTANCE_A/"
               "02_CORRECTED_SIZE_FACTOR_FREEZE/SIZE_FACTOR_FREEZE.json")
    if qj.exists():
        q = json.loads(qj.read_text())
        keys = ["source_full_count_sha256", "cell_id_sha256",
                "total_umi_sha256", "size_factor_sha256", "median_denominator"]
        freeze["cross_instance_consistency_vs_instance_A"] = {
            k: {"instance_A": q.get(k), "this_run": freeze.get(k),
                "match": q.get(k) == freeze.get(k)} for k in keys}
    (out / "SIZE_FACTOR_FREEZE.json").write_text(json.dumps(freeze,
                                                            indent=2))
    log(f"02_CORRECTED_SIZE_FACTOR_FREEZE: median_denominator={med}, "
        f"all>0, median(sf)=1, hashes frozen")
    return freeze


def main():
    assert not (A / "05_B0_CORRECTED_RUN1").exists() or not any(
        (A / "05_B0_CORRECTED_RUN1").iterdir()), "STOP_DIRTY_OUTPUT"
    stage00()
    c17 = stage01()
    stage02(c17)
    log("prep complete (stages 00-02)")


if __name__ == "__main__":
    main()
