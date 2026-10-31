"""Task11B shared utilities: ONE prediction assembler and ONE cache
API (anchored schema v1). Stabilization mandate 2026-10-07: no
script may hand-write prediction assembly or parse cache keys from
arbitrary npz key names.

assemble_predictions contract
-----------------------------
ALL posterior quantities are LOCAL to the test fold:
    q_local[local_i], mean_local[local_i], R_local[local_i],
    entropy_local[local_i], ci_local[local_i]
ALL metadata is GLOBAL:
    obs_ids_global[global_i], sample_ids_global[global_i],
    lineage_global[global_i]
iterating: for local_i, global_i in enumerate(test_global_idx).

Cache schema v1
---------------
save_fold_cache(path, *, schema_version=1, fold_ids,
train_sample_ids, test_sample_ids, gene_names, cell_count_train,
cell_count_test, params_by_fold, cpi, per_fold, code_hash,
input_hash)
load_fold_cache(path) -> dict (validates schema, returns None if
legacy/invalid, with reason logged)
validate_fold_cache(path) -> (ok, reason)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
CACHE_KEYS_REQUIRED = ("schema_version", "fold_ids", "gene_names",
                       "cpi", "per_fold", "code_hash", "input_hash")


def assemble_predictions(test_global_idx, q_local, mean_local,
                         R_local, entropy_local, ci_local,
                         obs_ids_global, sample_ids_global,
                         lineage_global=None, map_local=None,
                         theta_grid_h=None):
    """Single source of truth for fold prediction assembly.

    test_global_idx: int array of GLOBAL cell indices (any order,
    non-contiguous allowed).
    q_local: (n_test, K) posterior LOCAL rows aligned with
    test_global_idx order.
    map_local: optional (n_test,) MAP bin (local); if None, computed
    from q_local with theta_grid_h.
    Returns DataFrame with one row per test cell in
    test_global_idx order.
    """
    n_test = len(test_global_idx)
    assert q_local.shape[0] == n_test == len(mean_local) \
        == len(R_local) == len(entropy_local) == len(ci_local), \
        "local/global length mismatch"
    if map_local is None:
        assert theta_grid_h is not None
        map_local = np.asarray(theta_grid_h)[q_local.argmax(axis=1)]
    ids = np.asarray(obs_ids_global)
    sids = np.asarray(sample_ids_global)
    if lineage_global is None:
        # default lineage array must be GLOBAL-sized so that
        # lins[global_i] is well-defined for non-contiguous folds
        lins = np.full(len(ids), "", dtype=object)
    else:
        lins = np.asarray(lineage_global, dtype=object)
        assert len(lins) == len(ids), "lineage_global must be " \
            "global-sized"
    rows = []
    for local_i, global_i in enumerate(test_global_idx):
        rows.append((ids[global_i],            # global metadata
                     sids[global_i],
                     lins[global_i],
                     mean_local[local_i],       # local posterior
                     map_local[local_i],
                     R_local[local_i],
                     entropy_local[local_i],
                     ci_local[local_i]))
    df = pd.DataFrame(rows, columns=[
        "cell_id", "blind_sample_id", "lineage", "phase_mean_h",
        "phase_MAP_h", "posterior_R", "entropy_nats",
        "ci95_width_h"])
    if len(df.cell_id) != len(df.cell_id.unique()):
        raise RuntimeError("duplicated cell_id in assembled "
                           "predictions")
    return df


# =========================================================== cache API
def save_fold_cache(path, *, fold_ids, train_sample_ids,
                    test_sample_ids, gene_names, cell_count_train,
                    cell_count_test, params_by_fold, cpi, per_fold,
                    code_hash, input_hash, extra=None):
    payload = dict(
        schema_version=SCHEMA_VERSION,
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        train_sample_ids=np.asarray(train_sample_ids,
                                    dtype=object),
        test_sample_ids=np.asarray(test_sample_ids, dtype=object),
        gene_names=np.asarray(gene_names, dtype=object),
        cell_count_train=np.asarray(cell_count_train,
                                    dtype=np.int64),
        cell_count_test=np.asarray(cell_count_test,
                                   dtype=np.int64),
        cpi=np.asarray(cpi), per_fold=np.asarray(per_fold),
        code_hash=str(code_hash), input_hash=str(input_hash))
    for k, v in (extra or {}).items():
        payload[f"x_{k}"] = v
    for fk, p in params_by_fold.items():
        for i, arr in enumerate(p):
            payload[f"fold{fk}_param{i}"] = np.asarray(arr)
    np.savez_compressed(path, **payload)


def validate_fold_cache(path):
    try:
        z = np.load(path, allow_pickle=True)
        if "schema_version" not in z.files:
            return False, "LEGACY_NO_SCHEMA (needs migration)"
        if int(z["schema_version"]) != SCHEMA_VERSION:
            return False, f"schema {int(z['schema_version'])}!=1"
        for k in CACHE_KEYS_REQUIRED:
            if k not in z.files:
                return False, f"missing {k}"
        n_fold = len(z["fold_ids"])
        if not all(f"fold{k}_param0" in z.files
                   for k in z["fold_ids"]):
            return False, "missing fold param arrays"
        cpi = z["cpi"]
        if not np.isfinite(cpi).all():
            return False, "non-finite CPI"
        return True, f"ok folds={n_fold} genes={len(cpi)}"
    except Exception as e:
        return False, f"unreadable: {e}"


def load_fold_cache(path):
    ok, reason = validate_fold_cache(path)
    if not ok:
        return None, reason
    z = np.load(path, allow_pickle=True)
    folds = [int(k) for k in z["fold_ids"]]

    def arity(k):
        i = 0
        while f"fold{k}_param{i}" in z.files:
            i += 1
        return i

    params = {k: tuple(z[f"fold{k}_param{i}"]
                       for i in range(arity(k))) for k in folds}
    return {
        "schema_version": int(z["schema_version"]),
        "fold_ids": folds,
        "train_sample_ids": list(z["train_sample_ids"]),
        "test_sample_ids": list(z["test_sample_ids"]),
        "gene_names": [str(g) for g in z["gene_names"]],
        "cell_count_train": z["cell_count_train"].tolist(),
        "cell_count_test": z["cell_count_test"].tolist(),
        "params": params, "cpi": z["cpi"], "per_fold": z["per_fold"],
        "code_hash": str(z["code_hash"]),
        "input_hash": str(z["input_hash"]),
        "extra": {k[2:]: z[k] for k in z.files
                  if k.startswith("x_")}}, None


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
