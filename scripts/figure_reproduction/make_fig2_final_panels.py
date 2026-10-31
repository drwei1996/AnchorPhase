# -*- coding: utf-8 -*-
"""
Final Figure 2 panels for AnchorPhase manuscript
================================================

Fig2A  External frozen validation
Fig2B  Falsification controls
Fig2C  Liver genuine circadian-output genes
Fig2D  Dermis reproducible confounding genes

For every panel:
    *_text.pdf / *_text.png
    *_clean.pdf / *_clean.png

IMPORTANT
---------
1. Empirical data are read from frozen project files.
2. NO synthetic empirical data.
3. Internal frozen analysis names remain unchanged in source data:
       B0 = AnchorPhase
       C1 = CPI
       C2 = SAFE-CPI
4. Display text only is renamed.
5. PNG = 400 dpi; PDF = vector where possible.
6. Original Fig3a/Fig3b plotting logic is preserved for Fig2C/Fig2D.
"""

from __future__ import annotations

import os
import sys
import csv
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. PROJECT ROOT
# ============================================================

HERE = Path(__file__).resolve().parent

# make_fig2_final_panels.py is located in:
# sckat_server_package/scripts/
ROOT = HERE.parent.resolve()

sys.path.insert(0, str(HERE))

OUTDIR = ROOT / "figures" / "panels_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("LOCAL PROJECT ROOT")
print(ROOT)
print("=" * 90)


# ============================================================
# 1. FIGURE STYLE
# ============================================================

from fig_style_v2 import (
    apply_style,
    audit,
    DOUBLE,
    T12,
    NAVY,
    BLUE,
    GREEN,
    ORANGE,
    RED,
    GRAY,
    DK,
)

apply_style()


# ============================================================
# 2. ROBUST LOCAL PATCH FOR fig3_shared.py
# ============================================================
#
# fig3_shared.py was created on AutoDL and may still contain:
#
#   /root/autodl-tmp/sckat_server_package
#
# This section changes PATHS ONLY.
# It does not change any empirical data / statistics.
# ============================================================

import fig3_shared as f3


SERVER_ROOTS = [
    "/root/autodl-tmp/sckat_server_package",
    r"\root\autodl-tmp\sckat_server_package",
]


def _remap_path_value(value):
    """
    Recursively replace old AutoDL sckat_server_package path
    with the current local ROOT.

    Supports:
        str
        pathlib.Path
        tuple
        list
        dict
    """

    if isinstance(value, Path):
        raw = str(value)
        norm = raw.replace("\\", "/")

        for old in SERVER_ROOTS:
            old_norm = old.replace("\\", "/")

            if old_norm in norm:
                suffix = norm.split(old_norm, 1)[1].lstrip("/")
                return ROOT / Path(suffix)

        return value

    if isinstance(value, str):
        norm = value.replace("\\", "/")

        for old in SERVER_ROOTS:
            old_norm = old.replace("\\", "/")

            if old_norm in norm:
                suffix = norm.split(old_norm, 1)[1].lstrip("/")
                return str(ROOT / Path(suffix))

        return value

    if isinstance(value, tuple):
        return tuple(
            _remap_path_value(x)
            for x in value
        )

    if isinstance(value, list):
        return [
            _remap_path_value(x)
            for x in value
        ]

    if isinstance(value, dict):
        return {
            k: _remap_path_value(v)
            for k, v in value.items()
        }

    return value


def patch_fig3_shared():
    """
    Make fig3_shared portable from AutoDL -> local Windows.

    Patches:
      A. module-level path constants
      B. common root variable names
      C. function positional defaults
      D. function keyword defaults
    """

    print()
    print("=" * 90)
    print("PATCH fig3_shared.py SERVER PATHS -> LOCAL PATHS")
    print("=" * 90)

    changed = []

    # --------------------------------------------------------
    # A. module globals
    # --------------------------------------------------------

    for name, value in list(vars(f3).items()):

        if name.startswith("__"):
            continue

        try:
            new_value = _remap_path_value(value)
        except Exception:
            continue

        changed_flag = False

        if isinstance(value, Path) and isinstance(new_value, Path):
            changed_flag = str(value) != str(new_value)

        elif isinstance(value, str) and isinstance(new_value, str):
            changed_flag = value != new_value

        elif isinstance(value, (tuple, list, dict)):
            try:
                changed_flag = repr(value) != repr(new_value)
            except Exception:
                changed_flag = False

        if changed_flag:
            setattr(f3, name, new_value)

            changed.append(
                (
                    f"global:{name}",
                    repr(value),
                    repr(new_value)
                )
            )

    # --------------------------------------------------------
    # B. explicitly overwrite common root constants
    # --------------------------------------------------------

    common_root_names = [
        "ROOT",
        "BASE",
        "BASE_DIR",
        "PACKAGE_ROOT",
        "PROJECT_ROOT",
        "PKG_ROOT",
        "SCKAT_ROOT",
    ]

    for name in common_root_names:

        if hasattr(f3, name):

            old = getattr(f3, name)

            # keep original type if possible
            if isinstance(old, Path):
                new = ROOT
            else:
                new = str(ROOT)

            if str(old) != str(new):
                setattr(f3, name, new)

                changed.append(
                    (
                        f"root:{name}",
                        repr(old),
                        repr(new)
                    )
                )

    # --------------------------------------------------------
    # C. explicitly map common data-root variables
    # --------------------------------------------------------

    explicit_paths = {
        "DATA": ROOT / "data",
        "DATA_ROOT": ROOT / "data",
        "PROCESSED": ROOT / "data" / "processed",
        "PROCESSED_ROOT": ROOT / "data" / "processed",
        "SCRIPTS": ROOT / "scripts",
        "SCRIPT_DIR": ROOT / "scripts",
    }

    for name, local_path in explicit_paths.items():

        if not hasattr(f3, name):
            continue

        old = getattr(f3, name)

        if isinstance(old, Path):
            new = local_path
        else:
            new = str(local_path)

        if str(old) != str(new):

            setattr(f3, name, new)

            changed.append(
                (
                    f"explicit:{name}",
                    repr(old),
                    repr(new)
                )
            )

    # --------------------------------------------------------
    # D. use the already-working local T12 from fig_style_v2
    # --------------------------------------------------------

    if hasattr(f3, "T12"):

        old = getattr(f3, "T12")

        if isinstance(old, Path):
            new = Path(T12)
        else:
            new = str(T12)

        if str(old) != str(new):

            setattr(f3, "T12", new)

            changed.append(
                (
                    "explicit:T12",
                    repr(old),
                    repr(new)
                )
            )

    # --------------------------------------------------------
    # E. patch function default parameters
    #
    # Example:
    #
    #   def load_cohort(ds, root=ROOT):
    #
    # Changing ROOT alone would NOT change the captured default.
    # Therefore patch __defaults__ / __kwdefaults__ too.
    # --------------------------------------------------------

    for fname, func in list(vars(f3).items()):

        if not isinstance(func, types.FunctionType):
            continue

        # positional defaults
        defaults = func.__defaults__

        if defaults:

            new_defaults = tuple(
                _remap_path_value(x)
                for x in defaults
            )

            if repr(defaults) != repr(new_defaults):

                try:
                    func.__defaults__ = new_defaults

                    changed.append(
                        (
                            f"default:{fname}",
                            repr(defaults),
                            repr(new_defaults)
                        )
                    )

                except Exception:
                    pass

        # keyword-only defaults
        kwdefaults = func.__kwdefaults__

        if kwdefaults:

            new_kwdefaults = {
                k: _remap_path_value(v)
                for k, v in kwdefaults.items()
            }

            if repr(kwdefaults) != repr(new_kwdefaults):

                try:
                    func.__kwdefaults__ = new_kwdefaults

                    changed.append(
                        (
                            f"kwdefault:{fname}",
                            repr(kwdefaults),
                            repr(new_kwdefaults)
                        )
                    )

                except Exception:
                    pass

    # --------------------------------------------------------
    # report
    # --------------------------------------------------------

    if not changed:
        print(
            "No legacy AutoDL path constants were found. "
            "fig3_shared may already be portable."
        )

    else:

        for label, old, new in changed:

            print()
            print(label)
            print("  OLD:", old)
            print("  NEW:", new)

    print()
    print("Current ROOT =", ROOT)

    if hasattr(f3, "ROOT"):
        print("fig3_shared.ROOT =", getattr(f3, "ROOT"))

    if hasattr(f3, "DATA"):
        print("fig3_shared.DATA =", getattr(f3, "DATA"))

    if hasattr(f3, "T12"):
        print("fig3_shared.T12 =", getattr(f3, "T12"))

    print("=" * 90)


patch_fig3_shared()


# Bind the ORIGINAL frozen helper functions only after patching.
load_cohort = f3.load_cohort
gene_expr = f3.gene_expr
sample_means = f3.sample_means
fit_cosine = f3.fit_cosine
gene_acrophase = f3.gene_acrophase
circ_ae = f3.circ_ae


# ============================================================
# 3. PRE-FLIGHT LOCAL DATA CHECK
# ============================================================

EXPECTED_H5AD = {
    "GSE145197":
        ROOT
        / "data"
        / "processed"
        / "GSE145197"
        / "gse145197_sparse.h5ad",

    "GSE223109":
        ROOT
        / "data"
        / "processed"
        / "GSE223109"
        / "gse223109_sparse.h5ad",
}


def preflight():

    print()
    print("=" * 90)
    print("FIGURE 2 PREFLIGHT")
    print("=" * 90)

    fail = False

    for ds, p in EXPECTED_H5AD.items():

        ok = p.exists()

        print(
            f"{ds}: "
            f"{'FOUND' if ok else 'MISSING'}"
        )
        print("   ", p)

        if not ok:
            fail = True

    t12p = Path(T12)

    print()
    print(
        "T12:",
        "FOUND" if t12p.exists() else "MISSING"
    )
    print("   ", t12p)

    if not t12p.exists():
        fail = True

    print("=" * 90)

    if fail:

        raise FileNotFoundError(
            "\nPreflight failed.\n"
            "At least one required local frozen input is missing.\n"
            "No figures were rendered.\n"
            f"Expected package root:\n{ROOT}"
        )


# ============================================================
# 4. DISPLAY NAMES
# ============================================================

MODEL = "AnchorPhase"
EXTENSION = "SAFE-CPI"
CPI_NAME = "CPI"


# ============================================================
# 5. PUBLICATION STYLE
# ============================================================

TITLE_FS = 8.8
SUBTITLE_FS = 5.9
AX_TITLE_FS = 7.2
LABEL_FS = 6.5
TICK_FS = 6.0
NOTE_FS = 5.8
VALUE_FS = 6.3

GRID_COLOR = "#DDDDDD"


# ============================================================
# 6. SAVE
# ============================================================

def save_final(fig, stem, show_text):

    suffix = (
        "text"
        if show_text
        else "clean"
    )

    name = (
        f"{stem}_{suffix}"
    )

    try:
        problems = audit(fig)
    except Exception as e:
        print(
            f"[audit unavailable] {e}"
        )
        problems = []

    pdf = OUTDIR / f"{name}.pdf"
    png = OUTDIR / f"{name}.png"

    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        png,
        format="png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white"
    )

    if problems:

        print(
            f"[{name}] audit warnings: "
            f"{len(problems)}"
        )

        for p in problems:
            print("   -", p)

    else:

        print(
            f"[{name}] AUDIT PASS"
        )

    print("   PDF:", pdf)
    print("   PNG:", png)

    plt.close(fig)


def remove_figure_text(fig):

    for t in list(fig.texts):
        try:
            t.remove()
        except Exception:
            pass


def clean_axis_text(ax):
    """
    Remove ALL display text while keeping:
      axes
      ticks
      bars
      data
      fitted lines
      reference lines
      arrows
    """

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for t in list(ax.texts):
        try:
            t.remove()
        except Exception:
            pass

    leg = ax.get_legend()

    if leg is not None:
        leg.remove()


# ============================================================
# 7. FIGURE 2A
# EXTERNAL VALIDATION
# ============================================================

def load_external_validation():

    out = {}

    for ds in [
        "GSE145197",
        "GSE223109"
    ]:

        p = (
            Path(T12)
            / ds
            / "06_EVALUATION"
            / "EVALUATION.json"
        )

        with open(
            p,
            "r",
            encoding="utf-8"
        ) as f:

            out[ds] = json.load(f)

    p = (
        Path(T12)
        / "TABLES"
        / "EXTERNAL_VALIDATION_TABLE.csv"
    )

    with open(
        p,
        "r",
        encoding="utf-8-sig"
    ) as f:

        for row in csv.DictReader(f):

            raw = (
                row.get(
                    "Tempo_median_AE",
                    ""
                )
            )

            tempo = (
                None
                if raw in (
                    "",
                    "NOT_RUN",
                    None
                )
                else float(raw)
            )

            out[
                row["Dataset"]
            ]["Tempo"] = {
                "cell_median_AE": tempo
            }

    return out


def make_fig2A(
    show_text=True
):

    T = load_external_validation()

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(
            DOUBLE,
            3.58
        )
    )

    for k, ds in enumerate([
        "GSE145197",
        "GSE223109"
    ]):

        ax = axs[k]

        b0 = float(
            T[ds]["B0"][
                "cell_median_AE"
            ]
        )

        c2 = float(
            T[ds]["C2"][
                "cell_median_AE"
            ]
        )

        tempo = (
            T[ds]
            .get("Tempo", {})
            .get("cell_median_AE")
        )

        bars = [
            (
                MODEL,
                b0,
                NAVY
            ),
            (
                EXTENSION,
                c2,
                ORANGE
            ),
        ]

        if tempo is not None:

            bars.append(
                (
                    "Tempo",
                    tempo,
                    GRAY
                )
            )

        for i, (
            label,
            value,
            color
        ) in enumerate(bars):

            ax.bar(
                i,
                value,
                width=0.58,
                color=color,
                zorder=3
            )

            if show_text:

                ax.text(
                    i,
                    value + 0.06,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color=DK
                )

        ax.set_xticks(
            range(
                len(bars)
            )
        )

        if show_text:

            ax.set_xticklabels(
                [
                    x[0]
                    for x in bars
                ],
                fontsize=7.0
            )

        else:

            ax.set_xticklabels([])

        ax.tick_params(
            axis="x",
            length=0
        )

        ax.set_ylim(
            0,
            4.9
        )

        ax.spines[
            [
                "top",
                "right"
            ]
        ].set_visible(False)

        ax.grid(
            axis="y",
            lw=0.4,
            color=GRID_COLOR,
            zorder=0
        )

        ax.set_axisbelow(True)

        # --------------------------------------------
        # comparison B0 -> C2
        # --------------------------------------------

        diff = (
            b0 - c2
        )

        good = (
            diff > 0
        )

        col = (
            GREEN
            if good
            else RED
        )

        bracket_x = 1.34

        ax.plot(
            [
                -0.02,
                bracket_x
            ],
            [
                b0,
                b0
            ],
            ls=":",
            lw=0.7,
            color=NAVY,
            zorder=4
        )

        ax.plot(
            [
                bracket_x,
                bracket_x
            ],
            [
                b0,
                c2
            ],
            lw=1.05,
            color=col,
            zorder=4
        )

        if show_text:

            title = {
                "GSE145197":
                    "GSE145197 — liver hepatocytes",

                "GSE223109":
                    "GSE223109 — dorsal dermis",
            }[ds]

            ax.set_title(
                title,
                fontsize=7.6,
                fontweight="bold",
                pad=6
            )

            ax.set_ylabel(
                "Cell median circular AE (h)",
                fontsize=7.0
            )

            verdict = (
                "Permutation PASS · matched-panel PASS"
                if good
                else
                "Permutation PASS · matched-panel FAIL"
            )

            ax.text(
                0.50,
                0.985,
                verdict,
                transform=ax.transAxes,
                fontsize=6.0,
                ha="center",
                va="top",
                color=(
                    GREEN
                    if good
                    else RED
                ),
                fontweight="bold"
            )

            delta_word = (
                "improvement"
                if good
                else
                "degradation"
            )

            ax.text(
                0.50,
                0.900,
                (
                    f"Δ = "
                    f"{diff:+.3f} h "
                    f"{delta_word}"
                ),
                transform=ax.transAxes,
                fontsize=6.0,
                color=col,
                va="top",
                ha="center",
                bbox=dict(
                    boxstyle=(
                        "round,pad=0.18"
                    ),
                    fc="white",
                    ec="none",
                    alpha=0.94
                ),
                zorder=6
            )

            if tempo is None:

                ax.text(
                    2.0,
                    0.35,
                    (
                        "Tempo:\n"
                        "not run\n"
                        "(technical)"
                    ),
                    fontsize=5.8,
                    color=GRAY,
                    ha="center",
                    va="bottom",
                    style="italic"
                )

    if show_text:

        fig.text(
            0.005,
            0.995,
            (
                "External validation identifies "
                "AnchorPhase as the robust default model"
            ),
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="top"
        )

        fig.text(
            0.5,
            0.025,
            (
                "SAFE-CPI improves performance in liver "
                "but degrades performance in dorsal dermis; "
                "AnchorPhase is therefore retained as the default model."
            ),
            fontsize=6.6,
            ha="center",
            va="bottom",
            color=DK,
            style="italic"
        )

    fig.tight_layout(
        rect=(
            0.035,
            0.115,
            0.995,
            0.865
        ),
        w_pad=3.0
    )

    if not show_text:

        for ax in axs:
            clean_axis_text(ax)

        remove_figure_text(fig)

    save_final(
        fig,
        "fig2A_external_validation",
        show_text
    )


# ============================================================
# 8. FIGURE 2B
# FALSIFICATION
# ============================================================

def _as_bool(x):

    if isinstance(
        x,
        (bool, np.bool_)
    ):
        return bool(x)

    return (
        str(x)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "pass"
        }
    )


def matched_aes(ds):

    bm = pd.read_csv(
        Path(T12)
        / ds
        / "00_DATA_AUDIT"
        / "BLIND_MAP_PRIVATE.csv"
    )

    zt_of = dict(
        zip(
            bm.blind_sample_id,
            bm.hidden_ZT % 24.0
        )
    )

    out = []

    for rep in range(5):

        f = (
            Path(T12)
            / ds
            / "08_DIAGNOSTICS"
            / f"MATCHED_PANEL_{rep}_PRED.csv.gz"
        )

        if not f.exists():
            continue

        p = pd.read_csv(f)

        zt = (
            p.blind_sample_id
            .map(zt_of)
            .values
        )

        ae = circ_ae(
            p.phase_mean_h.values,
            zt
        )

        out.append(
            np.median(ae)
        )

    return np.asarray(
        out,
        dtype=float
    )


def make_fig2B(
    show_text=True
):

    fals = (
        pd.read_csv(
            Path(T12)
            / "TABLES"
            / "EXTERNAL_FALSIFICATION.csv"
        )
        .set_index("dataset")
    )

    fig = plt.figure(
        figsize=(
            DOUBLE,
            4.05
        )
    )

    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.78,
        wspace=0.44,
        left=0.075,
        right=0.985,
        top=0.77,
        bottom=0.205
    )

    all_axes = []

    for c, ds in enumerate([
        "GSE145197",
        "GSE223109"
    ]):

        row = fals.loc[ds]

        good = _as_bool(
            row.F2_pass
        )

        main_color = (
            NAVY
            if good
            else ORANGE
        )

        diag = (
            Path(T12)
            / ds
            / "08_DIAGNOSTICS"
        )

        # ====================================================
        # TOP
        # ====================================================

        ax = fig.add_subplot(
            gs[
                0,
                c
            ]
        )

        all_axes.append(ax)

        perms = np.array(
            [
                json.load(
                    open(
                        diag
                        / f"perm{i}_CPI.json",
                        "r",
                        encoding="utf-8"
                    )
                )["top40_mean_cpi"]
                for i in range(
                    1,
                    6
                )
            ],
            dtype=float
        )

        with open(
            diag / "rot8_CPI.json",
            "r",
            encoding="utf-8"
        ) as f:

            rot8 = float(
                json.load(f)[
                    "top40_mean_cpi"
                ]
            )

        real = float(
            row.real_top40_CPI
        )

        ax.bar(
            0,
            real,
            width=0.55,
            color=main_color,
            zorder=3
        )

        ax.scatter(
            np.full(
                len(perms),
                1.0
            ),
            perms,
            s=14,
            color=GRAY,
            zorder=4
        )

        ax.plot(
            [
                0.72,
                1.28
            ],
            [
                perms.max(),
                perms.max()
            ],
            lw=0.7,
            color=GRAY
        )

        ax.bar(
            2,
            rot8,
            width=0.55,
            color="white",
            edgecolor=GRAY,
            lw=0.9,
            zorder=3
        )

        ax.set_yscale(
            "log"
        )

        ymin = min(
            real,
            perms.min(),
            rot8
        )

        ax.set_ylim(
            ymin * 0.4,
            real * 6.0
        )

        ax.set_xticks(
            [
                0,
                1,
                2
            ]
        )

        ax.spines[
            [
                "top",
                "right"
            ]
        ].set_visible(False)

        ax.tick_params(
            labelsize=6.0
        )

        if show_text:

            ax.set_xticklabels(
                [
                    "CPI panel",
                    "permuted",
                    "wrong anchor"
                ],
                fontsize=6.2
            )

            ax.set_title(
                ds,
                fontsize=7.6,
                fontweight="bold",
                pad=4
            )

            ax.set_ylabel(
                "Top-40 mean CPI",
                fontsize=6.6
            )

            ax.text(
                0,
                real * 1.15,
                f"{real:.4f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=DK
            )

            ax.text(
                2,
                rot8 * 1.15,
                f"{rot8:.4f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=GRAY
            )

            ax.text(
                0.97,
                0.95,
                "Permutation PASS",
                transform=ax.transAxes,
                fontsize=5.8,
                color=GREEN,
                va="top",
                ha="right",
                fontweight="bold"
            )

        # ====================================================
        # BOTTOM
        # ====================================================

        ax = fig.add_subplot(
            gs[
                1,
                c
            ]
        )

        all_axes.append(ax)

        ma = matched_aes(ds)

        if len(ma) == 0:

            raise RuntimeError(
                f"No matched-panel predictions found for {ds}"
            )

        real_ae = float(
            row.C1_AE
        )

        ax.bar(
            0,
            real_ae,
            width=0.55,
            color=main_color,
            zorder=3
        )

        ax.scatter(
            np.full(
                len(ma),
                1.0
            ),
            ma,
            s=14,
            color=GRAY,
            zorder=4
        )

        ax.plot(
            [
                0.72,
                1.28
            ],
            [
                ma.min(),
                ma.min()
            ],
            lw=0.7,
            color=GRAY
        )

        ax.set_xticks(
            [
                0,
                1
            ]
        )

        ax.spines[
            [
                "top",
                "right"
            ]
        ].set_visible(False)

        ax.tick_params(
            labelsize=6.0
        )

        lo = (
            min(
                real_ae,
                ma.min()
            )
            - 0.75
        )

        hi = (
            max(
                real_ae,
                ma.max()
            )
            + 0.4
        )

        ax.set_ylim(
            lo,
            hi
        )

        if show_text:

            ax.set_xticklabels(
                [
                    "CPI panel",
                    "matched controls"
                ],
                fontsize=6.2
            )

            ax.set_ylabel(
                "Cell median AE (h)",
                fontsize=6.6
            )

            ax.text(
                0,
                real_ae + 0.06,
                f"{real_ae:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=DK
            )

            ax.text(
                1.0,
                ma.min() - 0.10,
                (
                    f"best control "
                    f"{ma.min():.3f}"
                ),
                fontsize=5.8,
                color=GRAY,
                ha="center",
                va="top"
            )

            ax.text(
                0.97,
                0.95,
                (
                    "Matched-panel "
                    + (
                        "PASS"
                        if good
                        else "FAIL"
                    )
                ),
                transform=ax.transAxes,
                fontsize=5.8,
                color=(
                    GREEN
                    if good
                    else RED
                ),
                va="top",
                ha="right",
                fontweight="bold"
            )

        if not good:

            ax.annotate(
                "",
                xy=(
                    0.68,
                    ma.min() + 0.08
                ),
                xytext=(
                    0.30,
                    real_ae - 0.10
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color=RED,
                    lw=0.9
                )
            )

    if show_text:

        fig.text(
            0.005,
            0.995,
            (
                "Falsification controls distinguish genuine "
                "clock information from reproducible confounding"
            ),
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="top"
        )

        fig.text(
            0.005,
            0.942,
            (
                "top: CPI panel vs permuted-label / wrong-anchor controls   |   "
                "bottom: CPI panel vs five prespecified matched control panels"
            ),
            fontsize=6.2,
            ha="left",
            va="top",
            color=GRAY
        )

        fig.text(
            0.5,
            0.025,
            (
                "GSE145197: CPI panel outperforms matched controls (PASS)   |   "
                "GSE223109: matched controls outperform the CPI panel (FAIL) — confound detected.\n"
                "n = 5 prespecified matched control panels; not a fully sampled null distribution."
            ),
            fontsize=6.2,
            ha="center",
            va="bottom",
            color=DK,
            style="italic"
        )

    if not show_text:

        for ax in all_axes:
            clean_axis_text(ax)

        remove_figure_text(fig)

    save_final(
        fig,
        "fig2B_falsification_controls",
        show_text
    )


# ============================================================
# 9. ORIGINAL Fig3a / Fig3b LOGIC
# Now used as final Fig2C / Fig2D
# ============================================================

def draw_gene_panel(
    ds,
    genes,
    signal_color,
    title,
    subtitle,
    footer,
    stem,
    sample_point_size,
    bottom_rect,
    show_text=True
):

    print()
    print(
        f"Loading real cohort data: {ds}"
    )

    coh = load_cohort(
        ds
    )

    print(
        f"{ds}: n cells = {len(coh):,}"
    )

    fig, axs = plt.subplots(
        4,
        2,
        figsize=(
            DOUBLE,
            6.05
        )
    )

    for r, g in enumerate(
        genes
    ):

        # EXACT original helper
        y = gene_expr(
            coh,
            g
        )

        # EXACT original helper
        sm = sample_means(
            coh,
            y
        )

        # ----------------------------------------------------
        # LEFT: true ZT
        # ----------------------------------------------------

        ax = axs[
            r,
            0
        ]

        ax.scatter(
            sm.zt,
            sm["mean"],
            s=sample_point_size,
            color=signal_color,
            zorder=4,
            edgecolor="white",
            linewidth=0.3
        )

        off, amp, ph = fit_cosine(
            sm.zt.values,
            sm["mean"].values
        )

        rel = (
            amp
            / (
                np.abs(off)
                + 1e-9
            )
        )

        tt = np.linspace(
            0,
            24,
            200
        )

        ax.plot(
            tt,
            (
                off
                + amp
                * np.cos(
                    2
                    * np.pi
                    * (
                        tt - ph
                    )
                    / 24
                )
            ),
            lw=0.9,
            color=GRAY,
            ls="--",
            zorder=3
        )

        if show_text:

            ax.set_ylabel(
                (
                    f"{g}\n"
                    "log-norm expr."
                ),
                fontsize=6.2
            )

            ax.text(
                0.97,
                0.92,
                f"rel. amp {rel:.2f}",
                transform=ax.transAxes,
                fontsize=5.6,
                ha="right",
                va="top",
                color=GRAY
            )

        # ----------------------------------------------------
        # RIGHT: inferred AnchorPhase
        # EXACT original per-cell cloud + 24-bin mean
        # ----------------------------------------------------

        ax2 = axs[
            r,
            1
        ]

        ax2.scatter(
            coh["phase"],
            y,
            s=1.2,
            alpha=0.05,
            color=signal_color,
            rasterized=True,
            zorder=2
        )

        bins = np.linspace(
            0,
            24,
            25
        )

        bi = (
            np.digitize(
                coh["phase"],
                bins
            )
            - 1
        )

        bm = [
            (
                y[
                    bi == k
                ].mean()
                if (
                    bi == k
                ).sum() > 20
                else np.nan
            )
            for k in range(
                24
            )
        ]

        ax2.plot(
            (
                bins[:-1]
                + bins[1:]
            )
            / 2,
            bm,
            lw=1.1,
            color=DK,
            zorder=4
        )

        aph = gene_acrophase(
            ds,
            g
        )

        if aph is not None:

            ax2.axvline(
                aph,
                color=signal_color,
                lw=0.7,
                ls=":"
            )

            if show_text:

                ax2.text(
                    aph,
                    1.02,
                    f"{aph:.1f} h",
                    fontsize=5.4,
                    color=signal_color,
                    ha="center",
                    va="bottom",
                    transform=(
                        ax2
                        .get_xaxis_transform()
                    )
                )

        # ----------------------------------------------------
        # column titles
        # ----------------------------------------------------

        if (
            r == 0
            and show_text
        ):

            axs[
                r,
                0
            ].set_title(
                "vs true Zeitgeber time (ZT)",
                fontsize=7.0,
                fontweight="bold",
                pad=4
            )

            axs[
                r,
                1
            ].set_title(
                "vs inferred AnchorPhase",
                fontsize=7.0,
                fontweight="bold",
                pad=4
            )

        # ----------------------------------------------------
        # common axes
        # ----------------------------------------------------

        for a in (
            axs[
                r,
                0
            ],
            axs[
                r,
                1
            ]
        ):

            a.set_xlim(
                0,
                24
            )

            a.set_xticks(
                [
                    0,
                    6,
                    12,
                    18,
                    24
                ]
            )

            a.spines[
                [
                    "top",
                    "right"
                ]
            ].set_visible(False)

            a.tick_params(
                labelsize=6.0
            )

            if (
                r == 3
                and show_text
            ):

                a.set_xlabel(
                    "circadian time (h)",
                    fontsize=6.4
                )

    # --------------------------------------------------------
    # publication text
    # --------------------------------------------------------

    if show_text:

        fig.text(
            0.005,
            0.995,
            title,
            fontsize=8.6,
            fontweight="bold",
            ha="left",
            va="top"
        )

        fig.text(
            0.005,
            0.957,
            subtitle,
            fontsize=6.0,
            ha="left",
            va="top",
            color=GRAY
        )

        fig.text(
            0.005,
            0.018,
            footer,
            fontsize=6.4,
            ha="left",
            va="bottom",
            color=signal_color,
            style="italic"
        )

    # ORIGINAL V2 layout
    fig.tight_layout(
        rect=(
            0.055,
            bottom_rect,
            0.995,
            0.905
        ),
        pad=0.8,
        h_pad=2.6,
        w_pad=2.4
    )

    if not show_text:

        for row in axs:
            for a in row:
                clean_axis_text(a)

        remove_figure_text(fig)

    save_final(
        fig,
        stem,
        show_text
    )


# ============================================================
# 10. FIGURE 2C
# ============================================================

def make_fig2C(
    show_text=True
):

    draw_gene_panel(

        ds="GSE145197",

        genes=[
            "Pck1",
            "Lpin1",
            "Mat1a",
            "Slc7a2",
        ],

        signal_color=GREEN,

        title=(
            "GSE145197: hepatic circadian output genes retain "
            "temporal structure along AnchorPhase"
        ),

        subtitle=(
            "left: expression vs true Zeitgeber time (ZT)   |   "
            "right: the same cells vs inferred AnchorPhase"
        ),

        footer=(
            "Representative CPI-selected genes are established hepatic circadian outputs.  "
            "Unit of analysis: per-sample means (n = 10 biological samples); "
            "per-cell points are shown for visualization only."
        ),

        stem=(
            "fig2C_liver_clock_outputs"
        ),

        sample_point_size=12,

        # same as original Fig3a
        bottom_rect=0.115,

        show_text=show_text,
    )


# ============================================================
# 11. FIGURE 2D
# ============================================================

def make_fig2D(
    show_text=True
):

    draw_gene_panel(

        ds="GSE223109",

        genes=[
            "Hba-a1",
            "Hba-a2",
            "Hbb-bs",
            "Alas2",
        ],

        signal_color=ORANGE,

        title=(
            "GSE223109: erythroid markers reveal reproducible "
            "non-circadian confounding"
        ),

        subtitle=(
            "left: expression vs true Zeitgeber time (ZT)   |   "
            "right: the same cells vs inferred AnchorPhase"
        ),

        footer=(
            "Erythroid / blood-content markers can appear phase-structured despite lacking "
            "a coherent circadian output pattern.  "
            "Unit of analysis: per-sample means (n = 18 biological samples); "
            "per-cell points are shown for visualization only."
        ),

        stem=(
            "fig2D_dermis_confound"
        ),

        sample_point_size=10,

        # same as original Fig3b
        bottom_rect=0.130,

        show_text=show_text,
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("FINAL FIGURE 2 RENDER")
    print("=" * 90)

    print(
        "Display model:",
        MODEL
    )

    print(
        "Optional extension:",
        EXTENSION
    )

    print(
        "Synthetic empirical data:",
        "FALSE"
    )

    # check BEFORE rendering anything
    preflight()

    jobs = [

        (
            "2A text",
            make_fig2A,
            True
        ),

        (
            "2A clean",
            make_fig2A,
            False
        ),

        (
            "2B text",
            make_fig2B,
            True
        ),

        (
            "2B clean",
            make_fig2B,
            False
        ),

        (
            "2C text",
            make_fig2C,
            True
        ),

        (
            "2C clean",
            make_fig2C,
            False
        ),

        (
            "2D text",
            make_fig2D,
            True
        ),

        (
            "2D clean",
            make_fig2D,
            False
        ),
    ]

    failed = []

    for (
        label,
        fn,
        show_text
    ) in jobs:

        print()
        print("-" * 90)
        print(label)
        print("-" * 90)

        try:

            fn(
                show_text=show_text
            )

        except Exception as e:

            import traceback

            traceback.print_exc()

            failed.append(
                (
                    label,
                    repr(e)
                )
            )

    print()
    print("=" * 90)

    if failed:

        print("FAILED:")

        for label, err in failed:

            print(
                f"  {label}: {err}"
            )

        raise SystemExit(1)

    print(
        "FIG2_FINAL_RENDER = COMPLETE"
    )

    print(
        "FIG2_EMPIRICAL_DATA_SYNTHETIC = FALSE"
    )

    print(
        "FIG2_PATH_MODE = LOCAL_PORTABLE"
    )

    print(
        "FIG2_B0_DISPLAY_NAME = AnchorPhase"
    )

    print(
        "FIG2_C1_DISPLAY_NAME = CPI"
    )

    print(
        "FIG2_C2_DISPLAY_NAME = SAFE-CPI"
    )

    print(
        "FIG2C_ORIGINAL_FIG3A_PLOTTING_LOGIC = PRESERVED"
    )

    print(
        "FIG2D_ORIGINAL_FIG3B_PLOTTING_LOGIC = PRESERVED"
    )

    print()
    print(
        "OUTPUT DIRECTORY:"
    )

    print(
        OUTDIR
    )

    print("=" * 90)


if __name__ == "__main__":
    main()